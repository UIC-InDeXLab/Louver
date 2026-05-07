// v1.32 Flash-Decoding-shaped sparse attention index kernel.
//
// Scope: replaces v1.31's `run_fused_attn_index_fp16q` (the heaviest Triton
// kernel). Keeps the existing Triton cluster_pass and fused reduce+buffer.
//
// Design:
//   - Grid (H_kv, NUM_SPLITS). Each CTA owns one kv-head and a contiguous
//     slice of parents.
//   - 1 warp per CTA (32 threads). Q for this kv-head's group rows is
//     loaded into SMEM ONCE at the start and reused across every parent
//     chunk.
//   - Per parent chunk (PARENTS_PER_CHUNK parents × BF children = COLS
//     keys): load gate bits, skip chunk if fully dead; otherwise cp.async
//     K/V tiles, compute S = Q @ K^T via mma.m16n8k16, mask/softmax,
//     accumulate O += P @ V.
//   - FP16 Q, K, V; FP32 softmax accumulators (m, l) and O fragment.
//   - Output: per-split partials (m, l, O). A downstream Triton kernel
//     reduces across splits + fuses the decoding buffer (reused from v1.31).
//
// Hardware: targets sm_80+. On sm_120 (RTX 5090) we use the same sm_80
// primitives — WGMMA/TMA would require a separate kernel path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int WARP_SIZE = 32;

// PTX wrappers ───────────────────────────────────────────────────────────────

__device__ __forceinline__ unsigned int cvta_to_shared_u32(const void* p) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"(p));
    return addr;
}

// 16-byte cp.async (cache-all). One thread copies 8 fp16 elements.
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    unsigned int dst = cvta_to_shared_u32(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   A: (16 × 16) fp16, row-major, loaded as 4 fp16x2 per thread
//   B: (16 × 8)  fp16, col-major, loaded as 2 fp16x2 per thread
//   D: (16 × 8)  fp32, per-thread 4 f32 outputs
__device__ __forceinline__ void mma_m16n8k16_row_col(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

// ldmatrix.sync.aligned.m8n8.x4.shared.b16 — fill A fragment (4 × b16x2).
__device__ __forceinline__ void ldmatrix_x4(unsigned int& a0, unsigned int& a1,
                                             unsigned int& a2, unsigned int& a3,
                                             unsigned int smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(smem_addr));
}

// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 — fill B fragment (2 × b16x2),
// transposed (col-major for mma).
__device__ __forceinline__ void ldmatrix_x2_trans(unsigned int& b0, unsigned int& b1,
                                                   unsigned int smem_addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
                 "{%0, %1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(smem_addr));
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, v, off);
        v = v > o ? v : o;
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// ────────────────────────────────────────────────────────────────────────────
// Main kernel
// ────────────────────────────────────────────────────────────────────────────

// Template parameters:
//   M_PAD   : padded query-row dim for the tensor-core tile (16).
//   D       : head dim (128).
//   D_V     : value head dim (128).
//   BF      : block factor / children per parent (4).
//   S       : number of subspaces (8).
//   PARENTS_PER_CHUNK : parents consumed per MMA chunk. COLS = PARENTS_PER_CHUNK*BF.
//   COLS    : must be 16 (single m16n8k16 pair covers N=16).
template <int M_PAD, int D, int D_V, int BF, int S, int PARENTS_PER_CHUNK, int COLS>
__global__ void sparse_attn_index_v1_32_kernel(
    const __half* __restrict__ Q,              // (H_q, D)
    const __half* __restrict__ KeysBlocksT,    // (H_kv, K, D, BF)
    const __half* __restrict__ ValuesBlocks,   // (H_kv, K, BF, D_v)
    const int16_t* __restrict__ AssignsBlocks, // (S, H_kv, K, BF)
    const int8_t* __restrict__ ClusterPass,    // (S, H_q, K)
    const int8_t* __restrict__ InvalidBlocks,  // (H_kv, K, BF)
    float* __restrict__ Out_M,                 // (H_q, NUM_SPLITS)
    float* __restrict__ Out_L,                 // (H_q, NUM_SPLITS)
    float* __restrict__ Out_O,                 // (H_q, NUM_SPLITS, D_v)
    int H_Q, int H_KV, int K, int NUM_SPLITS,
    int GROUPS, int ANCHOR_S, float SCALE) {

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;              // 0..31
    const int lane = tid;                     // 1 warp per CTA
    const int parents_per_split = (K + NUM_SPLITS - 1) / NUM_SPLITS;
    const int p_start = split * parents_per_split;
    const int p_end = min(p_start + parents_per_split, K);

    // SMEM layout ──────────────────────────────────────────────────────────
    extern __shared__ __align__(16) unsigned char smem[];
    __half* Q_smem = reinterpret_cast<__half*>(smem);                      // M_PAD*D
    __half* K_smem = Q_smem + M_PAD * D;                                   // COLS*D
    __half* V_smem = K_smem + COLS * D;                                    // COLS*D_V
    int8_t* gate_smem = reinterpret_cast<int8_t*>(V_smem + COLS * D_V);    // COLS

    // ── Load Q once ─────────────────────────────────────────────────────
    // Q rows [kvh*GROUPS ... kvh*GROUPS+GROUPS) get loaded; remaining
    // M_PAD-GROUPS rows are zero-padded.
    // Total elements: M_PAD * D = 16*128 = 2048 fp16. 32 threads × 64 iters.
    constexpr int Q_TOTAL = M_PAD * D;
    // Zero-init
    #pragma unroll
    for (int i = lane; i < Q_TOTAL; i += WARP_SIZE) {
        Q_smem[i] = __float2half(0.f);
    }
    __syncwarp();
    // Real rows (8 fp16 per cp.async)
    const int q_row_base = kvh * GROUPS;
    #pragma unroll
    for (int g = 0; g < GROUPS; ++g) {
        const int hq = q_row_base + g;
        // Each of the 32 lanes loads 4 halfs → 32*4 = 128 = D.
        const int idx = lane * 4;
        if (idx < D) {
            ((uint2*)&Q_smem[g * D + idx])[0] = ((const uint2*)&Q[hq * D + idx])[0];
        }
    }
    __syncwarp();

    // ── Online softmax state per query row (M_PAD total).
    // Thread holds fragments for specific rows. The m16n8k16 layout puts
    // thread t on rows {t/4, t/4+8}. Each thread owns state for 2 rows.
    float m_acc[2];
    float l_acc[2];
    m_acc[0] = -1.0e30f; m_acc[1] = -1.0e30f;
    l_acc[0] = 0.f;     l_acc[1] = 0.f;

    // O accumulator: each thread holds (2 rows × 4 fp32) per 16-column
    // tile of O. With D_V=128 we have 8 tiles; each tile has 4 outputs
    // per thread but only 2 rows matter here — wait:
    // mma.m16n8k16 output D layout: thread t holds {(r=t/4, c=(t%4)*2),
    //                                                (r=t/4, c=(t%4)*2+1),
    //                                                (r=t/4+8, c=(t%4)*2),
    //                                                (r=t/4+8, c=(t%4)*2+1)}
    // So 4 outputs per N=8 tile. For N=16 we do 2 tiles = 8 outputs.
    // For O with D_V=128 we have 128/8 = 16 N-tiles. Each tile has 4 outputs
    // per thread. Total O per thread: 16 * 4 = 64 fp32.
    // Layout: o_acc[n_tile][i] where i in {0..3} corresponds to (r0,c0),(r0,c1),(r1,c0),(r1,c1).
    constexpr int DV_TILES = D_V / 8;
    float o_acc[DV_TILES][4];
    #pragma unroll
    for (int t = 0; t < DV_TILES; ++t) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[t][i] = 0.f;
    }

    // Which rows does this thread own for softmax bookkeeping?
    const int my_row0 = lane / 4;          // 0..7
    const int my_row1 = my_row0 + 8;       // 8..15
    const bool row0_valid = my_row0 < GROUPS;
    const bool row1_valid = my_row1 < GROUPS;
    // Global q-head indices for my two rows
    const int my_hq0 = q_row_base + my_row0;
    const int my_hq1 = q_row_base + my_row1;

    // ── Parent chunk loop ─────────────────────────────────────────────────
    constexpr int D_TILES = D / 16;                  // 8 K-loop tiles for score compute
    for (int p_chunk_start = p_start; p_chunk_start < p_end; p_chunk_start += PARENTS_PER_CHUNK) {

        // Build gate bits: survive[col] = anchor_pass[hq,p] & !invalid[kvh,p,c] &
        //                                  AND over s!=ANCHOR of cp[s,hq,assigns[s,kvh,p,c]]
        // COLS = PARENTS_PER_CHUNK*BF. Each of 32 lanes covers up to 2 cols
        // (but COLS=16 < 32, so lanes 0..15 handle one col each; others idle).
        // But survive is per-(row, col) since we AND per-hq cluster_pass.
        // Simplification: compute a per-(row,col) mask only for anchor + per-col
        // validity; cross-subspace checks reduce per-col assignment to per-(row,col)
        // gate.
        //
        // To keep the inner loop simple, we compute a per-(row,col) i1 mask (16*16=256)
        // → store as 16 × 16-bit rows in SMEM. But that balloons.
        //
        // Simpler: compute the gate per-(row,col) inside each thread using 2 cols
        // per thread (lane → cols 2*lane, 2*lane+1). Each thread holds per-col
        // anchor_pass[row]. Cross-subspace checks per-col per-row: reload.
        //
        // To avoid repeated loads, we pack into int bitmask: 16 rows × 16 cols = 256 bits.
        // Per thread: hold 8 bits (one u8) covering rows {my_row0, my_row1} × 4 cols.
        // Simplification: just materialize a byte mask survive_smem[16 rows][16 cols].

        // Use gate_smem[0..COLS-1] to mark "column has a valid child & any row passes".
        // And we need a per-(row,col) mask too. Put both in SMEM: we reuse
        // gate_smem[0..COLS-1] as "any row survives" bit, and compute per-(row,col)
        // mask packed as 32-bit per row in a small SMEM block.
        // With COLS=16, a 16-bit packed mask per row fits. We'll stash it into
        // an int[M_PAD] block right after gate_smem.

        int* survive_mask = reinterpret_cast<int*>(gate_smem + COLS);  // [M_PAD]

        if (lane < M_PAD) survive_mask[lane] = 0;
        if (lane < COLS) gate_smem[lane] = 0;
        __syncwarp();

        // Each lane handles one col (COLS=16, lanes 0..15).
        if (lane < COLS) {
            const int c = lane;
            const int parent_off = c / BF;
            const int child_off = c % BF;
            const int parent = p_chunk_start + parent_off;
            int anychild_any_row = 0;
            if (parent < p_end) {
                // invalid
                int inv = (int)InvalidBlocks[(kvh * K + parent) * BF + child_off];
                if (inv == 0) {
                    int row_mask_accum = 0;
                    // For each row check anchor + all non-anchor subspaces
                    #pragma unroll
                    for (int r = 0; r < M_PAD; ++r) {
                        if (r >= GROUPS) continue;
                        const int hq = q_row_base + r;
                        int anchor_pass = (int)ClusterPass[(ANCHOR_S * H_Q + hq) * K + parent];
                        if (!anchor_pass) continue;
                        bool all_pass = true;
                        #pragma unroll
                        for (int s = 0; s < S; ++s) {
                            if (s == ANCHOR_S) continue;
                            int assign =
                                (int)AssignsBlocks[((s * H_KV + kvh) * K + parent) * BF + child_off];
                            int p2 = (int)ClusterPass[(s * H_Q + hq) * K + assign];
                            if (!p2) { all_pass = false; break; }
                        }
                        if (all_pass) {
                            row_mask_accum |= (1 << r);
                        }
                    }
                    // Merge into survive_mask[r]: need atomic/OR across lanes.
                    // Instead, write column bits: survive_mask[r] |= (row_mask_accum bit r) << c.
                    #pragma unroll
                    for (int r = 0; r < M_PAD; ++r) {
                        if (row_mask_accum & (1 << r)) {
                            atomicOr(&survive_mask[r], 1 << c);
                        }
                    }
                    anychild_any_row = (row_mask_accum != 0) ? 1 : 0;
                }
            }
            gate_smem[c] = (int8_t)anychild_any_row;
        }
        __syncwarp();

        // Skip entire chunk if no column has any survivor
        int chunk_any = 0;
        if (lane < COLS) chunk_any = (int)gate_smem[lane];
        chunk_any = __any_sync(0xffffffff, chunk_any != 0);
        if (!chunk_any) continue;

        // ── Load K tile: (COLS, D) fp16 = 16*128*2 = 4KB.
        // Source layout: KeysBlocksT[kvh, parent, d, child]. With COLS=16 = 4*4,
        // we scatter loads.
        // Each lane loads 16 halfs = 32B. 16 lanes × 32B = 512B per iter. 8 iters → 4KB.
        // Iterate cols then d.
        // Flatten: total 2048 halfs. 32 lanes × 64 halfs. Per lane: 64 halfs = 16 iterations of 4 halfs (uint2).
        #pragma unroll
        for (int i = 0; i < M_PAD * D / (WARP_SIZE * 4); ++i) {
            // Wait we want COLS*D = 16*128 = 2048. 32 lanes × 4 halfs = 128 per iter. 16 iters.
            int flat = (i * WARP_SIZE + lane) * 4;
            if (flat >= COLS * D) break;
            int c = flat / D;
            int d = flat % D;
            int parent = p_chunk_start + (c / BF);
            int child = c % BF;
            __half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
            if (gate_smem[c] && parent < p_end) {
                // Contiguous load along d: d is innermost?
                // Layout is (H_kv, K, D, BF) — so at fixed (parent, d, child), element
                // is contiguous in child-dim (stride 1). To load 4 along d, stride is BF.
                // That's not contiguous in d. To make this efficient we load along BF
                // direction instead — but we have one child. So loads are strided.
                // For correctness, do 4 scalar loads.
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int dj = d + j;
                    ((__half*)&K_smem[c * D + d])[j] =
                        KeysBlocksT[((kvh * K + parent) * D + dj) * BF + child];
                }
            } else {
                ((uint2*)&K_smem[c * D + d])[0] = ((uint2*)&zeros[0])[0];
            }
        }
        // Also load V tile: (COLS, D_V) fp16 = 16*128*2 = 4KB. Layout (H_kv, K, BF, D_v).
        // Contiguous along D_v for fixed (parent, child).
        #pragma unroll
        for (int i = 0; i < COLS * D_V / (WARP_SIZE * 4); ++i) {
            int flat = (i * WARP_SIZE + lane) * 4;
            int c = flat / D_V;
            int d = flat % D_V;
            int parent = p_chunk_start + (c / BF);
            int child = c % BF;
            __half zeros[4] = {__float2half(0.f), __float2half(0.f), __float2half(0.f), __float2half(0.f)};
            if (gate_smem[c] && parent < p_end) {
                ((uint2*)&V_smem[c * D_V + d])[0] =
                    ((uint2*)&ValuesBlocks[((kvh * K + parent) * BF + child) * D_V + d])[0];
            } else {
                ((uint2*)&V_smem[c * D_V + d])[0] = ((uint2*)&zeros[0])[0];
            }
        }
        __syncwarp();

        // ── Compute scores S = Q @ K^T via mma.m16n8k16.
        // We want S[16, 16] fp32. Do two m16n8k16 tiles for N=0..7 and N=8..15.
        float s[2][4];  // s[n_tile][0..3] matches mma D output
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt][0] = 0.f; s[nt][1] = 0.f; s[nt][2] = 0.f; s[nt][3] = 0.f;
        }

        #pragma unroll
        for (int kt = 0; kt < D_TILES; ++kt) {
            // Load A fragment: 16 rows × 16 cols of Q from Q_smem[row][kt*16 + col]
            // Using ldmatrix.x4: 32 lanes form 4 8×8 tiles.
            // A layout: Q_smem has row-major (M_PAD rows × D cols). For ldmatrix x4,
            // address is for 8×8 tiles. Thread provides address of row based on lane:
            //   lane t: row = t%16, col = 0 for the 16x16 tile → address = &Q_smem[row * D + kt*16]
            unsigned int a0, a1, a2, a3;
            unsigned int q_row = lane % 16;
            unsigned int q_col = (lane / 16) * 8;
            unsigned int q_addr = cvta_to_shared_u32(&Q_smem[q_row * D + kt * 16 + q_col]);
            ldmatrix_x4(a0, a1, a2, a3, q_addr);

            // For each N tile:
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                // B is K^T, i.e., (K, N) fp16, col-major with respect to (K, N).
                // But our K_smem is (COLS, D) = (N, K) row-major → this IS K^T in mma terms.
                // For ldmatrix.x2.trans: each lane provides addr for its 8×8 tile row.
                // For a 16×8 B tile (K=16, N=8): two 8×8 subtiles stacked in K dim.
                // lane t: row = t%8 (within 0..7) actually ldmatrix_x2 wants 2 x 8×8 tiles.
                // Use .trans since we want K in rows, N in cols, but stored (N,K).
                unsigned int b0, b1;
                unsigned int k_row = lane % 8;           // K row 0..7
                unsigned int k_addr_base = kt * 16;
                // Two 8x8 tiles: first k=0..7, second k=8..15. N fixed at nt*8..nt*8+7.
                unsigned int n_col = nt * 8 + (lane / 8) * 8;  // lane/8 chooses 0 or 1 → col offset
                // Actually ldmatrix.x2 wants 2 tiles. Each lane provides one address per tile.
                // For K×N tile with K=16, N=8: subtile0 = K 0..7, N 0..7; subtile1 = K 8..15, N 0..7.
                // Lanes 0..7 address subtile0, lanes 8..15 address subtile1; lanes 16..31 mirror lanes 0..15.
                // Address: K_smem[n][k] = K_smem[nt*8 + n_in_tile][kt*16 + k_in_tile]
                // For lane t in 0..7: this lane is K-row t, N=0..7 (tile 0's row t).
                //   → address is &K_smem[nt*8 + 0][kt*16 + t]  (wrong — need 8 contiguous N)
                // Actually ldmatrix gives 8 contiguous cols of one row per lane.
                // Correct layout for x2.trans:
                //   ldmatrix.m8n8.x2.trans wants: each lane provides the address of 1 row of a
                //   source 8×8 matrix stored in (row, col) layout. But trans means the fragment
                //   returned is the transpose: we provide (8 rows) × (8 cols) in row-major, get
                //   fragments suitable for B (col-major).
                // For our case: we want B = K_tile^T. Source = K_smem[n_range, k_range] = 8 rows
                // (N) × 16 cols (K). For the N=nt tile and K subset:
                //   subtile0 (k 0..7): addresses K_smem[nt*8 + r][kt*16 + 0..7] for r in 0..7.
                //   subtile1 (k 8..15): addresses K_smem[nt*8 + r][kt*16 + 8..15] for r in 0..7.
                // Lane t in 0..7 → subtile0, row t
                // Lane t in 8..15 → subtile1, row t-8
                // Lane t in 16..31 → doesn't matter for x2; unused.
                int n_row_for_lane = nt * 8 + (lane % 8);
                int k_col_for_lane = kt * 16 + ((lane / 8) & 1) * 8;
                unsigned int b_addr = cvta_to_shared_u32(&K_smem[n_row_for_lane * D + k_col_for_lane]);
                ldmatrix_x2_trans(b0, b1, b_addr);

                mma_m16n8k16_row_col(
                    s[nt][0], s[nt][1], s[nt][2], s[nt][3],
                    a0, a1, a2, a3,
                    b0, b1,
                    s[nt][0], s[nt][1], s[nt][2], s[nt][3]);
            }
        }

        // ── Apply scale, gate mask, and compute per-row max/sum.
        // Each thread owns 4 outputs per N tile:
        //   (r=lane/4,   c=nt*8 + (lane%4)*2)
        //   (r=lane/4,   c=nt*8 + (lane%4)*2 + 1)
        //   (r=lane/4+8, c=nt*8 + (lane%4)*2)
        //   (r=lane/4+8, c=nt*8 + (lane%4)*2 + 1)
        float my_m[2] = {m_acc[0], m_acc[1]};
        float my_scores[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int col_base = nt * 8 + (lane % 4) * 2;
            int mask_row0 = survive_mask[my_row0];
            int mask_row1 = survive_mask[my_row1];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r_is_1 = (i / 2);   // 0 or 1
                int c_off = (i & 1);
                int col = col_base + c_off;
                int mask_r = r_is_1 ? mask_row1 : mask_row0;
                bool r_valid = r_is_1 ? row1_valid : row0_valid;
                bool alive = r_valid && (mask_r & (1 << col));
                float v = s[nt][i] * SCALE;
                my_scores[nt][i] = alive ? v : -1.0e30f;
            }
        }

        // Per-row max reduction. Each thread holds pieces of rows {my_row0, my_row1}.
        // Within a warp, rows are spread across lanes: 4 lanes share a row (same lane/4).
        float row0_max = -1.0e30f;
        float row1_max = -1.0e30f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            row0_max = fmaxf(row0_max, fmaxf(my_scores[nt][0], my_scores[nt][1]));
            row1_max = fmaxf(row1_max, fmaxf(my_scores[nt][2], my_scores[nt][3]));
        }
        // Reduce across the 4 lanes that share the row (xor 1, 2 within the 4-lane group).
        row0_max = fmaxf(row0_max, __shfl_xor_sync(0xffffffff, row0_max, 1));
        row0_max = fmaxf(row0_max, __shfl_xor_sync(0xffffffff, row0_max, 2));
        row1_max = fmaxf(row1_max, __shfl_xor_sync(0xffffffff, row1_max, 1));
        row1_max = fmaxf(row1_max, __shfl_xor_sync(0xffffffff, row1_max, 2));

        // New running max
        float new_m0 = fmaxf(m_acc[0], row0_max);
        float new_m1 = fmaxf(m_acc[1], row1_max);
        float alpha0 = __expf(m_acc[0] - new_m0);
        float alpha1 = __expf(m_acc[1] - new_m1);

        // P = exp(S - new_m). Compute per thread.
        float p_frag[2][4];
        float l_partial0 = 0.f, l_partial1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r_is_1 = (i / 2);
                float m_ref = r_is_1 ? new_m1 : new_m0;
                float v = __expf(my_scores[nt][i] - m_ref);
                // Fill zero if input was -inf (sentinel)
                if (my_scores[nt][i] <= -1.0e29f) v = 0.f;
                p_frag[nt][i] = v;
                if (r_is_1) l_partial1 += v; else l_partial0 += v;
            }
        }
        // Reduce sum across the 4-lane row group
        l_partial0 += __shfl_xor_sync(0xffffffff, l_partial0, 1);
        l_partial0 += __shfl_xor_sync(0xffffffff, l_partial0, 2);
        l_partial1 += __shfl_xor_sync(0xffffffff, l_partial1, 1);
        l_partial1 += __shfl_xor_sync(0xffffffff, l_partial1, 2);

        l_acc[0] = alpha0 * l_acc[0] + l_partial0;
        l_acc[1] = alpha1 * l_acc[1] + l_partial1;
        m_acc[0] = new_m0;
        m_acc[1] = new_m1;

        // O *= alpha per row
        #pragma unroll
        for (int t = 0; t < DV_TILES; ++t) {
            o_acc[t][0] *= alpha0;
            o_acc[t][1] *= alpha0;
            o_acc[t][2] *= alpha1;
            o_acc[t][3] *= alpha1;
        }

        // ── Convert P to fp16 and compute O += P @ V via mma.m16n8k16.
        // A (P) layout for mma: 16 rows × 16 cols fp16.
        // Pack my 4 p_frag per N tile → build A fragment for next mma.
        // Each thread's four mma output values (r0c0, r0c1, r1c0, r1c1) correspond to the
        // matching positions in the A input fragment.
        // The A fragment for m16n8k16 uses 4 fp16x2 = 8 fp16 per thread:
        //   {A[t/4,     (t%4)*2], A[t/4,     (t%4)*2+1],
        //    A[t/4+8,   (t%4)*2], A[t/4+8,   (t%4)*2+1],
        //    A[t/4,     (t%4)*2+8], A[t/4,     (t%4)*2+9],
        //    A[t/4+8,   (t%4)*2+8], A[t/4+8,   (t%4)*2+9]}
        // i.e., K dim split into 0..7 and 8..15. Our p_frag[0][*] covers K=0..7 (N=0..7 in S)
        // and p_frag[1][*] covers K=8..15 (N=8..15 in S).
        unsigned int a0p, a1p, a2p, a3p;
        {
            __half hp[8];
            hp[0] = __float2half(p_frag[0][0]);  // r0 c0
            hp[1] = __float2half(p_frag[0][1]);  // r0 c1
            hp[2] = __float2half(p_frag[0][2]);  // r1 c0
            hp[3] = __float2half(p_frag[0][3]);  // r1 c1
            hp[4] = __float2half(p_frag[1][0]);
            hp[5] = __float2half(p_frag[1][1]);
            hp[6] = __float2half(p_frag[1][2]);
            hp[7] = __float2half(p_frag[1][3]);
            a0p = ((unsigned int*)&hp[0])[0];
            a1p = ((unsigned int*)&hp[2])[0];
            a2p = ((unsigned int*)&hp[4])[0];
            a3p = ((unsigned int*)&hp[6])[0];
        }

        // For each N tile of O (D_V split into 8-col tiles), compute O[16, 8] += P @ V.
        // V layout in SMEM: V_smem[COLS=16][D_V=128] row-major → V is (K=COLS=16, N=D_V=128).
        // For mma: B = V, row-major with K in rows. We want B (K=16, N=8) per tile.
        // Use ldmatrix.x2.trans on V_smem[k_row][d_tile*8 + n_col]. Actually V is already in
        // the right shape for mma (B is K-major for mma.row.col: A row-major, B col-major).
        // V_smem is (K=16, N=D_V=128) row-major → for B col-major (K, N) we need transpose.
        // x2.trans gives transpose of the SMEM fragment → fragments suitable for col-major B.
        #pragma unroll
        for (int dvt = 0; dvt < DV_TILES; ++dvt) {
            unsigned int b0v, b1v;
            // Source: V_smem[0..15][dvt*8 + 0..7] = 16 K-rows × 8 N-cols.
            // Subtile split for x2: K 0..7 and K 8..15, same N.
            int v_row = lane % 8;
            int k_start = (lane / 8) & 1 ? 8 : 0;
            unsigned int v_addr = cvta_to_shared_u32(&V_smem[(k_start + v_row) * D_V + dvt * 8]);
            ldmatrix_x2_trans(b0v, b1v, v_addr);

            mma_m16n8k16_row_col(
                o_acc[dvt][0], o_acc[dvt][1], o_acc[dvt][2], o_acc[dvt][3],
                a0p, a1p, a2p, a3p,
                b0v, b1v,
                o_acc[dvt][0], o_acc[dvt][1], o_acc[dvt][2], o_acc[dvt][3]);
        }
    }

    // ── Write per-split partials ─────────────────────────────────────────
    // Each thread owns 2 rows (my_row0, my_row1). Within 4-lane row groups,
    // lane%4==0 is the "leader" who writes m, l for the row.
    if ((lane % 4) == 0) {
        if (row0_valid) {
            Out_M[my_hq0 * NUM_SPLITS + split] = m_acc[0];
            Out_L[my_hq0 * NUM_SPLITS + split] = l_acc[0];
        }
        if (row1_valid) {
            Out_M[my_hq1 * NUM_SPLITS + split] = m_acc[1];
            Out_L[my_hq1 * NUM_SPLITS + split] = l_acc[1];
        }
    }
    // Write O: each thread owns 4 outputs per DV tile at (my_row0/1, col_base, col_base+1).
    #pragma unroll
    for (int dvt = 0; dvt < DV_TILES; ++dvt) {
        int col_base = dvt * 8 + (lane % 4) * 2;
        if (row0_valid) {
            Out_O[(my_hq0 * NUM_SPLITS + split) * D_V + col_base + 0] = o_acc[dvt][0];
            Out_O[(my_hq0 * NUM_SPLITS + split) * D_V + col_base + 1] = o_acc[dvt][1];
        }
        if (row1_valid) {
            Out_O[(my_hq1 * NUM_SPLITS + split) * D_V + col_base + 0] = o_acc[dvt][2];
            Out_O[(my_hq1 * NUM_SPLITS + split) * D_V + col_base + 1] = o_acc[dvt][3];
        }
    }
}

}  // anonymous namespace

// Host-side launcher
//
// Fixed template instantiation for the benchmark's shape: D=128, D_V=128,
// BF=4, S=8, COLS=16 (= PARENTS_PER_CHUNK=4 * BF=4), M_PAD=16.

void sparse_attn_index_v1_32(
    torch::Tensor q,                // (H_q, D) fp16
    torch::Tensor keys_t,           // (H_kv, K, D, BF) fp16
    torch::Tensor values,           // (H_kv, K, BF, D_v) fp16
    torch::Tensor assigns,          // (S, H_kv, K, BF) int16
    torch::Tensor cluster_pass,     // (S, H_q, K) int8
    torch::Tensor invalid,          // (H_kv, K, BF) int8
    torch::Tensor out_m,            // (H_q, NUM_SPLITS) fp32
    torch::Tensor out_l,            // (H_q, NUM_SPLITS) fp32
    torch::Tensor out_o,            // (H_q, NUM_SPLITS, D_v) fp32
    int64_t h_q,
    int64_t h_kv,
    int64_t k_parents,
    int64_t num_splits,
    int64_t groups,
    int64_t anchor_s,
    double scale) {

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(keys_t.dtype() == torch::kFloat16, "keys_t must be fp16");
    TORCH_CHECK(values.dtype() == torch::kFloat16, "values must be fp16");
    TORCH_CHECK(assigns.dtype() == torch::kInt16, "assigns must be int16");
    TORCH_CHECK(cluster_pass.dtype() == torch::kInt8, "cluster_pass must be int8");
    TORCH_CHECK(invalid.dtype() == torch::kInt8, "invalid must be int8");
    TORCH_CHECK(out_m.dtype() == torch::kFloat32, "out_m must be fp32");
    TORCH_CHECK(out_l.dtype() == torch::kFloat32, "out_l must be fp32");
    TORCH_CHECK(out_o.dtype() == torch::kFloat32, "out_o must be fp32");

    constexpr int M_PAD = 16;
    constexpr int D = 128;
    constexpr int D_V = 128;
    constexpr int BF = 4;
    constexpr int S = 8;
    constexpr int PARENTS_PER_CHUNK = 4;
    constexpr int COLS = PARENTS_PER_CHUNK * BF;  // 16

    // SMEM bytes: Q + K + V + gate + survive_mask[M_PAD]
    size_t smem_bytes = (M_PAD * D + COLS * D + COLS * D_V) * sizeof(__half)
                      + COLS * sizeof(int8_t) + M_PAD * sizeof(int);

    dim3 grid((int)h_kv, (int)num_splits, 1);
    dim3 block(32, 1, 1);

    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_attn_index_v1_32_kernel<M_PAD, D, D_V, BF, S, PARENTS_PER_CHUNK, COLS>
        <<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(keys_t.data_ptr()),
            reinterpret_cast<const __half*>(values.data_ptr()),
            reinterpret_cast<const int16_t*>(assigns.data_ptr()),
            reinterpret_cast<const int8_t*>(cluster_pass.data_ptr()),
            reinterpret_cast<const int8_t*>(invalid.data_ptr()),
            out_m.data_ptr<float>(),
            out_l.data_ptr<float>(),
            out_o.data_ptr<float>(),
            (int)h_q, (int)h_kv, (int)k_parents, (int)num_splits,
            (int)groups, (int)anchor_s, (float)scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attn_index_v1_32", &sparse_attn_index_v1_32,
          "v1.32 sparse attention index kernel (CUDA, tensor cores)");
}
