// v1.46 — Native CUDA fused anchor-gate attention kernel.
//
// Architecture: warp-per-head (MAX_GROUPS warps/CTA).
// Lane-per-column Q@K^T: each of 32 lanes owns COLS/32 columns,
// computing full D-dimensional dot products with NO warp_reduce.
// K_PAD=8 halfs per K-row to reduce smem bank conflicts (4-way).
// exp2f for softmax, cp.async for K/V loads, inline anchor gate.
// Grid: (H_kv, NUM_SPLITS).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>
#include <cstdint>

namespace {

constexpr int WARP_SIZE = 32;
constexpr float NEG_INF = -1.0e30f;
constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ unsigned int cvta_to_shared_u32(const void* p) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr) : "l"(p));
    return addr;
}

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    unsigned int dst = cvta_to_shared_u32(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_group_1() {
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// ──────────────────────────────────────────────────────────────────────
// Main kernel: lane-per-column Q@K^T, warp-per-head
// ──────────────────────────────────────────────────────────────────────
// Each warp handles one GQA query head.
// For Q@K^T, each lane independently computes dot products for its
// assigned columns (COLS/32 columns per lane), iterating over the full
// D dimension — no warp_reduce_sum needed for scores.
// For P@V, lanes switch to dim-parallel (4 output dims per lane).

template <int D, int D_V, int BF, int MAX_GROUPS, int AW, int COLS>
__global__ void __launch_bounds__(MAX_GROUPS * WARP_SIZE, 1)
sparse_attn_v1_46_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ KeysBlocks,     // (H_kv, K, BF, D)
    const __half* __restrict__ ValuesBlocks,   // (H_kv, K, BF, D_v)
    const __half* __restrict__ CentersAnchor,  // (H_kv, K, AW)
    const __half* __restrict__ RadiiAnchor,    // (H_kv, K)
    const __half* __restrict__ ThAnchor,       // (H_q,)
    const __half* __restrict__ QNormAnchor,    // (H_q,)
    const int8_t* __restrict__ InvalidBlocks,  // (H_kv, K, BF)
    float* __restrict__ Out_M,
    float* __restrict__ Out_L,
    float* __restrict__ Out_O,
    int H_Q, int K, int NUM_SPLITS, int GROUPS,
    int DIM_OFFSET, float SCALE) {

    constexpr int PPC = COLS / BF;
    constexpr int CPL = COLS / WARP_SIZE;  // cols per lane (4 for COLS=128)
    constexpr int K_PAD = 8;               // halfs padding per K row
    constexpr int K_STRIDE = D + K_PAD;    // 136 for D=128
    constexpr int BLOCK = MAX_GROUPS * WARP_SIZE;

    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    const int parents_per_split = (K + NUM_SPLITS - 1) / NUM_SPLITS;
    const int p_start = split * parents_per_split;
    const int p_end = min(p_start + parents_per_split, K);
    const int q_row_base = kvh * GROUPS;
    const bool row_valid = warp_id < GROUPS;
    const int hq = q_row_base + warp_id;

    // ── Shared memory layout ────────────────────────────────────────
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half*  q_smem   = reinterpret_cast<__half*>(smem_raw);
    // Q: (MAX_GROUPS, D)   — broadcast access, no padding needed
    __half*  k_smem   = q_smem + MAX_GROUPS * D;
    // K: (COLS, K_STRIDE)  — padded for bank-conflict reduction
    __half*  v_smem   = k_smem + COLS * K_STRIDE;
    // V: (COLS, D_V)       — 2-way conflict acceptable
    __half*  anc_smem = v_smem + COLS * D_V;
    // Anchor centers: (PPC, AW)
    float*   rad_smem = reinterpret_cast<float*>(anc_smem + PPC * AW);
    // Anchor radii: (PPC,)
    int8_t*  inv_smem = reinterpret_cast<int8_t*>(rad_smem + PPC);
    // Invalid blocks: (COLS,)
    int8_t*  col_live = inv_smem + COLS;
    // Per-column liveness: (COLS,)
    int8_t*  row_gate = col_live + COLS;
    // Per-(head, parent) gate: (MAX_GROUPS * PPC), aligned
    __half*  p_smem   = reinterpret_cast<__half*>(
        row_gate + ((MAX_GROUPS * PPC + 7) & ~7));
    // P values for P@V: (MAX_GROUPS, COLS)

    // ── Load Q to smem ──────────────────────────────────────────────
    constexpr int Q_PACKS = D / 8;
    for (int i = tid; i < GROUPS * Q_PACKS; i += BLOCK) {
        int row = i / Q_PACKS, pack = i % Q_PACKS;
        reinterpret_cast<uint4*>(&q_smem[row * D])[pack] =
            reinterpret_cast<const uint4*>(&Q[(q_row_base + row) * D])[pack];
    }
    for (int i = tid; i < (MAX_GROUPS - GROUPS) * Q_PACKS; i += BLOCK) {
        int row = GROUPS + i / Q_PACKS, pack = i % Q_PACKS;
        if (row < MAX_GROUPS)
            reinterpret_cast<uint4*>(&q_smem[row * D])[pack] = make_uint4(0,0,0,0);
    }
    __syncthreads();

    // ── Per-warp accumulators ───────────────────────────────────────
    float m_acc = NEG_INF, l_acc = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float my_th = row_valid ? __half2float(ThAnchor[hq]) : 1e30f;
    float my_qn = row_valid ? __half2float(QNormAnchor[hq]) : 0.0f;

    // ── Main loop over parent chunks ────────────────────────────────
    for (int pc = p_start; pc < p_end; pc += PPC) {
        const int chunk_p = min(PPC, p_end - pc);
        const int chunk_cols = chunk_p * BF;

        // Load anchor centers, radii, invalid flags
        for (int i = tid; i < chunk_p * AW; i += BLOCK) {
            int pr = i / AW, w = i % AW;
            anc_smem[pr * AW + w] = CentersAnchor[(kvh * K + pc + pr) * AW + w];
        }
        for (int i = tid; i < chunk_p; i += BLOCK)
            rad_smem[i] = __half2float(RadiiAnchor[kvh * K + pc + i]);
        for (int c = tid; c < COLS; c += BLOCK) {
            inv_smem[c] = (c < chunk_cols)
                ? InvalidBlocks[(kvh * K + pc + c/BF) * BF + c%BF] : 1;
            col_live[c] = 0;
        }
        for (int i = tid; i < MAX_GROUPS * PPC; i += BLOCK)
            row_gate[i] = 0;
        __syncthreads();

        // ── Anchor gate per warp ────────────────────────────────────
        if (row_valid) {
            for (int pr = 0; pr < chunk_p; ++pr) {
                float dot = 0.0f;
                for (int w = lane; w < AW; w += WARP_SIZE)
                    dot += __half2float(q_smem[warp_id * D + DIM_OFFSET + w])
                         * __half2float(anc_smem[pr * AW + w]);
                dot = warp_reduce_sum(dot);
                float ub = dot + rad_smem[pr] * my_qn;
                int ok = __shfl_sync(0xffffffff, (ub >= my_th) ? 1 : 0, 0);
                if (ok) {
                    if (lane == 0) row_gate[warp_id * PPC + pr] = 1;
                    if (lane < BF) {
                        int c = pr * BF + lane;
                        if (c < chunk_cols && inv_smem[c] == 0)
                            col_live[c] = 1;
                    }
                }
            }
        }
        __syncthreads();

        // Skip entirely dead chunks
        {
            int any = 0;
            for (int c = lane; c < chunk_cols; c += WARP_SIZE)
                if (col_live[c]) any = 1;
            if (!__any_sync(0xffffffff, any)) { __syncthreads(); continue; }
        }

        // ── Load K via cp.async (group 0) ──────────────────────────
        constexpr int K_PACKS = D / 8;
        for (int i = tid; i < chunk_cols * K_PACKS; i += BLOCK) {
            int col = i / K_PACKS, pack = i % K_PACKS;
            if (col_live[col]) {
                cp_async_16B(&k_smem[col * K_STRIDE + pack * 8],
                    &KeysBlocks[((kvh * K + pc + col/BF) * BF + col%BF) * D + pack * 8]);
            } else {
                reinterpret_cast<uint4*>(&k_smem[col * K_STRIDE + pack * 8])[0] =
                    make_uint4(0,0,0,0);
            }
        }
        // Zero K padding region
        for (int c = tid; c < chunk_cols; c += BLOCK) {
            #pragma unroll
            for (int p = 0; p < K_PAD; p += 2)
                reinterpret_cast<uint*>(&k_smem[c * K_STRIDE + D + p])[0] = 0;
        }
        cp_async_commit();  // Commit group 0 (K)

        // ── Load V via cp.async (group 1) ──────────────────────────
        constexpr int V_PACKS = D_V / 8;
        for (int i = tid; i < chunk_cols * V_PACKS; i += BLOCK) {
            int col = i / V_PACKS, pack = i % V_PACKS;
            if (col_live[col]) {
                cp_async_16B(&v_smem[col * D_V + pack * 8],
                    &ValuesBlocks[((kvh * K + pc + col/BF) * BF + col%BF) * D_V + pack * 8]);
            } else {
                reinterpret_cast<uint4*>(&v_smem[col * D_V + pack * 8])[0] =
                    make_uint4(0,0,0,0);
            }
        }
        cp_async_commit();  // Commit group 1 (V)

        // Wait for K (group 0) — V (group 1) may still be in flight
        cp_async_wait_group_1();
        __syncthreads();

        // ── Q@K^T: lane-per-column (overlaps with V loading) ────────
        // K is ready (group 0 waited above). V may still be in flight.
        // Each lane computes dot products for CPL columns (interleaved).
        // Lane l handles columns: l, l+32, l+64, l+96.
        float chunk_max_local = NEG_INF;
        float alpha_local = 1.0f;
        bool need_pv = false;

        if (row_valid) {
            float scores[CPL];

            #pragma unroll
            for (int ci = 0; ci < CPL; ++ci) {
                const int col = lane + ci * WARP_SIZE;
                const bool alive = (col < chunk_cols)
                    && col_live[col]
                    && row_gate[warp_id * PPC + col / BF];
                float dot = 0.0f;
                if (alive) {
                    const __half* q_row = &q_smem[warp_id * D];
                    const __half* k_row = &k_smem[col * K_STRIDE];
                    #pragma unroll
                    for (int d = 0; d < D; d += 8) {
                        const float2 q0 = __half22float2(
                            *reinterpret_cast<const __half2*>(q_row + d));
                        const float2 q1 = __half22float2(
                            *reinterpret_cast<const __half2*>(q_row + d + 2));
                        const float2 q2 = __half22float2(
                            *reinterpret_cast<const __half2*>(q_row + d + 4));
                        const float2 q3 = __half22float2(
                            *reinterpret_cast<const __half2*>(q_row + d + 6));
                        const float2 k0 = __half22float2(
                            *reinterpret_cast<const __half2*>(k_row + d));
                        const float2 k1 = __half22float2(
                            *reinterpret_cast<const __half2*>(k_row + d + 2));
                        const float2 k2 = __half22float2(
                            *reinterpret_cast<const __half2*>(k_row + d + 4));
                        const float2 k3 = __half22float2(
                            *reinterpret_cast<const __half2*>(k_row + d + 6));
                        dot += q0.x*k0.x + q0.y*k0.y
                             + q1.x*k1.x + q1.y*k1.y
                             + q2.x*k2.x + q2.y*k2.y
                             + q3.x*k3.x + q3.y*k3.y;
                    }
                }
                scores[ci] = alive ? dot * SCALE : NEG_INF;
            }

            // ── Softmax (compute P, store to smem) ──────────────────
            float local_max = NEG_INF;
            #pragma unroll
            for (int ci = 0; ci < CPL; ++ci)
                local_max = fmaxf(local_max, scores[ci]);
            chunk_max_local = warp_reduce_max(local_max);

            if (chunk_max_local > NEG_INF + 1.0f) {
                float new_m = fmaxf(m_acc, chunk_max_local);
                alpha_local = exp2f((m_acc - new_m) * LOG2E_F);
                float local_sum = 0.0f;

                #pragma unroll
                for (int ci = 0; ci < CPL; ++ci) {
                    float p = 0.0f;
                    if (scores[ci] > NEG_INF + 1.0f)
                        p = exp2f((scores[ci] - new_m) * LOG2E_F);
                    const int col = lane + ci * WARP_SIZE;
                    p_smem[warp_id * COLS + col] = __float2half_rn(p);
                    local_sum += p;
                }
                float chunk_sum = warp_reduce_sum(local_sum);
                chunk_sum = __shfl_sync(0xffffffff, chunk_sum, 0);
                l_acc = alpha_local * l_acc + chunk_sum;
                m_acc = new_m;
                need_pv = true;
            }
        }

        // ── Wait for V (ALL threads) ────────────────────────────────
        cp_async_wait_all();
        __syncthreads();

        // ── P@V: dim-parallel ───────────────────────────────────────
        if (need_pv) {
            float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int c = 0; c < chunk_cols; ++c) {
                float pv = __half2float(p_smem[warp_id * COLS + c]);
                if (pv > 0.0f) {
                    const float2 v01 = __half22float2(
                        *reinterpret_cast<const __half2*>(
                            &v_smem[c * D_V + lane * 4]));
                    const float2 v23 = __half22float2(
                        *reinterpret_cast<const __half2*>(
                            &v_smem[c * D_V + lane * 4 + 2]));
                    v_acc[0] += pv * v01.x;
                    v_acc[1] += pv * v01.y;
                    v_acc[2] += pv * v23.x;
                    v_acc[3] += pv * v23.y;
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                o_acc[j] = alpha_local * o_acc[j] + v_acc[j];
        }
        __syncthreads();
    }

    // ── Write partial results ───────────────────────────────────────
    if (row_valid) {
        if (lane == 0) {
            Out_M[hq * NUM_SPLITS + split] = m_acc;
            Out_L[hq * NUM_SPLITS + split] = l_acc;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int dv = lane * 4 + j;
            if (dv < D_V)
                Out_O[(hq * NUM_SPLITS + split) * D_V + dv] = o_acc[j];
        }
    }
}

// ── Reduce kernel ───────────────────────────────────────────────────
__global__ void attn_reduce_v1_46_kernel(
    const float* __restrict__ M_idx, const float* __restrict__ L_idx,
    const float* __restrict__ O_idx, const float* __restrict__ M_buf,
    const float* __restrict__ L_buf, const float* __restrict__ O_buf,
    float* __restrict__ Out, int NUM_SPLITS, int D_V) {

    const int hq = blockIdx.x;
    const int lane = threadIdx.x;

    float m_g = NEG_INF;
    for (int s = 0; s < NUM_SPLITS; ++s)
        m_g = fmaxf(m_g, M_idx[hq * NUM_SPLITS + s]);
    m_g = fmaxf(m_g, M_buf[hq]);

    float l_s = 0.0f;
    for (int s = 0; s < NUM_SPLITS; ++s) {
        float a = exp2f((M_idx[hq * NUM_SPLITS + s] - m_g) * LOG2E_F);
        l_s += a * L_idx[hq * NUM_SPLITS + s];
    }
    float a_b = exp2f((M_buf[hq] - m_g) * LOG2E_F);
    l_s += a_b * L_buf[hq];
    float l_safe = l_s > 0.0f ? l_s : 1.0f;

    for (int dv = lane; dv < D_V; dv += WARP_SIZE) {
        float o = 0.0f;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            float a = exp2f((M_idx[hq * NUM_SPLITS + s] - m_g) * LOG2E_F);
            o += a * O_idx[(hq * NUM_SPLITS + s) * D_V + dv];
        }
        o += a_b * O_buf[hq * D_V + dv];
        Out[hq * D_V + dv] = o / l_safe;
    }
}

}  // namespace

// ── Host launchers ──────────────────────────────────────────────────
void sparse_attn_v1_46_index(
    torch::Tensor q, torch::Tensor keys, torch::Tensor values,
    torch::Tensor centers_anchor, torch::Tensor radii_anchor,
    torch::Tensor th_anchor, torch::Tensor qnorm_anchor,
    torch::Tensor invalid,
    torch::Tensor out_m, torch::Tensor out_l, torch::Tensor out_o,
    int64_t h_q, int64_t h_kv, int64_t k_parents,
    int64_t num_splits, int64_t groups,
    int64_t dim_offset, int64_t anchor_width, double scale,
    int64_t cols_per_chunk) {

    TORCH_CHECK(q.dtype() == torch::kFloat16);
    TORCH_CHECK(keys.dtype() == torch::kFloat16);
    TORCH_CHECK(values.dtype() == torch::kFloat16);

    constexpr int D = 128, D_V = 128, BF = 4, MAX_GROUPS = 8;
    constexpr int BLOCK = MAX_GROUPS * WARP_SIZE;
    const int AW = (int)anchor_width;
    const int COLS = (int)cols_per_chunk;
    const int PPC = COLS / BF;

    auto compute_smem = [&](int cols) -> size_t {
        const int ppc = cols / BF;
        const int k_stride = D + 8;  // K_PAD=8
        return (MAX_GROUPS * D) * sizeof(__half)              // q_smem
             + (cols * k_stride) * sizeof(__half)             // k_smem (padded)
             + (cols * D_V) * sizeof(__half)                  // v_smem
             + (ppc * AW) * sizeof(__half)                    // anc_smem
             + ppc * sizeof(float)                            // rad_smem
             + cols * sizeof(int8_t)                          // inv_smem
             + cols * sizeof(int8_t)                          // col_live
             + ((MAX_GROUPS * ppc + 7) & ~7) * sizeof(int8_t) // row_gate
             + (MAX_GROUPS * cols) * sizeof(__half);           // p_smem
    };

    size_t smem = compute_smem(COLS);
    dim3 grid((int)h_kv, (int)num_splits);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Macro to set max dynamic smem and launch
    #define L46(AW_T, COLS_T) do { \
        auto* kfn = &sparse_attn_v1_46_kernel<D,D_V,BF,MAX_GROUPS,AW_T,COLS_T>; \
        cudaFuncSetAttribute(kfn, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); \
        kfn<<<grid, BLOCK, smem, stream>>>( \
            (const __half*)q.data_ptr(), (const __half*)keys.data_ptr(), \
            (const __half*)values.data_ptr(), (const __half*)centers_anchor.data_ptr(), \
            (const __half*)radii_anchor.data_ptr(), (const __half*)th_anchor.data_ptr(), \
            (const __half*)qnorm_anchor.data_ptr(), (const int8_t*)invalid.data_ptr(), \
            out_m.data_ptr<float>(), out_l.data_ptr<float>(), out_o.data_ptr<float>(), \
            (int)h_q, (int)k_parents, (int)num_splits, (int)groups, \
            (int)dim_offset, (float)scale); \
    } while(0)

    if (COLS == 128) {
        if (AW == 16) { L46(16, 128); }
        else if (AW == 32) { L46(32, 128); }
        else if (AW == 64) { L46(64, 128); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else if (COLS == 64) {
        if (AW == 16) { L46(16, 64); }
        else if (AW == 32) { L46(32, 64); }
        else if (AW == 64) { L46(64, 64); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else if (COLS == 32) {
        if (AW == 16) { L46(16, 32); }
        else if (AW == 32) { L46(32, 32); }
        else if (AW == 64) { L46(64, 32); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else {
        TORCH_CHECK(false, "cols_per_chunk must be 32/64/128");
    }
    #undef L46
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_attn_v1_46_reduce(
    torch::Tensor m_idx, torch::Tensor l_idx, torch::Tensor o_idx,
    torch::Tensor m_buf, torch::Tensor l_buf, torch::Tensor o_buf,
    torch::Tensor out, int64_t num_splits) {
    int h_q = (int)m_idx.size(0), d_v = (int)o_idx.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    attn_reduce_v1_46_kernel<<<h_q, WARP_SIZE, 0, stream>>>(
        m_idx.data_ptr<float>(), l_idx.data_ptr<float>(), o_idx.data_ptr<float>(),
        m_buf.data_ptr<float>(), l_buf.data_ptr<float>(), o_buf.data_ptr<float>(),
        out.data_ptr<float>(), (int)num_splits, d_v);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attn_v1_46_index", &sparse_attn_v1_46_index);
    m.def("sparse_attn_v1_46_reduce", &sparse_attn_v1_46_reduce);
}
