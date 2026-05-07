#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <float.h>

// v7.10 — Single-launch fused pipeline via cooperative groups.
// Grid (Hq, MAX_BLK), Block 256.  MAX_BLK = max(4, N_TILES).
// Phase 1: blockIdx.y < 4 → score_top_l for (hq, s).
// grid.sync()
// Phase 2: blockIdx.y == 0 → depth + reset live_count[hq].
// grid.sync()
// Phase 3: blockIdx.y < N_TILES → alive_compact for (hq, tile).
//
// Specialised on L=256, BLOCK=256, IPT_L=1, TILE_N=2048, PER_THREAD=8, S=4, bf=4.

namespace cg = cooperative_groups;

namespace {

constexpr int BLOCK       = 256;
constexpr int L           = 256;   // IPT_L=1
constexpr int IPT_L       = 1;
constexpr int N_WARPS     = BLOCK / 32;  // 8

template <int IPT>
__device__ __forceinline__
void heap_push_l(float* keys, int* vals, float new_k, int new_v) {
    if (new_k <= keys[0]) return;
    keys[0] = new_k;
    vals[0] = new_v;
    int i = 0;
    while (true) {
        int left  = 2 * i + 1;
        int right = 2 * i + 2;
        int smallest = i;
        if (left  < IPT && keys[left]  < keys[smallest]) smallest = left;
        if (right < IPT && keys[right] < keys[smallest]) smallest = right;
        if (smallest == i) break;
        float tk = keys[i]; keys[i] = keys[smallest]; keys[smallest] = tk;
        int   tv = vals[i]; vals[i] = vals[smallest]; vals[smallest] = tv;
        i = smallest;
    }
}

template <int PER_T>
__device__ __forceinline__ void load_pk_n(
    const uint64_t* __restrict__ a_packed,
    int n_base, int n_end, uint64_t* pk)
{
    bool full_chunk = (n_base + PER_T <= n_end);
    if (full_chunk) {
        const int4* base_p = reinterpret_cast<const int4*>(a_packed + n_base);
        #pragma unroll
        for (int it = 0; it < PER_T / 2; it++) {
            union { int4 v; uint64_t k[2]; } u;
            u.v = base_p[it];
            pk[2*it]     = u.k[0];
            pk[2*it + 1] = u.k[1];
        }
    } else {
        #pragma unroll
        for (int it = 0; it < PER_T; it++) {
            int n = n_base + it;
            pk[it] = (n < n_end) ? a_packed[n] : (uint64_t)0xFFFFFFFFFFFFFFFFULL;
        }
    }
}

typedef cub::BlockRadixSort<float, BLOCK, IPT_L, int> BlockSort;

template <int TILE_N, int PER_THREAD>
__global__ void fused_pipeline_kernel_t(
    // inputs
    const half*    __restrict__ q,
    const half*    __restrict__ centers,
    const int32_t* __restrict__ dim_offsets,
    const int32_t* __restrict__ dim_widths,
    const int64_t* __restrict__ q_head_to_kv,
    const float*   __restrict__ threshold,
    const int64_t* __restrict__ assigns_packed,
    // workspace / outputs
    float*         __restrict__ top_scores,
    int32_t*       __restrict__ top_indices,
    int32_t*       __restrict__ depth_g,
    int32_t*       __restrict__ live_idx,
    int32_t*       __restrict__ live_count,
    // shapes
    int Hq, int Hkv, int K, int K_stride, int max_w, int D,
    int Npad, int N_eff, int K_words, int N_TILES, int has_map)
{
    cg::grid_group grid = cg::this_grid();
    int hq  = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;

    int kvh = has_map ? (int)q_head_to_kv[hq] : hq;
    bool kvh_ok = (kvh >= 0 && kvh < Hkv);

    extern __shared__ uint8_t smem_raw[];

    // ─────────────── PHASE 1: SCORE (blk < 4) ───────────────
    if (blk < 4) {
        int s    = blk;
        int off  = dim_offsets[s];
        int width= dim_widths[s];
        const half* q_ptr = q + hq * D + off;

        float heap_keys[IPT_L];
        int   heap_vals[IPT_L];
        #pragma unroll
        for (int i = 0; i < IPT_L; i++) { heap_keys[i] = -FLT_MAX; heap_vals[i] = -1; }

        if (kvh_ok) {
            const half2* q2 = reinterpret_cast<const half2*>(q_ptr);
            int pairs = width >> 1;     // half2 count
            int tail  = pairs << 1;
            // Use int4 (16B = 8 fp16 = 4 half2) loads when width is a
            // multiple of 8 fp16 (= 4 half2). Width=32 satisfies this.
            int int4_groups = pairs >> 2;   // each int4 covers 4 half2
            int int4_tail   = int4_groups << 2;
            for (int k = tid; k < K; k += BLOCK) {
                const half* c_ptr = centers + (((s * Hkv + kvh) * K_stride + k) * max_w);
                float acc = 0.0f;
                if (int4_groups > 0 && (((uintptr_t)c_ptr) & 0xF) == 0) {
                    const int4* qv = reinterpret_cast<const int4*>(q_ptr);
                    const int4* cv = reinterpret_cast<const int4*>(c_ptr);
                    #pragma unroll
                    for (int g = 0; g < int4_groups; g++) {
                        int4 qg = qv[g];
                        int4 cg = cv[g];
                        const half2* qh2 = reinterpret_cast<const half2*>(&qg);
                        const half2* ch2 = reinterpret_cast<const half2*>(&cg);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            float2 f = __half22float2(__hmul2(qh2[j], ch2[j]));
                            acc += f.x + f.y;
                        }
                    }
                    // Remaining half2 pairs (if width not multiple of 8 fp16).
                    const half2* c2 = reinterpret_cast<const half2*>(c_ptr);
                    for (int p = int4_tail; p < pairs; p++) {
                        float2 f = __half22float2(__hmul2(q2[p], c2[p]));
                        acc += f.x + f.y;
                    }
                } else {
                    const half2* c2 = reinterpret_cast<const half2*>(c_ptr);
                    for (int p = 0; p < pairs; p++) {
                        float2 f = __half22float2(__hmul2(q2[p], c2[p]));
                        acc += f.x + f.y;
                    }
                }
                if (tail < width) acc += __half2float(q_ptr[tail]) * __half2float(c_ptr[tail]);
                heap_push_l<IPT_L>(heap_keys, heap_vals, acc, k);
            }
        }

        typename BlockSort::TempStorage* temp_storage = reinterpret_cast<typename BlockSort::TempStorage*>(smem_raw);
        BlockSort(*temp_storage).SortDescendingBlockedToStriped(heap_keys, heap_vals);

        int base = (hq * 4 + s) * L;
        #pragma unroll
        for (int i = 0; i < IPT_L; i++) {
            int rank = tid + i * BLOCK;
            top_scores[base + rank]  = heap_keys[i];
            top_indices[base + rank] = heap_vals[i];
        }
    }

    grid.sync();

    // ─────────────── PHASE 2: DEPTH (blk == 0) ───────────────
    if (blk == 0) {
        float* smem_sums = reinterpret_cast<float*>(smem_raw);
        int*   smem_vwarp_hit = reinterpret_cast<int*>(smem_raw + L * sizeof(float));

        constexpr int N_VWARPS = L / 32;
        float th = threshold[hq];
        const float* ts = top_scores + (int64_t)hq * 4 * L;

        for (int r = tid; r < L; r += BLOCK) {
            smem_sums[r] = ts[r] + ts[L + r] + ts[2*L + r] + ts[3*L + r];
        }
        __syncthreads();

        {
            int warp = tid >> 5;
            int lane = tid & 31;
            for (int vw = warp; vw < N_VWARPS; vw += N_WARPS) {
                int ballot = __ballot_sync(0xFFFFFFFFu, smem_sums[vw * 32 + lane] < th);
                if (lane == 0)
                    smem_vwarp_hit[vw] = (ballot != 0) ? __ffs(ballot) - 1 : 32;
            }
        }
        __syncthreads();

        if (tid < 32) {
            bool has_hit = (tid < N_VWARPS) & (smem_vwarp_hit[tid] < 32);
            int  meta    = __ballot_sync(0xFFFFFFFFu, has_hit);
            if (tid == 0) {
                int d;
                if (meta != 0) {
                    int fvw = __ffs(meta) - 1;
                    d = fvw * 32 + smem_vwarp_hit[fvw] + 1;
                } else {
                    d = L;
                }
                depth_g[hq]    = d;
                live_count[hq] = 0;
            }
        }
    }

    grid.sync();

    // ─────────────── PHASE 3: ALIVE (blk < N_TILES) ───────────────
    if (blk < N_TILES) {
        int tile    = blk;
        int n_start = tile * TILE_N;
        int n_limit = (N_eff > 0 && N_eff < Npad) ? N_eff : Npad;
        if (n_start >= n_limit) return;
        int n_end   = n_start + TILE_N;
        if (n_end > n_limit) n_end = n_limit;

        uint32_t* smem_bm    = reinterpret_cast<uint32_t*>(smem_raw);
        int* s_aux           = reinterpret_cast<int*>(smem_raw + 4 * K_words * sizeof(uint32_t));
        int* s_depth_p       = s_aux + 0;
        int* s_warp_pop      = s_aux + 1;
        int* s_warp_off      = s_warp_pop + N_WARPS;
        int* s_block_off     = s_warp_off + N_WARPS;

        int bm_total = 4 * K_words;
        for (int i = tid; i < bm_total; i += BLOCK) smem_bm[i] = 0u;
        if (tid == 0) *s_depth_p = depth_g[hq];
        __syncthreads();

        int d = *s_depth_p;
        const int32_t* ti = top_indices + (int64_t)hq * 4 * L;
        for (int s = 0; s < 4; s++) {
            const int32_t* tis = ti + s * L;
            for (int r = tid; r < d; r += BLOCK) {
                int32_t c = tis[r];
                if ((unsigned)c < (unsigned)K) {
                    atomicOr(&smem_bm[s * K_words + (c >> 5)], 1u << (c & 31));
                }
            }
        }
        __syncthreads();

        if (!kvh_ok) return;

        const int64_t kv_off = (int64_t)kvh * Npad;
        const uint32_t* bm0 = smem_bm + 0 * K_words;
        const uint32_t* bm1 = smem_bm + 1 * K_words;
        const uint32_t* bm2 = smem_bm + 2 * K_words;
        const uint32_t* bm3 = smem_bm + 3 * K_words;
        const unsigned K_bits = (unsigned)(K_words * 32);
        const uint64_t* a_packed = reinterpret_cast<const uint64_t*>(assigns_packed) + kv_off;

        const int lane    = tid & 31;
        const int warp_id = tid >> 5;
        const unsigned lane_lt = (lane == 0) ? 0u : ((1u << lane) - 1u);

        int n_base = n_start + tid * PER_THREAD;
        uint64_t pk[PER_THREAD];
        load_pk_n<PER_THREAD>(a_packed, n_base, n_end, pk);

        unsigned ballot_arr[PER_THREAD];
        int warp_total = 0;
        #pragma unroll
        for (int it = 0; it < PER_THREAD; it++) {
            int n = n_base + it;
            bool alive = false;
            if (n < n_end) {
                uint64_t packed = pk[it];
                unsigned p0 = (unsigned)( packed        & 0xFFFFu);
                unsigned p1 = (unsigned)((packed >> 16) & 0xFFFFu);
                unsigned p2 = (unsigned)((packed >> 32) & 0xFFFFu);
                unsigned p3 = (unsigned)((packed >> 48) & 0xFFFFu);
                int hit = 0;
                if (p0 < K_bits) hit |= (int)((bm0[p0 >> 5] >> (p0 & 31)) & 1u);
                if (p1 < K_bits) hit |= (int)((bm1[p1 >> 5] >> (p1 & 31)) & 1u);
                if (p2 < K_bits) hit |= (int)((bm2[p2 >> 5] >> (p2 & 31)) & 1u);
                if (p3 < K_bits) hit |= (int)((bm3[p3 >> 5] >> (p3 & 31)) & 1u);
                alive = (hit != 0);
            }
            unsigned bits = __ballot_sync(0xFFFFFFFFu, alive);
            ballot_arr[it] = bits;
            warp_total    += __popc(bits);
        }

        if (lane == 0) s_warp_pop[warp_id] = warp_total;
        __syncthreads();

        if (warp_id == 0) {
            int my_pop = (lane < N_WARPS) ? s_warp_pop[lane] : 0;
            int v = my_pop;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                int t = __shfl_up_sync(0xFFFFFFFFu, v, off);
                if (lane >= off) v += t;
            }
            if (lane < N_WARPS) s_warp_off[lane] = v - my_pop;
            if (lane == N_WARPS - 1) {
                *s_block_off = atomicAdd(&live_count[hq], v);
            }
        }
        __syncthreads();

        int warp_off  = s_warp_off[warp_id];
        int block_off = *s_block_off;
        int64_t out_base = (int64_t)hq * Npad + block_off + warp_off;

        int cum = 0;
        #pragma unroll
        for (int it = 0; it < PER_THREAD; it++) {
            unsigned bits = ballot_arr[it];
            int rank = __popc(bits & lane_lt);
            bool alive = ((bits >> lane) & 1u) != 0u;
            int n = n_base + it;
            if (alive) live_idx[out_base + cum + rank] = n;
            cum += __popc(bits);
        }
    }
}

}  // namespace

void ta_filter_v7_10_fused_launch(
    torch::Tensor q, torch::Tensor centers,
    torch::Tensor dim_offsets, torch::Tensor dim_widths,
    torch::Tensor q_head_to_kv,
    torch::Tensor threshold,
    torch::Tensor assigns_packed,
    torch::Tensor top_scores, torch::Tensor top_indices,
    torch::Tensor depth, torch::Tensor live_idx, torch::Tensor live_count,
    int64_t k_clusters,
    int64_t k_stride,
    int64_t n_eff,
    int64_t tile_n)
{
    TORCH_CHECK(q.is_cuda() && centers.is_cuda() && assigns_packed.is_cuda());
    TORCH_CHECK(q.scalar_type()              == torch::kFloat16);
    TORCH_CHECK(centers.scalar_type()        == torch::kFloat16);
    TORCH_CHECK(threshold.scalar_type()      == torch::kFloat32);
    TORCH_CHECK(assigns_packed.scalar_type() == torch::kInt64);
    TORCH_CHECK(top_scores.scalar_type()     == torch::kFloat32);
    TORCH_CHECK(top_indices.scalar_type()    == torch::kInt32);
    TORCH_CHECK(depth.scalar_type()          == torch::kInt32);
    TORCH_CHECK(live_idx.scalar_type()       == torch::kInt32);
    TORCH_CHECK(live_count.scalar_type()     == torch::kInt32);

    int Hq    = (int)q.size(0);
    int D     = (int)q.size(1);
    int S     = (int)centers.size(0);
    int Hkv   = (int)centers.size(1);
    int K_full = (int)centers.size(2);          // memory stride along the K dim
    int K     = (int)k_clusters;                // loop bound for scoring
    int K_stride = (int)k_stride;               // memory stride (= K_full normally)
    int max_w = (int)centers.size(3);
    if (K_stride <= 0) K_stride = K_full;
    if (K <= 0 || K > K_full) K = K_full;
    int Npad  = (int)assigns_packed.size(1);
    int N_eff_i = (int)n_eff;
    if (N_eff_i <= 0 || N_eff_i > Npad) N_eff_i = Npad;
    int K_words = (K + 31) / 32;
    int TILE_N  = (int)tile_n;
    TORCH_CHECK(TILE_N == 2048 || TILE_N == 4096, "tile_n must be 2048 or 4096");
    int N_TILES = (N_eff_i + TILE_N - 1) / TILE_N;
    if (N_TILES < 1) N_TILES = 1;
    TORCH_CHECK(S == 4);
    TORCH_CHECK(K <= 32767);
    TORCH_CHECK(top_scores.size(2) == L, "v7.10/v7.11 specialised on L=256");

    bool has_map = q_head_to_kv.defined() && q_head_to_kv.numel() > 0;
    if (has_map) {
        TORCH_CHECK(q_head_to_kv.scalar_type() == torch::kInt64);
        TORCH_CHECK(q_head_to_kv.numel() == Hq);
    } else {
        TORCH_CHECK(Hq == Hkv);
    }

    int max_blk = (N_TILES > 4) ? N_TILES : 4;
    dim3 grid(Hq, max_blk);
    dim3 block(BLOCK);

    // smem: max of CUB temp, depth scratch, alive scratch.
    size_t smem_cub    = sizeof(BlockSort::TempStorage);
    size_t smem_depth  = L * sizeof(float) + 8 * sizeof(int);
    size_t smem_alive  = (size_t)4 * K_words * sizeof(uint32_t)
                       + (1 + 2 * N_WARPS + 1) * sizeof(int);
    size_t smem_bytes = std::max(smem_cub, std::max(smem_depth, smem_alive));

    auto stream = at::cuda::getCurrentCUDAStream();

    const half*    q_ptr  = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half*    c_ptr  = reinterpret_cast<const half*>(centers.data_ptr<at::Half>());
    const int32_t* doff_p = dim_offsets.data_ptr<int32_t>();
    const int32_t* dwid_p = dim_widths.data_ptr<int32_t>();
    const int64_t* kv_ptr = has_map ? q_head_to_kv.data_ptr<int64_t>() : nullptr;
    const float*   th_ptr = threshold.data_ptr<float>();
    const int64_t* ap_ptr = assigns_packed.data_ptr<int64_t>();
    float*         ts_ptr = top_scores.data_ptr<float>();
    int32_t*       ti_ptr = top_indices.data_ptr<int32_t>();
    int32_t*       dp_ptr = depth.data_ptr<int32_t>();
    int32_t*       li_ptr = live_idx.data_ptr<int32_t>();
    int32_t*       lc_ptr = live_count.data_ptr<int32_t>();
    int hm_i = (int)has_map;

    void* args[] = {
        (void*)&q_ptr, (void*)&c_ptr, (void*)&doff_p, (void*)&dwid_p,
        (void*)&kv_ptr, (void*)&th_ptr, (void*)&ap_ptr,
        (void*)&ts_ptr, (void*)&ti_ptr, (void*)&dp_ptr, (void*)&li_ptr, (void*)&lc_ptr,
        (void*)&Hq, (void*)&Hkv, (void*)&K, (void*)&K_stride, (void*)&max_w, (void*)&D,
        (void*)&Npad, (void*)&N_eff_i, (void*)&K_words, (void*)&N_TILES, (void*)&hm_i,
    };

    cudaError_t err;
    if (TILE_N == 2048) {
        err = cudaLaunchCooperativeKernel(
            (void*)fused_pipeline_kernel_t<2048, 8>,
            grid, block, args, smem_bytes, stream);
    } else {
        err = cudaLaunchCooperativeKernel(
            (void*)fused_pipeline_kernel_t<4096, 16>,
            grid, block, args, smem_bytes, stream);
    }
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
}
