// v9.0 — same as v7.10/v7.11 for Phases 1 (score top-L) and 2 (depth);
// Phase 3 is simplified to dump the per-subspace parent_alive bitmap to
// global memory and exit. The sparse-attn kernel (v2.6) consumes that
// bitmap directly instead of the live_idx materialised by v7.10's
// Phase 3, eliminating the per-key atomicOr / scan / atomicAdd /
// live_idx writeback work.
//
// Output: parent_alive_bitmap[hq, s, K_words] uint32 (S=4 fixed).
// Specialised on L=256, BLOCK=256, IPT_L=1, S=4, bf=4 (same as v7.10).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <float.h>

namespace cg = cooperative_groups;

namespace {

constexpr int BLOCK   = 256;
constexpr int L       = 256;
constexpr int IPT_L   = 1;
constexpr int N_WARPS = BLOCK / 32;

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

typedef cub::BlockRadixSort<float, BLOCK, IPT_L, int> BlockSort;

__global__ void filter_v9_kernel(
    const half*    __restrict__ q,
    const half*    __restrict__ centers,
    const int32_t* __restrict__ dim_offsets,
    const int32_t* __restrict__ dim_widths,
    const int64_t* __restrict__ q_head_to_kv,
    const float*   __restrict__ threshold,
    float*         __restrict__ top_scores,
    int32_t*       __restrict__ top_indices,
    int32_t*       __restrict__ depth_g,
    uint32_t*      __restrict__ parent_alive_bitmap,  // (Hq, 4, K_words)
    int Hq, int Hkv, int K, int max_w, int D,
    int K_words, int has_map)
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
            int pairs = width >> 1;
            int tail  = pairs << 1;
            for (int k = tid; k < K; k += BLOCK) {
                const half* c_ptr = centers + (((s * Hkv + kvh) * K + k) * max_w);
                const half2* c2 = reinterpret_cast<const half2*>(c_ptr);
                float acc = 0.0f;
                for (int p = 0; p < pairs; p++) {
                    float2 f = __half22float2(__hmul2(q2[p], c2[p]));
                    acc += f.x + f.y;
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

    // ─────────────── PHASE 2+3 FUSED (blk == 0) ───────────────
    // Compute depth, broadcast to all threads via smem, then build bitmap
    // and dump to global — all without a second grid_sync.
    if (blk == 0) {
        // smem layout (reused per phase via offsets):
        //   [smem_sums float×L] [smem_vwarp_hit int×N_VWARPS] [s_depth int]
        //   [smem_bm  uint32×4*K_words]   (overlaps once depth is known)
        float* smem_sums      = reinterpret_cast<float*>(smem_raw);
        int*   smem_vwarp_hit = reinterpret_cast<int*>(smem_raw + L * sizeof(float));
        int*   s_depth_p      = smem_vwarp_hit + (L / 32);

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
                depth_g[hq] = d;
                *s_depth_p = d;
            }
        }
        __syncthreads();
        int d = *s_depth_p;

        // Build bitmap in fresh smem region (sized 4*K_words uint32).
        // It overlaps with smem_sums but we no longer need that.
        uint32_t* smem_bm = reinterpret_cast<uint32_t*>(smem_raw);
        int bm_total = 4 * K_words;
        for (int i = tid; i < bm_total; i += BLOCK) smem_bm[i] = 0u;
        __syncthreads();

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

        uint32_t* gbm = parent_alive_bitmap + (int64_t)hq * 4 * K_words;
        for (int i = tid; i < bm_total; i += BLOCK) {
            gbm[i] = smem_bm[i];
        }
    }
}

}  // namespace

void ta_filter_v9_0_launch(
    torch::Tensor q, torch::Tensor centers,
    torch::Tensor dim_offsets, torch::Tensor dim_widths,
    torch::Tensor q_head_to_kv,
    torch::Tensor threshold,
    torch::Tensor top_scores, torch::Tensor top_indices,
    torch::Tensor depth, torch::Tensor parent_alive_bitmap,
    int64_t k_clusters)
{
    TORCH_CHECK(q.is_cuda() && centers.is_cuda());
    TORCH_CHECK(q.scalar_type()                  == torch::kFloat16);
    TORCH_CHECK(centers.scalar_type()            == torch::kFloat16);
    TORCH_CHECK(threshold.scalar_type()          == torch::kFloat32);
    TORCH_CHECK(top_scores.scalar_type()         == torch::kFloat32);
    TORCH_CHECK(top_indices.scalar_type()        == torch::kInt32);
    TORCH_CHECK(depth.scalar_type()              == torch::kInt32);
    TORCH_CHECK(parent_alive_bitmap.scalar_type()== torch::kInt32);

    int Hq    = (int)q.size(0);
    int D     = (int)q.size(1);
    int S     = (int)centers.size(0);
    int Hkv   = (int)centers.size(1);
    int K     = (int)centers.size(2);
    int max_w = (int)centers.size(3);
    int K_words = (K + 31) / 32;
    TORCH_CHECK(S == 4);
    TORCH_CHECK(K <= 32767);
    TORCH_CHECK(top_scores.size(2) == L, "v9.0 specialised on L=256");
    TORCH_CHECK(parent_alive_bitmap.size(0) == Hq);
    TORCH_CHECK(parent_alive_bitmap.size(1) == 4);
    TORCH_CHECK(parent_alive_bitmap.size(2) >= K_words);

    bool has_map = q_head_to_kv.defined() && q_head_to_kv.numel() > 0;
    if (has_map) {
        TORCH_CHECK(q_head_to_kv.scalar_type() == torch::kInt64);
        TORCH_CHECK(q_head_to_kv.numel() == Hq);
    } else {
        TORCH_CHECK(Hq == Hkv);
    }

    // Grid: (Hq, 4) — phase 1 needs blk<4; phases 2/3 use blk==0.
    dim3 grid(Hq, 4);
    dim3 block(BLOCK);

    size_t smem_cub   = sizeof(BlockSort::TempStorage);
    size_t smem_depth = L * sizeof(float) + (L / 32) * sizeof(int);
    size_t smem_bm    = (size_t)4 * K_words * sizeof(uint32_t) + 1 * sizeof(int);
    size_t smem_bytes = std::max(smem_cub, std::max(smem_depth, smem_bm));

    auto stream = at::cuda::getCurrentCUDAStream();

    const half*    q_ptr  = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half*    c_ptr  = reinterpret_cast<const half*>(centers.data_ptr<at::Half>());
    const int32_t* doff_p = dim_offsets.data_ptr<int32_t>();
    const int32_t* dwid_p = dim_widths.data_ptr<int32_t>();
    const int64_t* kv_ptr = has_map ? q_head_to_kv.data_ptr<int64_t>() : nullptr;
    const float*   th_ptr = threshold.data_ptr<float>();
    float*         ts_ptr = top_scores.data_ptr<float>();
    int32_t*       ti_ptr = top_indices.data_ptr<int32_t>();
    int32_t*       dp_ptr = depth.data_ptr<int32_t>();
    uint32_t*      bm_ptr = reinterpret_cast<uint32_t*>(parent_alive_bitmap.data_ptr<int32_t>());
    int hm_i = (int)has_map;

    void* args[] = {
        (void*)&q_ptr, (void*)&c_ptr, (void*)&doff_p, (void*)&dwid_p,
        (void*)&kv_ptr, (void*)&th_ptr,
        (void*)&ts_ptr, (void*)&ti_ptr, (void*)&dp_ptr, (void*)&bm_ptr,
        (void*)&Hq, (void*)&Hkv, (void*)&K, (void*)&max_w, (void*)&D,
        (void*)&K_words, (void*)&hm_i,
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)filter_v9_kernel,
        grid, block, args, smem_bytes, stream);
    TORCH_CHECK(err == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(err));
}
