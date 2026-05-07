// update_v1_1 — fast incremental cluster kernel for the TA-filter index.
//
// Per (s, h_kv): cluster 256 buffer keys into K_BUF=64 clusters of BF=4 by:
//   1. Projecting each key onto axis = sum-of-subspace-dims (cheap, no PCA).
//   2. Sorting the 256 (proj, idx) pairs with CUB BlockRadixSort.
//   3. Cluster c = sorted_rank / BF.
//   4. Center[s, h, c, :w] = mean of the 4 keys whose sorted rank is in [4c, 4c+4).
//
// One thread per buffer slot (256 threads/block, blockDim = (S=4, H_kv)).
// Writes:
//   centers_padded_f16[s, h, K_used + c, :w] (padded slice; we zero the tail
//      [w..max_w)).
//   assigns_padded[s, h, N_used + b]  ← global cluster id (K_used + c) cast to
//      the assigns dtype (int16 or int32).
// Does NOT touch: invalid_mask, _assigns_packed_u64_v34, K_used/N_used, keys
// and values arena. Those are the publish phase.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace {

constexpr int BLOCK = 256;
constexpr int BUFFER_SIZE = 256;
constexpr int K_BUF = 64;     // BUFFER_SIZE / BF
constexpr int BF = 4;

template <typename AssignsT>
__global__ void cluster_kernel(
    const half*    __restrict__ buffer_keys,   // (H_kv, B, D)
    const int32_t* __restrict__ dim_offsets,   // (S,)
    const int32_t* __restrict__ dim_widths,    // (S,)
    half*          __restrict__ centers_arena, // (S, H_kv, K_cap, max_w)
    AssignsT*      __restrict__ assigns_arena, // (S, H_kv, N_pad)
    int H_kv, int D, int max_w,
    int K_cap, int K_used,
    int N_pad, int N_used)
{
    int s   = blockIdx.x;
    int h   = blockIdx.y;
    int tid = threadIdx.x;

    int off = dim_offsets[s];
    int w   = dim_widths[s];

    const half* kh = buffer_keys + (int64_t)h * BUFFER_SIZE * D;

    // ── 1. Projection: sum-of-w-dims of buffer key `tid` in this subspace. ──
    float proj = 0.0f;
    {
        const half* kp = kh + (int64_t)tid * D + off;
        for (int j = 0; j < w; j++) proj += __half2float(kp[j]);
    }

    // ── 2. Sort (proj, idx) ascending. With BlockSort items_per_thread=1
    //       and 256 threads, blocked == striped layout — thread t holds the
    //       item with sorted rank t. ──
    using BlockSort = cub::BlockRadixSort<float, BLOCK, 1, int>;
    __shared__ typename BlockSort::TempStorage tmp_sort;
    float k_arr[1] = { proj };
    int   v_arr[1] = { tid };
    BlockSort(tmp_sort).Sort(k_arr, v_arr);
    int rank_buffer_idx = v_arr[0];   // buffer idx whose proj has rank `tid`

    __shared__ int sorted_idx[BUFFER_SIZE];   // sorted_idx[rank] = orig buffer idx
    __shared__ int rank_of[BUFFER_SIZE];      // rank_of[orig idx]  = rank
    sorted_idx[tid] = rank_buffer_idx;
    rank_of[rank_buffer_idx] = tid;
    __syncthreads();

    // ── 3. Per-buffer assignment: cluster id = rank_of[orig_idx] / BF. ──
    int my_cluster = rank_of[tid] / BF;
    int my_global_cluster = K_used + my_cluster;
    int64_t assigns_off =
        ((int64_t)s * H_kv + h) * (int64_t)N_pad + (int64_t)(N_used + tid);
    assigns_arena[assigns_off] = (AssignsT)my_global_cluster;

    // ── 4. Cluster centers: 64 clusters * w dims = up to 64*max_w pairs. ──
    int total_pairs = K_BUF * w;
    int64_t centers_base =
        ((int64_t)s * H_kv + h) * (int64_t)K_cap * (int64_t)max_w;

    for (int idx = tid; idx < total_pairs; idx += BLOCK) {
        int c     = idx / w;
        int d_loc = idx - c * w;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < BF; i++) {
            int b = sorted_idx[c * BF + i];
            acc += __half2float(kh[(int64_t)b * D + off + d_loc]);
        }
        acc *= 0.25f;
        int64_t off_c =
            centers_base + (int64_t)(K_used + c) * (int64_t)max_w + d_loc;
        centers_arena[off_c] = __float2half(acc);
    }

    // Zero the unused padding [w..max_w) on the new center rows.
    if (w < max_w) {
        int pad_pairs = K_BUF * (max_w - w);
        for (int idx = tid; idx < pad_pairs; idx += BLOCK) {
            int c       = idx / (max_w - w);
            int d_local = w + (idx - c * (max_w - w));
            int64_t off_c =
                centers_base + (int64_t)(K_used + c) * (int64_t)max_w + d_local;
            centers_arena[off_c] = __float2half(0.0f);
        }
    }
}

}  // namespace


// Public launcher.
void update_v1_1_cluster_launch(
    torch::Tensor buffer_keys,    // (H_kv, 256, D) fp16
    torch::Tensor dim_offsets,    // (S,) int32
    torch::Tensor dim_widths,     // (S,) int32
    torch::Tensor centers_arena,  // (S, H_kv, K_cap, max_w) fp16
    torch::Tensor assigns_arena,  // (S, H_kv, N_pad) int16 or int32
    int64_t K_used,
    int64_t N_used)
{
    TORCH_CHECK(buffer_keys.is_cuda());
    TORCH_CHECK(buffer_keys.scalar_type() == torch::kFloat16);
    TORCH_CHECK(centers_arena.scalar_type() == torch::kFloat16);
    TORCH_CHECK(buffer_keys.size(1) == BUFFER_SIZE,
                "update_v1_1 requires buffer of exactly 256 keys");

    int S      = (int)centers_arena.size(0);
    int H_kv   = (int)centers_arena.size(1);
    int K_cap  = (int)centers_arena.size(2);
    int max_w  = (int)centers_arena.size(3);
    int N_pad  = (int)assigns_arena.size(2);
    int D      = (int)buffer_keys.size(2);

    TORCH_CHECK(S == 4, "S must be 4");
    TORCH_CHECK((int)assigns_arena.size(0) == S);
    TORCH_CHECK((int)assigns_arena.size(1) == H_kv);

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(S, H_kv);
    dim3 block(BLOCK);

    if (assigns_arena.scalar_type() == torch::kInt16) {
        cluster_kernel<int16_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(buffer_keys.data_ptr<at::Half>()),
            dim_offsets.data_ptr<int32_t>(),
            dim_widths.data_ptr<int32_t>(),
            reinterpret_cast<half*>(centers_arena.data_ptr<at::Half>()),
            assigns_arena.data_ptr<int16_t>(),
            H_kv, D, max_w, K_cap, (int)K_used, N_pad, (int)N_used);
    } else {
        TORCH_CHECK(assigns_arena.scalar_type() == torch::kInt32,
                    "assigns_arena must be int16 or int32");
        cluster_kernel<int32_t><<<grid, block, 0, stream>>>(
            reinterpret_cast<const half*>(buffer_keys.data_ptr<at::Half>()),
            dim_offsets.data_ptr<int32_t>(),
            dim_widths.data_ptr<int32_t>(),
            reinterpret_cast<half*>(centers_arena.data_ptr<at::Half>()),
            assigns_arena.data_ptr<int32_t>(),
            H_kv, D, max_w, K_cap, (int)K_used, N_pad, (int)N_used);
    }
}
