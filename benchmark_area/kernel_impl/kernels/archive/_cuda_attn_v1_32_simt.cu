#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

template <int D, int D_V, int BF, int COLS, int MAX_GROUPS>
__global__ void sparse_attn_index_v1_32_simt_kernel(
    const __half* __restrict__ Q,            // (H_q, D)
    const __half* __restrict__ KeysBlocks,   // (H_kv, K, BF, D)
    const __half* __restrict__ ValuesBlocks, // (H_kv, K, BF, D_v)
    const int8_t* __restrict__ ClusterPass,  // (S, H_q, K)
    const int8_t* __restrict__ InvalidBlocks,// (H_kv, K, BF)
    float* __restrict__ Out_M,               // (H_q, NUM_SPLITS)
    float* __restrict__ Out_L,               // (H_q, NUM_SPLITS)
    float* __restrict__ Out_O,               // (H_q, NUM_SPLITS, D_v)
    int H_Q,
    int K,
    int NUM_SPLITS,
    int GROUPS,
    int ANCHOR_S,
    float SCALE) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    __shared__ __half q_smem[MAX_GROUPS][D];
    __shared__ __half k_smem[COLS][D];
    __shared__ __half v_smem[COLS][D_V];
    __shared__ float score_smem[MAX_GROUPS][COLS];
    __shared__ __half p_smem[MAX_GROUPS][COLS];
    __shared__ int parent_smem[COLS];
    __shared__ int8_t invalid_smem[COLS];
    __shared__ int8_t row_gate_smem[MAX_GROUPS][COLS];
    __shared__ int8_t col_live_smem[COLS];

    const int parents_per_split = (K + NUM_SPLITS - 1) / NUM_SPLITS;
    const int p_start = split * parents_per_split;
    const int p_end = min(p_start + parents_per_split, K);
    const int q_row_base = kvh * GROUPS;
    const bool row_valid = warp_id < GROUPS;
    const int hq = q_row_base + warp_id;

    constexpr int Q_PACKS = D / 8;
    for (int pack_idx = tid; pack_idx < GROUPS * Q_PACKS; pack_idx += blockDim.x) {
        const int row = pack_idx / Q_PACKS;
        const int pack = pack_idx % Q_PACKS;
        reinterpret_cast<uint4*>(&q_smem[row][0])[pack] =
            reinterpret_cast<const uint4*>(&Q[(q_row_base + row) * D])[pack];
    }
    __syncthreads();

    float m_acc = -1.0e30f;
    float l_acc = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    constexpr int PARENTS_PER_CHUNK = COLS / BF;
    constexpr int COLS_PER_LANE = COLS / WARP_SIZE;
    static_assert(COLS % WARP_SIZE == 0, "COLS must be divisible by warp size");

    for (int p_chunk_start = p_start; p_chunk_start < p_end; p_chunk_start += PARENTS_PER_CHUNK) {
        const int cols_this_chunk = min(COLS, (p_end - p_chunk_start) * BF);

        if (tid < COLS) {
            const int col = tid;
            const int parent = p_chunk_start + (col / BF);
            parent_smem[col] = parent;
            invalid_smem[col] = (col < cols_this_chunk)
                ? InvalidBlocks[(kvh * K + parent) * BF + (col % BF)]
                : 1;
            col_live_smem[col] = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < GROUPS * COLS; idx += blockDim.x) {
            const int row = idx / COLS;
            const int col = idx % COLS;
            int8_t alive = 0;
            if (row < GROUPS && col < cols_this_chunk) {
                const int parent = parent_smem[col];
                alive = invalid_smem[col] == 0
                    && ClusterPass[(ANCHOR_S * H_Q + q_row_base + row) * K + parent] != 0;
            }
            row_gate_smem[row][col] = alive;
        }
        __syncthreads();

        if (tid < COLS) {
            int8_t live = 0;
            if (tid < cols_this_chunk && invalid_smem[tid] == 0) {
                #pragma unroll
                for (int row = 0; row < MAX_GROUPS; ++row) {
                    if (row < GROUPS && row_gate_smem[row][tid]) {
                        live = 1;
                        break;
                    }
                }
            }
            col_live_smem[tid] = live;
        }
        __syncthreads();

        constexpr int K_PACKS = D / 8;
        for (int pack_idx = tid; pack_idx < COLS * K_PACKS; pack_idx += blockDim.x) {
            const int col = pack_idx / K_PACKS;
            const int pack = pack_idx % K_PACKS;
            uint4 v = {};
            if (col < cols_this_chunk && col_live_smem[col]) {
                const int parent = parent_smem[col];
                const int child = col % BF;
                v = reinterpret_cast<const uint4*>(
                    &KeysBlocks[((kvh * K + parent) * BF + child) * D]
                )[pack];
            }
            reinterpret_cast<uint4*>(&k_smem[col][0])[pack] = v;
        }

        constexpr int V_PACKS = D_V / 8;
        for (int pack_idx = tid; pack_idx < COLS * V_PACKS; pack_idx += blockDim.x) {
            const int col = pack_idx / V_PACKS;
            const int pack = pack_idx % V_PACKS;
            uint4 v = {};
            if (col < cols_this_chunk && col_live_smem[col]) {
                const int parent = parent_smem[col];
                const int child = col % BF;
                v = reinterpret_cast<const uint4*>(
                    &ValuesBlocks[((kvh * K + parent) * BF + child) * D_V]
                )[pack];
            }
            reinterpret_cast<uint4*>(&v_smem[col][0])[pack] = v;
        }
        __syncthreads();

        if (row_valid) {
            for (int col = 0; col < cols_this_chunk; ++col) {
                float dot = 0.0f;
                if (row_gate_smem[warp_id][col]) {
                    #pragma unroll
                    for (int d = lane * 4; d < D; d += WARP_SIZE * 4) {
                        const __half2 q0 =
                            *reinterpret_cast<const __half2*>(&q_smem[warp_id][d + 0]);
                        const __half2 q1 =
                            *reinterpret_cast<const __half2*>(&q_smem[warp_id][d + 2]);
                        const __half2 k0 =
                            *reinterpret_cast<const __half2*>(&k_smem[col][d + 0]);
                        const __half2 k1 =
                            *reinterpret_cast<const __half2*>(&k_smem[col][d + 2]);
                        const float2 q0f = __half22float2(q0);
                        const float2 q1f = __half22float2(q1);
                        const float2 k0f = __half22float2(k0);
                        const float2 k1f = __half22float2(k1);
                        dot += q0f.x * k0f.x + q0f.y * k0f.y;
                        dot += q1f.x * k1f.x + q1f.y * k1f.y;
                    }
                }
                dot = warp_reduce_sum(dot);
                if (lane == 0) {
                    score_smem[warp_id][col] = row_gate_smem[warp_id][col]
                        ? dot * SCALE
                        : -1.0e30f;
                }
            }
            __syncwarp();

            float local_max = -1.0e30f;
            for (int col = lane; col < cols_this_chunk; col += WARP_SIZE) {
                local_max = fmaxf(local_max, score_smem[warp_id][col]);
            }
            const float chunk_max = __shfl_sync(
                0xffffffff,
                warp_reduce_max(local_max),
                0
            );
            if (chunk_max > -1.0e29f) {
                const float new_m = fmaxf(m_acc, chunk_max);
                const float alpha = __expf(m_acc - new_m);
                float local_sum = 0.0f;

                for (int col = lane; col < cols_this_chunk; col += WARP_SIZE) {
                    float p = 0.0f;
                    const float score = score_smem[warp_id][col];
                    if (score > -1.0e29f) {
                        p = __expf(score - new_m);
                    }
                    p_smem[warp_id][col] = __float2half_rn(p);
                    local_sum += p;
                }

                const float chunk_sum = __shfl_sync(
                    0xffffffff,
                    warp_reduce_sum(local_sum),
                    0
                );
                l_acc = alpha * l_acc + chunk_sum;
                __syncwarp();

                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int dv = lane * 4 + j;
                    float acc = 0.0f;
                    if (dv < D_V) {
                        #pragma unroll 8
                        for (int col = 0; col < COLS; ++col) {
                            if (col < cols_this_chunk) {
                                acc += __half2float(p_smem[warp_id][col]) *
                                       __half2float(v_smem[col][dv]);
                            }
                        }
                    }
                    o_acc[j] = alpha * o_acc[j] + acc;
                }
                m_acc = new_m;
            }
        }
        __syncthreads();
    }

    if (row_valid) {
        if (lane == 0) {
            Out_M[hq * NUM_SPLITS + split] = m_acc;
            Out_L[hq * NUM_SPLITS + split] = l_acc;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dv = lane * 4 + j;
            if (dv < D_V) {
                Out_O[(hq * NUM_SPLITS + split) * D_V + dv] = o_acc[j];
            }
        }
    }
}

}  // namespace

void sparse_attn_index_v1_32(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor cluster_pass,
    torch::Tensor invalid,
    torch::Tensor out_m,
    torch::Tensor out_l,
    torch::Tensor out_o,
    int64_t h_q,
    int64_t h_kv,
    int64_t k_parents,
    int64_t num_splits,
    int64_t groups,
    int64_t anchor_s,
    double scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(cluster_pass.is_cuda(), "cluster_pass must be CUDA");
    TORCH_CHECK(invalid.is_cuda(), "invalid must be CUDA");
    TORCH_CHECK(out_m.is_cuda(), "out_m must be CUDA");
    TORCH_CHECK(out_l.is_cuda(), "out_l must be CUDA");
    TORCH_CHECK(out_o.is_cuda(), "out_o must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(keys.dtype() == torch::kFloat16, "keys must be fp16");
    TORCH_CHECK(values.dtype() == torch::kFloat16, "values must be fp16");
    TORCH_CHECK(cluster_pass.dtype() == torch::kInt8, "cluster_pass must be int8");
    TORCH_CHECK(invalid.dtype() == torch::kInt8, "invalid must be int8");
    TORCH_CHECK(out_m.dtype() == torch::kFloat32, "out_m must be fp32");
    TORCH_CHECK(out_l.dtype() == torch::kFloat32, "out_l must be fp32");
    TORCH_CHECK(out_o.dtype() == torch::kFloat32, "out_o must be fp32");

    constexpr int D = 128;
    constexpr int D_V = 128;
    constexpr int BF = 4;
    constexpr int COLS = 64;
    constexpr int MAX_GROUPS = 8;

    TORCH_CHECK(q.size(1) == D, "v1.32 expects D=128");
    TORCH_CHECK(keys.size(2) == BF, "v1.32 expects BF=4");
    TORCH_CHECK(keys.size(3) == D, "v1.32 expects contiguous key rows");
    TORCH_CHECK(values.size(2) == BF, "v1.32 expects BF=4 values");
    TORCH_CHECK(values.size(3) == D_V, "v1.32 expects D_v=128");
    TORCH_CHECK(groups <= MAX_GROUPS, "v1.32 expects groups <= 8");
    (void)h_q;
    (void)h_kv;
    (void)anchor_s;

    dim3 grid((int)h_kv, (int)num_splits, 1);
    dim3 block(MAX_GROUPS * WARP_SIZE, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_attn_index_v1_32_simt_kernel<D, D_V, BF, COLS, MAX_GROUPS>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(keys.data_ptr()),
            reinterpret_cast<const __half*>(values.data_ptr()),
            reinterpret_cast<const int8_t*>(cluster_pass.data_ptr()),
            reinterpret_cast<const int8_t*>(invalid.data_ptr()),
            out_m.data_ptr<float>(),
            out_l.data_ptr<float>(),
            out_o.data_ptr<float>(),
            (int)h_q,
            (int)k_parents,
            (int)num_splits,
            (int)groups,
            (int)anchor_s,
            (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sparse_attn_index_v1_32",
        &sparse_attn_index_v1_32,
        "v1.32 sparse attention index kernel (CUDA, chunkwise SIMT)");
}
