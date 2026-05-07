#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int WARP_SIZE = 32;
constexpr float LOG2E_F = 1.4426950408889634f;

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

__device__ __forceinline__ float half2_dot16(
    const __half* __restrict__ a,
    const __half* __restrict__ b) {
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 af = __half22float2(reinterpret_cast<const __half2*>(a)[i]);
        const float2 bf = __half22float2(reinterpret_cast<const __half2*>(b)[i]);
        acc += af.x * bf.x + af.y * bf.y;
    }
    return acc;
}

template <int D, int D_V, int BF, int SUB_D, int COLS, int MAX_GROUPS>
__global__ void sparse_attn_index_v1_33_kernel(
    const __half* __restrict__ Q,             // (H_q, D)
    const __half* __restrict__ QNormAnchor,   // (H_q)
    const __half* __restrict__ ThAnchor,      // (H_q)
    const __half* __restrict__ CentersAnchor, // (H_kv, K, SUB_D)
    const __half* __restrict__ RadiiAnchor,   // (H_kv, K)
    const __half* __restrict__ KeysBlocks,    // (H_kv, K, BF, D)
    const __half* __restrict__ ValuesBlocks,  // (H_kv, K, BF, D_v)
    const int8_t* __restrict__ InvalidBlocks, // (H_kv, K, BF)
    float* __restrict__ Out_M,                // (H_q, NUM_SPLITS)
    float* __restrict__ Out_L,                // (H_q, NUM_SPLITS)
    float* __restrict__ Out_O,                // (H_q, NUM_SPLITS, D_v)
    int H_Q,
    int K,
    int NUM_SPLITS,
    int GROUPS,
    int ANCHOR_D_OFFSET,
    float SCALE) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    constexpr int PARENTS_PER_CHUNK = COLS / BF;
    static_assert(COLS % BF == 0, "COLS must be divisible by BF");

    __shared__ __half q_smem[MAX_GROUPS][D];
    __shared__ __half q_norm_smem[MAX_GROUPS];
    __shared__ __half th_smem[MAX_GROUPS];
    __shared__ __half center_smem[PARENTS_PER_CHUNK][SUB_D];
    __shared__ __half radius_smem[PARENTS_PER_CHUNK];
    __shared__ __half k_smem[COLS][D];
    __shared__ __half v_smem[COLS][D_V];
    __shared__ float score_smem[MAX_GROUPS][COLS];
    __shared__ __half p_smem[MAX_GROUPS][COLS];
    __shared__ int parent_smem[COLS];
    __shared__ int8_t invalid_smem[COLS];
    __shared__ int8_t parent_pass_smem[MAX_GROUPS][PARENTS_PER_CHUNK];
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
    for (int idx = tid; idx < GROUPS; idx += blockDim.x) {
        q_norm_smem[idx] = QNormAnchor[q_row_base + idx];
        th_smem[idx] = ThAnchor[q_row_base + idx];
    }
    __syncthreads();

    float m_acc = -1.0e30f;
    float l_acc = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int p_chunk_start = p_start; p_chunk_start < p_end; p_chunk_start += PARENTS_PER_CHUNK) {
        const int parents_this_chunk = min(PARENTS_PER_CHUNK, p_end - p_chunk_start);
        const int cols_this_chunk = parents_this_chunk * BF;

        for (int tidx = tid; tidx < COLS; tidx += blockDim.x) {
            const int parent_rel = tidx / BF;
            const int parent = p_chunk_start + parent_rel;
            parent_smem[tidx] = parent;
            invalid_smem[tidx] = (tidx < cols_this_chunk)
                ? InvalidBlocks[(kvh * K + parent) * BF + (tidx % BF)]
                : 1;
            col_live_smem[tidx] = 0;
        }
        constexpr int CENTER_PACKS = SUB_D / 8;
        for (int pack_idx = tid; pack_idx < PARENTS_PER_CHUNK * CENTER_PACKS; pack_idx += blockDim.x) {
            const int parent_rel = pack_idx / CENTER_PACKS;
            const int pack = pack_idx % CENTER_PACKS;
            uint4 v = {};
            if (parent_rel < parents_this_chunk) {
                v = reinterpret_cast<const uint4*>(
                    &CentersAnchor[(kvh * K + (p_chunk_start + parent_rel)) * SUB_D]
                )[pack];
            }
            reinterpret_cast<uint4*>(&center_smem[parent_rel][0])[pack] = v;
        }
        for (int idx = tid; idx < PARENTS_PER_CHUNK; idx += blockDim.x) {
            radius_smem[idx] = idx < parents_this_chunk
                ? RadiiAnchor[kvh * K + (p_chunk_start + idx)]
                : __float2half(0.0f);
        }
        __syncthreads();

        for (int idx = tid; idx < GROUPS * PARENTS_PER_CHUNK; idx += blockDim.x) {
            const int row = idx / PARENTS_PER_CHUNK;
            const int parent_rel = idx % PARENTS_PER_CHUNK;
            int8_t pass = 0;
            if (row < GROUPS && parent_rel < parents_this_chunk) {
                const float dot = half2_dot16(
                    &q_smem[row][ANCHOR_D_OFFSET],
                    &center_smem[parent_rel][0]
                );
                const float ub =
                    dot
                    + __half2float(radius_smem[parent_rel]) * __half2float(q_norm_smem[row]);
                pass = ub >= __half2float(th_smem[row]) ? 1 : 0;
            }
            parent_pass_smem[row][parent_rel] = pass;
        }
        __syncthreads();

        for (int idx = tid; idx < GROUPS * COLS; idx += blockDim.x) {
            const int row = idx / COLS;
            const int col = idx % COLS;
            int8_t alive = 0;
            if (row < GROUPS && col < cols_this_chunk) {
                alive = invalid_smem[col] == 0 && parent_pass_smem[row][col / BF] != 0;
            }
            row_gate_smem[row][col] = alive;
        }
        __syncthreads();

        for (int idx = tid; idx < COLS; idx += blockDim.x) {
            int8_t live = 0;
            if (idx < cols_this_chunk && invalid_smem[idx] == 0) {
                #pragma unroll
                for (int row = 0; row < MAX_GROUPS; ++row) {
                    if (row < GROUPS && row_gate_smem[row][idx]) {
                        live = 1;
                        break;
                    }
                }
            }
            col_live_smem[idx] = live;
        }
        __syncthreads();

        int chunk_live = 0;
        if (tid < cols_this_chunk) {
            chunk_live = col_live_smem[tid] != 0;
        }
        if (!__syncthreads_or(chunk_live != 0)) {
            continue;
        }

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
                const float alpha = exp2f((m_acc - new_m) * LOG2E_F);
                float local_sum = 0.0f;

                for (int col = lane; col < cols_this_chunk; col += WARP_SIZE) {
                    float p = 0.0f;
                    const float score = score_smem[warp_id][col];
                    if (score > -1.0e29f) {
                        p = exp2f((score - new_m) * LOG2E_F);
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

void sparse_attn_index_v1_33(
    torch::Tensor q,
    torch::Tensor q_norm_anchor,
    torch::Tensor th_anchor,
    torch::Tensor centers_anchor,
    torch::Tensor radii_anchor,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor invalid,
    torch::Tensor out_m,
    torch::Tensor out_l,
    torch::Tensor out_o,
    int64_t h_q,
    int64_t h_kv,
    int64_t k_parents,
    int64_t num_splits,
    int64_t groups,
    int64_t anchor_d_offset,
    double scale) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q_norm_anchor.is_cuda(), "q_norm_anchor must be CUDA");
    TORCH_CHECK(th_anchor.is_cuda(), "th_anchor must be CUDA");
    TORCH_CHECK(centers_anchor.is_cuda(), "centers_anchor must be CUDA");
    TORCH_CHECK(radii_anchor.is_cuda(), "radii_anchor must be CUDA");
    TORCH_CHECK(keys.is_cuda(), "keys must be CUDA");
    TORCH_CHECK(values.is_cuda(), "values must be CUDA");
    TORCH_CHECK(invalid.is_cuda(), "invalid must be CUDA");
    TORCH_CHECK(out_m.is_cuda(), "out_m must be CUDA");
    TORCH_CHECK(out_l.is_cuda(), "out_l must be CUDA");
    TORCH_CHECK(out_o.is_cuda(), "out_o must be CUDA");

    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be fp16");
    TORCH_CHECK(q_norm_anchor.dtype() == torch::kFloat16, "q_norm_anchor must be fp16");
    TORCH_CHECK(th_anchor.dtype() == torch::kFloat16, "th_anchor must be fp16");
    TORCH_CHECK(centers_anchor.dtype() == torch::kFloat16, "centers_anchor must be fp16");
    TORCH_CHECK(radii_anchor.dtype() == torch::kFloat16, "radii_anchor must be fp16");
    TORCH_CHECK(keys.dtype() == torch::kFloat16, "keys must be fp16");
    TORCH_CHECK(values.dtype() == torch::kFloat16, "values must be fp16");
    TORCH_CHECK(invalid.dtype() == torch::kInt8, "invalid must be int8");
    TORCH_CHECK(out_m.dtype() == torch::kFloat32, "out_m must be fp32");
    TORCH_CHECK(out_l.dtype() == torch::kFloat32, "out_l must be fp32");
    TORCH_CHECK(out_o.dtype() == torch::kFloat32, "out_o must be fp32");

    constexpr int D = 128;
    constexpr int D_V = 128;
    constexpr int BF = 4;
    constexpr int SUB_D = 16;
    constexpr int COLS = 64;
    constexpr int MAX_GROUPS = 8;

    TORCH_CHECK(q.size(1) == D, "v1.33 expects D=128");
    TORCH_CHECK(keys.size(2) == BF, "v1.33 expects BF=4");
    TORCH_CHECK(keys.size(3) == D, "v1.33 expects row-major key blocks");
    TORCH_CHECK(values.size(2) == BF, "v1.33 expects BF=4 values");
    TORCH_CHECK(values.size(3) == D_V, "v1.33 expects D_v=128");
    TORCH_CHECK(centers_anchor.size(2) == SUB_D, "v1.33 expects anchor width 16");
    TORCH_CHECK(groups <= MAX_GROUPS, "v1.33 expects groups <= 8");
    TORCH_CHECK(anchor_d_offset >= 0 && anchor_d_offset + SUB_D <= D, "invalid anchor slice");
    (void)h_q;

    dim3 grid((int)h_kv, (int)num_splits, 1);
    dim3 block(MAX_GROUPS * WARP_SIZE, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_attn_index_v1_33_kernel<D, D_V, BF, SUB_D, COLS, MAX_GROUPS>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(q.data_ptr()),
            reinterpret_cast<const __half*>(q_norm_anchor.data_ptr()),
            reinterpret_cast<const __half*>(th_anchor.data_ptr()),
            reinterpret_cast<const __half*>(centers_anchor.data_ptr()),
            reinterpret_cast<const __half*>(radii_anchor.data_ptr()),
            reinterpret_cast<const __half*>(keys.data_ptr()),
            reinterpret_cast<const __half*>(values.data_ptr()),
            reinterpret_cast<const int8_t*>(invalid.data_ptr()),
            out_m.data_ptr<float>(),
            out_l.data_ptr<float>(),
            out_o.data_ptr<float>(),
            (int)h_q,
            (int)k_parents,
            (int)num_splits,
            (int)groups,
            (int)anchor_d_offset,
            (float)scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "sparse_attn_index_v1_33",
        &sparse_attn_index_v1_33,
        "v1.33 sparse attention index kernel (CUDA, anchor-gated split-KV)");
}
