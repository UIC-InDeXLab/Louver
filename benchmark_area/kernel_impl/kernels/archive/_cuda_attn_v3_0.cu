// v3.0 native CUDA sparse attention.
//
// Compared with v1.46:
//   - keeps the native fused index kernel
//   - adds a native reduce+buffer kernel so the hot path is fully CUDA
//   - leaves split selection to Python-side heuristics instead of hard-coding

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMacros.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int WARP_SIZE = 32;
constexpr float NEG_INF = -1.0e30f;
constexpr float LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ unsigned int cvta_to_shared_u32(const void* p) {
    unsigned int addr;
    asm("{ .reg .u64 tmp; cvta.to.shared.u64 tmp, %1; cvt.u32.u64 %0, tmp; }"
        : "=r"(addr)
        : "l"(p));
    return addr;
}

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    unsigned int dst = cvta_to_shared_u32(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(gmem));
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
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

template <int D>
__device__ __forceinline__ float dot_half_row(
    const __half* __restrict__ q_row,
    const __half* __restrict__ k_row
) {
    float dot = 0.0f;
    #pragma unroll
    for (int d = 0; d < D; d += 8) {
        const float2 q0 = __half22float2(*reinterpret_cast<const __half2*>(q_row + d));
        const float2 q1 = __half22float2(*reinterpret_cast<const __half2*>(q_row + d + 2));
        const float2 q2 = __half22float2(*reinterpret_cast<const __half2*>(q_row + d + 4));
        const float2 q3 = __half22float2(*reinterpret_cast<const __half2*>(q_row + d + 6));
        const float2 k0 = __half22float2(*reinterpret_cast<const __half2*>(k_row + d));
        const float2 k1 = __half22float2(*reinterpret_cast<const __half2*>(k_row + d + 2));
        const float2 k2 = __half22float2(*reinterpret_cast<const __half2*>(k_row + d + 4));
        const float2 k3 = __half22float2(*reinterpret_cast<const __half2*>(k_row + d + 6));
        dot += q0.x * k0.x + q0.y * k0.y
             + q1.x * k1.x + q1.y * k1.y
             + q2.x * k2.x + q2.y * k2.y
             + q3.x * k3.x + q3.y * k3.y;
    }
    return dot;
}

template <int D, int D_V, int BF, int MAX_GROUPS, int AW, int COLS>
__global__ void __launch_bounds__(MAX_GROUPS * WARP_SIZE, 1)
sparse_attn_v3_0_index_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ KeysBlocks,
    const __half* __restrict__ ValuesBlocks,
    const __half* __restrict__ CentersAnchor,
    const __half* __restrict__ RadiiAnchor,
    const __half* __restrict__ ThAnchor,
    const __half* __restrict__ QNormAnchor,
    const int8_t* __restrict__ InvalidBlocks,
    float* __restrict__ Out_M,
    float* __restrict__ Out_L,
    float* __restrict__ Out_O,
    int H_Q,
    int K,
    int NUM_SPLITS,
    int GROUPS,
    int DIM_OFFSET,
    float SCALE
) {
    constexpr int PPC = COLS / BF;
    constexpr int CPL = COLS / WARP_SIZE;
    constexpr int K_PAD = 8;
    constexpr int K_STRIDE = D + K_PAD;
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

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* q_smem = reinterpret_cast<__half*>(smem_raw);
    __half* k_smem = q_smem + MAX_GROUPS * D;
    __half* v_smem = k_smem + COLS * K_STRIDE;
    __half* anc_smem = v_smem + COLS * D_V;
    float* rad_smem = reinterpret_cast<float*>(anc_smem + PPC * AW);
    int8_t* inv_smem = reinterpret_cast<int8_t*>(rad_smem + PPC);
    int8_t* col_live = inv_smem + COLS;
    int8_t* row_gate = col_live + COLS;
    __half* p_smem = reinterpret_cast<__half*>(row_gate + ((MAX_GROUPS * PPC + 7) & ~7));

    constexpr int Q_PACKS = D / 8;
    for (int i = tid; i < GROUPS * Q_PACKS; i += BLOCK) {
        const int row = i / Q_PACKS;
        const int pack = i % Q_PACKS;
        reinterpret_cast<uint4*>(&q_smem[row * D])[pack] =
            reinterpret_cast<const uint4*>(&Q[(q_row_base + row) * D])[pack];
    }
    for (int i = tid; i < (MAX_GROUPS - GROUPS) * Q_PACKS; i += BLOCK) {
        const int row = GROUPS + i / Q_PACKS;
        const int pack = i % Q_PACKS;
        if (row < MAX_GROUPS) {
            reinterpret_cast<uint4*>(&q_smem[row * D])[pack] = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    float m_acc = NEG_INF;
    float l_acc = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float my_th = row_valid ? __half2float(ThAnchor[hq]) : 1e30f;
    const float my_qn = row_valid ? __half2float(QNormAnchor[hq]) : 0.0f;

    for (int pc = p_start; pc < p_end; pc += PPC) {
        const int chunk_p = min(PPC, p_end - pc);
        const int chunk_cols = chunk_p * BF;

        for (int i = tid; i < chunk_p * AW; i += BLOCK) {
            const int pr = i / AW;
            const int w = i % AW;
            anc_smem[pr * AW + w] = CentersAnchor[(kvh * K + pc + pr) * AW + w];
        }
        for (int i = tid; i < chunk_p; i += BLOCK) {
            rad_smem[i] = __half2float(RadiiAnchor[kvh * K + pc + i]);
        }
        for (int c = tid; c < COLS; c += BLOCK) {
            inv_smem[c] = (c < chunk_cols)
                ? InvalidBlocks[(kvh * K + pc + c / BF) * BF + c % BF]
                : 1;
            col_live[c] = 0;
        }
        for (int i = tid; i < MAX_GROUPS * PPC; i += BLOCK) {
            row_gate[i] = 0;
        }
        __syncthreads();

        if (row_valid) {
            for (int pr = 0; pr < chunk_p; ++pr) {
                float dot = 0.0f;
                for (int w = lane; w < AW; w += WARP_SIZE) {
                    dot += __half2float(q_smem[warp_id * D + DIM_OFFSET + w])
                         * __half2float(anc_smem[pr * AW + w]);
                }
                dot = warp_reduce_sum(dot);
                const float ub = dot + rad_smem[pr] * my_qn;
                const int ok = __shfl_sync(0xffffffff, (ub >= my_th) ? 1 : 0, 0);
                if (ok) {
                    if (lane == 0) {
                        row_gate[warp_id * PPC + pr] = 1;
                    }
                    if (lane < BF) {
                        const int c = pr * BF + lane;
                        if (c < chunk_cols && inv_smem[c] == 0) {
                            col_live[c] = 1;
                        }
                    }
                }
            }
        }
        __syncthreads();

        int any = 0;
        for (int c = lane; c < chunk_cols; c += WARP_SIZE) {
            if (col_live[c]) {
                any = 1;
            }
        }
        if (!__any_sync(0xffffffff, any)) {
            __syncthreads();
            continue;
        }

        constexpr int K_PACKS = D / 8;
        for (int i = tid; i < chunk_cols * K_PACKS; i += BLOCK) {
            const int col = i / K_PACKS;
            const int pack = i % K_PACKS;
            if (col_live[col]) {
                cp_async_16B(
                    &k_smem[col * K_STRIDE + pack * 8],
                    &KeysBlocks[((kvh * K + pc + col / BF) * BF + col % BF) * D + pack * 8]
                );
            } else {
                reinterpret_cast<uint4*>(&k_smem[col * K_STRIDE + pack * 8])[0] = make_uint4(0, 0, 0, 0);
            }
        }
        for (int c = tid; c < chunk_cols; c += BLOCK) {
            #pragma unroll
            for (int p = 0; p < K_PAD; p += 2) {
                reinterpret_cast<unsigned*>(&k_smem[c * K_STRIDE + D + p])[0] = 0;
            }
        }
        cp_async_commit();

        constexpr int V_PACKS = D_V / 8;
        for (int i = tid; i < chunk_cols * V_PACKS; i += BLOCK) {
            const int col = i / V_PACKS;
            const int pack = i % V_PACKS;
            if (col_live[col]) {
                cp_async_16B(
                    &v_smem[col * D_V + pack * 8],
                    &ValuesBlocks[((kvh * K + pc + col / BF) * BF + col % BF) * D_V + pack * 8]
                );
            } else {
                reinterpret_cast<uint4*>(&v_smem[col * D_V + pack * 8])[0] = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();

        cp_async_wait_group_1();
        __syncthreads();

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
                float score = NEG_INF;
                if (alive) {
                    const __half* q_row = &q_smem[warp_id * D];
                    const __half* k_row = &k_smem[col * K_STRIDE];
                    score = dot_half_row<D>(q_row, k_row) * SCALE;
                }
                scores[ci] = score;
            }

            float local_max = NEG_INF;
            #pragma unroll
            for (int ci = 0; ci < CPL; ++ci) {
                local_max = fmaxf(local_max, scores[ci]);
            }
            const float chunk_max = warp_reduce_max(local_max);
            if (chunk_max > NEG_INF + 1.0f) {
                const float new_m = fmaxf(m_acc, chunk_max);
                alpha_local = exp2f((m_acc - new_m) * LOG2E_F);
                float local_sum = 0.0f;

                #pragma unroll
                for (int ci = 0; ci < CPL; ++ci) {
                    float p = 0.0f;
                    if (scores[ci] > NEG_INF + 1.0f) {
                        p = exp2f((scores[ci] - new_m) * LOG2E_F);
                    }
                    const int col = lane + ci * WARP_SIZE;
                    p_smem[warp_id * COLS + col] = __float2half_rn(p);
                    local_sum += p;
                }
                const float chunk_sum = __shfl_sync(0xffffffff, warp_reduce_sum(local_sum), 0);
                l_acc = alpha_local * l_acc + chunk_sum;
                m_acc = new_m;
                need_pv = true;
            }
        }

        cp_async_wait_all();
        __syncthreads();

        if (need_pv) {
            float v_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int c = 0; c < chunk_cols; ++c) {
                const float pv = __half2float(p_smem[warp_id * COLS + c]);
                if (pv > 0.0f) {
                    const float2 v01 = __half22float2(
                        *reinterpret_cast<const __half2*>(&v_smem[c * D_V + lane * 4]));
                    const float2 v23 = __half22float2(
                        *reinterpret_cast<const __half2*>(&v_smem[c * D_V + lane * 4 + 2]));
                    v_acc[0] += pv * v01.x;
                    v_acc[1] += pv * v01.y;
                    v_acc[2] += pv * v23.x;
                    v_acc[3] += pv * v23.y;
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                o_acc[j] = alpha_local * o_acc[j] + v_acc[j];
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

__global__ void attn_reduce_v3_0_kernel(
    const float* __restrict__ M_idx,
    const float* __restrict__ L_idx,
    const float* __restrict__ O_idx,
    const float* __restrict__ M_buf,
    const float* __restrict__ L_buf,
    const float* __restrict__ O_buf,
    float* __restrict__ Out,
    int NUM_SPLITS,
    int D_V
) {
    const int hq = blockIdx.x;
    const int lane = threadIdx.x;

    float m_g = NEG_INF;
    for (int s = 0; s < NUM_SPLITS; ++s) {
        m_g = fmaxf(m_g, M_idx[hq * NUM_SPLITS + s]);
    }
    m_g = fmaxf(m_g, M_buf[hq]);

    float l_s = 0.0f;
    for (int s = 0; s < NUM_SPLITS; ++s) {
        const float a = exp2f((M_idx[hq * NUM_SPLITS + s] - m_g) * LOG2E_F);
        l_s += a * L_idx[hq * NUM_SPLITS + s];
    }
    const float a_b = exp2f((M_buf[hq] - m_g) * LOG2E_F);
    l_s += a_b * L_buf[hq];
    const float l_safe = l_s > 0.0f ? l_s : 1.0f;

    for (int dv = lane; dv < D_V; dv += WARP_SIZE) {
        float o = 0.0f;
        for (int s = 0; s < NUM_SPLITS; ++s) {
            const float a = exp2f((M_idx[hq * NUM_SPLITS + s] - m_g) * LOG2E_F);
            o += a * O_idx[(hq * NUM_SPLITS + s) * D_V + dv];
        }
        o += a_b * O_buf[hq * D_V + dv];
        Out[hq * D_V + dv] = o / l_safe;
    }
}

template <int D, int D_V>
__global__ void __launch_bounds__(WARP_SIZE, 1) attn_reduce_buffer_v3_0_kernel(
    const __half* __restrict__ Q,
    const float* __restrict__ M_idx,
    const float* __restrict__ L_idx,
    const float* __restrict__ O_idx,
    const __half* __restrict__ BufKeys,
    const __half* __restrict__ BufValues,
    const int8_t* __restrict__ BufInvalid,
    float* __restrict__ Out,
    int NUM_SPLITS,
    int GROUPS,
    int L_BUF_MAX,
    float SCALE
) {
    const int hq = blockIdx.x;
    const int lane = threadIdx.x;
    const int kvh = GROUPS > 1 ? (hq / GROUPS) : hq;

    __shared__ __half q_smem[D];
    for (int d = lane; d < D; d += WARP_SIZE) {
        q_smem[d] = Q[hq * D + d];
    }
    __syncthreads();

    float m_g = NEG_INF;
    for (int s = 0; s < NUM_SPLITS; ++s) {
        m_g = fmaxf(m_g, M_idx[hq * NUM_SPLITS + s]);
    }

    float m_buf_local = NEG_INF;
    for (int c = lane; c < L_BUF_MAX; c += WARP_SIZE) {
        if (BufInvalid[kvh * L_BUF_MAX + c] == 0) {
            const __half* k_row = &BufKeys[(kvh * L_BUF_MAX + c) * D];
            const float score = dot_half_row<D>(q_smem, k_row) * SCALE;
            m_buf_local = fmaxf(m_buf_local, score);
        }
    }
    const float m_buf = __shfl_sync(0xffffffff, warp_reduce_max(m_buf_local), 0);
    m_g = fmaxf(m_g, m_buf);

    float l_total = 0.0f;
    float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int s = 0; s < NUM_SPLITS; ++s) {
        const float a = exp2f((M_idx[hq * NUM_SPLITS + s] - m_g) * LOG2E_F);
        l_total += a * L_idx[hq * NUM_SPLITS + s];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int dv = lane * 4 + j;
            if (dv < D_V) {
                o_acc[j] += a * O_idx[(hq * NUM_SPLITS + s) * D_V + dv];
            }
        }
    }

    float l_buf_local = 0.0f;
    for (int base = 0; base < L_BUF_MAX; base += WARP_SIZE) {
        const int c = base + lane;
        float p = 0.0f;
        if (c < L_BUF_MAX && BufInvalid[kvh * L_BUF_MAX + c] == 0) {
            const __half* k_row = &BufKeys[(kvh * L_BUF_MAX + c) * D];
            const float score = dot_half_row<D>(q_smem, k_row) * SCALE;
            p = exp2f((score - m_g) * LOG2E_F);
        }
        l_buf_local += p;

        #pragma unroll
        for (int src = 0; src < WARP_SIZE; ++src) {
            const int c_tok = base + src;
            const float p_tok = __shfl_sync(0xffffffff, p, src);
            if (c_tok >= L_BUF_MAX || p_tok <= 0.0f) {
                continue;
            }
            const int dv0 = lane * 4;
            if (dv0 < D_V) {
                const __half* v_row = &BufValues[(kvh * L_BUF_MAX + c_tok) * D_V + dv0];
                const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(v_row));
                const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(v_row + 2));
                o_acc[0] += p_tok * v01.x;
                o_acc[1] += p_tok * v01.y;
                o_acc[2] += p_tok * v23.x;
                o_acc[3] += p_tok * v23.y;
            }
        }
    }

    l_total += __shfl_sync(0xffffffff, warp_reduce_sum(l_buf_local), 0);
    const float l_safe = l_total > 0.0f ? l_total : 1.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int dv = lane * 4 + j;
        if (dv < D_V) {
            Out[hq * D_V + dv] = o_acc[j] / l_safe;
        }
    }
}

}  // namespace

void sparse_attn_v3_0_index(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor centers_anchor,
    torch::Tensor radii_anchor,
    torch::Tensor th_anchor,
    torch::Tensor qnorm_anchor,
    torch::Tensor invalid,
    torch::Tensor out_m,
    torch::Tensor out_l,
    torch::Tensor out_o,
    int64_t h_q,
    int64_t h_kv,
    int64_t k_parents,
    int64_t num_splits,
    int64_t groups,
    int64_t dim_offset,
    int64_t anchor_width,
    double scale,
    int64_t cols_per_chunk
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16);
    TORCH_CHECK(keys.dtype() == torch::kFloat16);
    TORCH_CHECK(values.dtype() == torch::kFloat16);

    constexpr int D = 128;
    constexpr int D_V = 128;
    constexpr int BF = 4;
    constexpr int MAX_GROUPS = 8;
    constexpr int BLOCK = MAX_GROUPS * WARP_SIZE;
    const int AW = static_cast<int>(anchor_width);
    const int COLS = static_cast<int>(cols_per_chunk);

    auto compute_smem = [&](int cols) -> size_t {
        const int ppc = cols / BF;
        const int k_stride = D + 8;
        return (MAX_GROUPS * D) * sizeof(__half)
             + (cols * k_stride) * sizeof(__half)
             + (cols * D_V) * sizeof(__half)
             + (ppc * AW) * sizeof(__half)
             + ppc * sizeof(float)
             + cols * sizeof(int8_t)
             + cols * sizeof(int8_t)
             + ((MAX_GROUPS * ppc + 7) & ~7) * sizeof(int8_t)
             + (MAX_GROUPS * cols) * sizeof(__half);
    };

    const size_t smem = compute_smem(COLS);
    dim3 grid(static_cast<int>(h_kv), static_cast<int>(num_splits));
    auto stream = at::cuda::getCurrentCUDAStream();

    #define L30(AW_T, COLS_T) do { \
        auto* kfn = &sparse_attn_v3_0_index_kernel<D, D_V, BF, MAX_GROUPS, AW_T, COLS_T>; \
        cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)); \
        kfn<<<grid, BLOCK, smem, stream>>>( \
            (const __half*)q.data_ptr(), \
            (const __half*)keys.data_ptr(), \
            (const __half*)values.data_ptr(), \
            (const __half*)centers_anchor.data_ptr(), \
            (const __half*)radii_anchor.data_ptr(), \
            (const __half*)th_anchor.data_ptr(), \
            (const __half*)qnorm_anchor.data_ptr(), \
            (const int8_t*)invalid.data_ptr(), \
            out_m.data_ptr<float>(), \
            out_l.data_ptr<float>(), \
            out_o.data_ptr<float>(), \
            static_cast<int>(h_q), \
            static_cast<int>(k_parents), \
            static_cast<int>(num_splits), \
            static_cast<int>(groups), \
            static_cast<int>(dim_offset), \
            static_cast<float>(scale)); \
    } while (0)

    if (COLS == 128) {
        if (AW == 16) { L30(16, 128); }
        else if (AW == 32) { L30(32, 128); }
        else if (AW == 64) { L30(64, 128); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else if (COLS == 64) {
        if (AW == 16) { L30(16, 64); }
        else if (AW == 32) { L30(32, 64); }
        else if (AW == 64) { L30(64, 64); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else if (COLS == 32) {
        if (AW == 16) { L30(16, 32); }
        else if (AW == 32) { L30(32, 32); }
        else if (AW == 64) { L30(64, 32); }
        else { TORCH_CHECK(false, "anchor_width must be 16/32/64"); }
    } else {
        TORCH_CHECK(false, "cols_per_chunk must be 32/64/128");
    }
    #undef L30
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_attn_v3_0_reduce(
    torch::Tensor m_idx,
    torch::Tensor l_idx,
    torch::Tensor o_idx,
    torch::Tensor m_buf,
    torch::Tensor l_buf,
    torch::Tensor o_buf,
    torch::Tensor out,
    int64_t num_splits
) {
    const int h_q = static_cast<int>(m_idx.size(0));
    const int d_v = static_cast<int>(o_idx.size(-1));
    auto stream = at::cuda::getCurrentCUDAStream();
    attn_reduce_v3_0_kernel<<<h_q, WARP_SIZE, 0, stream>>>(
        m_idx.data_ptr<float>(),
        l_idx.data_ptr<float>(),
        o_idx.data_ptr<float>(),
        m_buf.data_ptr<float>(),
        l_buf.data_ptr<float>(),
        o_buf.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(num_splits),
        d_v
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_attn_v3_0_reduce_buffer(
    torch::Tensor q,
    torch::Tensor m_idx,
    torch::Tensor l_idx,
    torch::Tensor o_idx,
    torch::Tensor buf_keys,
    torch::Tensor buf_values,
    torch::Tensor buf_invalid,
    int64_t num_splits,
    int64_t groups,
    double scale,
    torch::Tensor out
) {
    TORCH_CHECK(q.dtype() == torch::kFloat16);
    TORCH_CHECK(buf_keys.dtype() == torch::kFloat16);
    TORCH_CHECK(buf_values.dtype() == torch::kFloat16);
    TORCH_CHECK(buf_invalid.dtype() == torch::kInt8);

    constexpr int D = 128;
    constexpr int D_V = 128;
    const int h_q = static_cast<int>(q.size(0));
    const int l_buf_max = static_cast<int>(buf_keys.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    attn_reduce_buffer_v3_0_kernel<D, D_V><<<h_q, WARP_SIZE, 0, stream>>>(
        (const __half*)q.data_ptr(),
        m_idx.data_ptr<float>(),
        l_idx.data_ptr<float>(),
        o_idx.data_ptr<float>(),
        (const __half*)buf_keys.data_ptr(),
        (const __half*)buf_values.data_ptr(),
        (const int8_t*)buf_invalid.data_ptr(),
        out.data_ptr<float>(),
        static_cast<int>(num_splits),
        static_cast<int>(groups),
        l_buf_max,
        static_cast<float>(scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attn_v3_0_index", &sparse_attn_v3_0_index);
    m.def("sparse_attn_v3_0_reduce", &sparse_attn_v3_0_reduce);
    m.def("sparse_attn_v3_0_reduce_buffer", &sparse_attn_v3_0_reduce_buffer);
}
