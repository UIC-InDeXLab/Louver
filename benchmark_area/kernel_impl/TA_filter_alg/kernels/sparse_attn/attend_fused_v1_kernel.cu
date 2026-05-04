// attend_fused_v1 — single cooperative kernel that runs the whole attend
// pipeline: score top-L per (hq, s) → depth → parent-alive bitmap →
// sparse SDPA over (alive index keys ∪ buffer keys) → cross-split combine.
//
// Grid: (Hq, max(4, num_splits)).  Phases:
//   P1 (blk.y < 4): score top-L per (hq, s).         Hq*4 blocks.
//   grid.sync()
//   P2 (blk.y == 0): depth + per-subspace bitmap (smem).  Hq blocks.
//   grid.sync()
//   P3 (blk.y < num_splits): sparse softmax over its split, using the
//        bitmap held in *each block's smem* (re-built since smem isn't
//        shared across blocks).  Writes partial_m/l/o.   Hq*num_splits blocks.
//   grid.sync()
//   P4 (blk.y == 0): cross-split combine; writes out.    Hq blocks.
//
// One coop launch instead of two regular launches → saves one set of
// launch overhead + lets the depth_bitmap data stay in (per-block) smem
// throughout P3 instead of round-tripping through global memory.
//
// Complications:
//   * Each P3 block needs its own bitmap copy in smem.  We compute the
//     bitmap centrally in P2 (one block per hq) but P3 blocks have
//     different blockIdx.y; smem is not shared across blocks.  So we
//     stash the bitmap in *global* parent_alive_bitmap[Hq, 4, K_words]
//     after P2, then P3 blocks pull it back into their smem at start.
//     The global round-trip is small (~1KB per hq) and read-amortised.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <float.h>

namespace cg = cooperative_groups;

namespace {

constexpr int BLOCK   = 256;
constexpr int L       = 256;
constexpr int IPT_L   = 1;
constexpr int N_WARPS = BLOCK / 32;
constexpr int kDim    = 128;
constexpr int kPerLane = kDim / 32;

template <int IPT>
__device__ __forceinline__
void heap_push_l(float* keys, int* vals, float new_k, int new_v) {
    if (new_k <= keys[0]) return;
    keys[0] = new_k;
    vals[0] = new_v;
}

typedef cub::BlockRadixSort<float, BLOCK, IPT_L, int> BlockSort;

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__global__ void attend_fused_v1_kernel(
    // index inputs
    const half*    __restrict__ q,
    const half*    __restrict__ centers,
    const int32_t* __restrict__ dim_offsets,
    const int32_t* __restrict__ dim_widths,
    const int64_t* __restrict__ q_head_to_kv,
    const float*   __restrict__ threshold,
    const half*    __restrict__ keys,
    const half*    __restrict__ values,
    const half*    __restrict__ buffer_keys,
    const half*    __restrict__ buffer_values,
    const int64_t* __restrict__ assigns_packed,
    // workspace (also used cross-phase)
    float*         __restrict__ top_scores,
    int32_t*       __restrict__ top_indices,
    int32_t*       __restrict__ depth_g,
    uint32_t*      __restrict__ parent_alive_bitmap,
    float*         __restrict__ partial_m,
    float*         __restrict__ partial_l,
    float*         __restrict__ partial_o,
    int*           __restrict__ counters,
    half*          __restrict__ out,
    int Hq, int Hkv, int K, int max_w, int D,
    int Npad, int Bmax, int N_used, int K_words, int l_buf,
    int num_splits, float scale_log2e, int has_map)
{
    cg::grid_group grid = cg::this_grid();
    int hq    = blockIdx.x;
    int blk_y = blockIdx.y;
    int tid   = threadIdx.x;
    int lane  = tid & 31;
    int warp  = tid >> 5;

    int kvh = has_map ? (int)q_head_to_kv[hq] : hq;
    bool kvh_ok = (kvh >= 0 && kvh < Hkv);

    extern __shared__ uint8_t smem_raw[];

    // ── P1: SCORE (blk_y < 4) ──
    if (blk_y < 4) {
        int s    = blk_y;
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

    // ── P2: DEPTH + BITMAP (blk_y == 0) ──
    if (blk_y == 0) {
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
            int w = tid >> 5;
            int la = tid & 31;
            for (int vw = w; vw < N_VWARPS; vw += N_WARPS) {
                int ballot = __ballot_sync(0xFFFFFFFFu, smem_sums[vw * 32 + la] < th);
                if (la == 0)
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

    grid.sync();

    // ── P3: SPARSE SDPA per split (blk_y < num_splits) ──
    int split = blk_y;
    if (split < num_splits) {
        // Each P3 block holds its own copy of the per-(hq) bitmap in smem
        // (loaded from global). SmEM size: 4 * K_words * 4B + small.
        uint32_t* smem_bm = reinterpret_cast<uint32_t*>(smem_raw);
        int bm_total = 4 * K_words;
        const uint32_t* gbm = parent_alive_bitmap + (int64_t)hq * 4 * K_words;
        for (int i = tid; i < bm_total; i += BLOCK) smem_bm[i] = gbm[i];
        __syncthreads();

        const uint32_t* bm0 = smem_bm + 0 * K_words;
        const uint32_t* bm1 = smem_bm + 1 * K_words;
        const uint32_t* bm2 = smem_bm + 2 * K_words;
        const uint32_t* bm3 = smem_bm + 3 * K_words;
        const unsigned K_bits = (unsigned)(K_words * 32);

        int total_count    = N_used + l_buf;
        int span_per_split = (total_count + num_splits - 1) / num_splits;
        int start = split * span_per_split;
        int end   = start + span_per_split;
        if (end > total_count) end = total_count;
        int span = (start < end) ? (end - start) : 0;

        const half* qh           = q + hq * kDim;
        const half* kh_base      = keys   + (int64_t)kvh * Npad * kDim;
        const half* vh_base      = values + (int64_t)kvh * Npad * kDim;
        const half* buf_kh_base  = buffer_keys   + (int64_t)kvh * Bmax * kDim;
        const half* buf_vh_base  = buffer_values + (int64_t)kvh * Bmax * kDim;
        const int64_t* a_packed_h = assigns_packed + (int64_t)kvh * Npad;

        const half2* qh2 = reinterpret_cast<const half2*>(qh);
        half2 q_reg[2];
        q_reg[0] = qh2[lane];
        q_reg[1] = qh2[lane + 32];

        float o_acc[kPerLane];
        #pragma unroll
        for (int d = 0; d < kPerLane; d++) o_acc[d] = 0.0f;
        float m_w = -CUDART_INF_F;
        float l_w = 0.0f;
        bool seen = false;

        for (int i = warp; i < span; i += N_WARPS) {
            int combined = start + i;
            const half* kp;
            const half* vp;
            if (combined < N_used) {
                uint64_t packed = (uint64_t)__ldg((const unsigned long long*)(a_packed_h + combined));
                unsigned p0 = (unsigned)( packed        & 0xFFFFu);
                unsigned p1 = (unsigned)((packed >> 16) & 0xFFFFu);
                unsigned p2 = (unsigned)((packed >> 32) & 0xFFFFu);
                unsigned p3 = (unsigned)((packed >> 48) & 0xFFFFu);
                int hit = 0;
                if (p0 < K_bits) hit |= (int)((bm0[p0 >> 5] >> (p0 & 31)) & 1u);
                if (p1 < K_bits) hit |= (int)((bm1[p1 >> 5] >> (p1 & 31)) & 1u);
                if (p2 < K_bits) hit |= (int)((bm2[p2 >> 5] >> (p2 & 31)) & 1u);
                if (p3 < K_bits) hit |= (int)((bm3[p3 >> 5] >> (p3 & 31)) & 1u);
                if (!hit) continue;
                kp = kh_base + (int64_t)combined * kDim;
                vp = vh_base + (int64_t)combined * kDim;
            } else {
                int b = combined - N_used;
                kp = buf_kh_base + (int64_t)b * kDim;
                vp = buf_vh_base + (int64_t)b * kDim;
            }
            const half2* kh2 = reinterpret_cast<const half2*>(kp);
            const half2* vh2 = reinterpret_cast<const half2*>(vp);
            half2 k0 = kh2[lane];
            half2 k1 = kh2[lane + 32];
            half2 p0v = __hmul2(q_reg[0], k0);
            half2 p1v = __hmul2(q_reg[1], k1);
            float2 f0 = __half22float2(p0v);
            float2 f1 = __half22float2(p1v);
            float partial = (f0.x + f0.y) + (f1.x + f1.y);
            float s = warp_reduce_sum(partial) * scale_log2e;

            half2 v0 = vh2[lane];
            half2 v1 = vh2[lane + 32];
            float2 fv0 = __half22float2(v0);
            float2 fv1 = __half22float2(v1);

            if (!seen) {
                m_w = s; l_w = 1.0f;
                o_acc[0] = fv0.x; o_acc[1] = fv0.y;
                o_acc[2] = fv1.x; o_acc[3] = fv1.y;
                seen = true;
            } else {
                float m_new = fmaxf(m_w, s);
                float alpha = exp2f(m_w - m_new);
                float p     = exp2f(s - m_new);
                l_w = l_w * alpha + p;
                o_acc[0] = o_acc[0] * alpha + p * fv0.x;
                o_acc[1] = o_acc[1] * alpha + p * fv0.y;
                o_acc[2] = o_acc[2] * alpha + p * fv1.x;
                o_acc[3] = o_acc[3] * alpha + p * fv1.y;
                m_w = m_new;
            }
        }

        // cross-warp combine within block
        // Smem layout (after bitmap step done):
        //  smem_m[N_WARPS], smem_l[N_WARPS], smem_o[N_WARPS][kDim]
        // We allocate them after the bitmap region (already consumed).
        __syncthreads();
        float* sm_m = reinterpret_cast<float*>(smem_raw);
        float* sm_l = sm_m + N_WARPS;
        float* sm_o = sm_l + N_WARPS;

        if (lane == 0) {
            sm_m[warp] = seen ? m_w : -CUDART_INF_F;
            sm_l[warp] = l_w;
        }
        sm_o[warp * kDim + 2 * lane]      = o_acc[0];
        sm_o[warp * kDim + 2 * lane + 1]  = o_acc[1];
        sm_o[warp * kDim + 2 * lane + 64] = o_acc[2];
        sm_o[warp * kDim + 2 * lane + 65] = o_acc[3];
        __syncthreads();

        __shared__ float s_mg, s_lg;

        if (warp == 0) {
            float v = (lane < N_WARPS) ? sm_m[lane] : -CUDART_INF_F;
            v = warp_reduce_max(v);
            if (lane == 0) s_mg = v;
        }
        __syncthreads();
        float m_global = s_mg;

        float alpha_w;
        if (lane == 0) {
            float mw = sm_m[warp];
            alpha_w = (mw == -CUDART_INF_F) ? 0.0f : exp2f(mw - m_global);
        }
        alpha_w = __shfl_sync(0xffffffffu, alpha_w, 0);

        if (warp == 0) {
            float mw = (lane < N_WARPS) ? sm_m[lane] : -CUDART_INF_F;
            float a = (mw == -CUDART_INF_F) ? 0.0f : exp2f(mw - m_global);
            float v = (lane < N_WARPS) ? a * sm_l[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_lg = v;
        }
        sm_o[warp * kDim + 2 * lane]      *= alpha_w;
        sm_o[warp * kDim + 2 * lane + 1]  *= alpha_w;
        sm_o[warp * kDim + 2 * lane + 64] *= alpha_w;
        sm_o[warp * kDim + 2 * lane + 65] *= alpha_w;
        __syncthreads();

        if (tid < kDim) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; w++) acc += sm_o[w * kDim + tid];
            partial_o[(hq * num_splits + split) * kDim + tid] = acc;
        }
        if (tid == 0) {
            partial_m[hq * num_splits + split] = m_global;
            partial_l[hq * num_splits + split] = s_lg;
        }
    }

    grid.sync();

    // ── P4: cross-split combine (blk_y == 0) ──
    if (blk_y == 0) {
        __shared__ float sm_m[N_WARPS];
        __shared__ float sm_l[N_WARPS];
        __shared__ float s_mg2, s_lg2;

        float m_split = -CUDART_INF_F;
        for (int s = tid; s < num_splits; s += blockDim.x)
            m_split = fmaxf(m_split, partial_m[hq * num_splits + s]);
        m_split = warp_reduce_max(m_split);
        if (lane == 0) sm_m[warp] = m_split;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < N_WARPS) ? sm_m[lane] : -CUDART_INF_F;
            v = warp_reduce_max(v);
            if (lane == 0) s_mg2 = v;
        }
        __syncthreads();
        float m_g = s_mg2;

        float l_split_local = 0.0f;
        for (int s = tid; s < num_splits; s += blockDim.x) {
            float ms = partial_m[hq * num_splits + s];
            float a = (ms == -CUDART_INF_F) ? 0.0f : exp2f(ms - m_g);
            l_split_local += a * partial_l[hq * num_splits + s];
        }
        l_split_local = warp_reduce_sum(l_split_local);
        if (lane == 0) sm_l[warp] = l_split_local;
        __syncthreads();
        if (warp == 0) {
            float v = (lane < N_WARPS) ? sm_l[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) s_lg2 = v;
        }
        __syncthreads();
        float l_g = s_lg2;
        float inv_l = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;

        for (int dv = tid; dv < kDim; dv += blockDim.x) {
            float acc = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                float ms = partial_m[hq * num_splits + s];
                float a = (ms == -CUDART_INF_F) ? 0.0f : exp2f(ms - m_g);
                acc += a * partial_o[(hq * num_splits + s) * kDim + dv];
            }
            out[hq * kDim + dv] = __float2half(acc * inv_l);
        }
    }
}

}  // namespace

void attend_fused_v1_launch(
    torch::Tensor q,
    torch::Tensor centers,
    torch::Tensor dim_offsets, torch::Tensor dim_widths,
    torch::Tensor q_head_to_kv,
    torch::Tensor threshold,
    torch::Tensor keys, torch::Tensor values,
    torch::Tensor buffer_keys, torch::Tensor buffer_values,
    torch::Tensor assigns_packed,
    torch::Tensor top_scores, torch::Tensor top_indices,
    torch::Tensor depth, torch::Tensor parent_alive_bitmap,
    torch::Tensor partial_m, torch::Tensor partial_l, torch::Tensor partial_o,
    torch::Tensor counters, torch::Tensor out,
    double scale,
    int64_t N_used, int64_t l_buf, int64_t num_splits)
{
    int Hq      = (int)q.size(0);
    int D       = (int)q.size(1);
    int S       = (int)centers.size(0);
    int Hkv     = (int)centers.size(1);
    int K       = (int)centers.size(2);
    int max_w   = (int)centers.size(3);
    int Npad    = (int)keys.size(1);
    int Bmax    = (int)buffer_keys.size(1);
    int K_words = (K + 31) / 32;
    int splits  = (int)num_splits;

    bool has_map = q_head_to_kv.defined() && q_head_to_kv.numel() > 0;

    int max_y = (4 > splits) ? 4 : splits;
    dim3 grid(Hq, max_y);
    dim3 block(BLOCK);

    size_t smem_p1   = sizeof(BlockSort::TempStorage);
    size_t smem_p2   = std::max(L * sizeof(float) + (L / 32) * sizeof(int) + sizeof(int),
                                (size_t)4 * K_words * sizeof(uint32_t));
    size_t smem_p3   = (size_t)4 * K_words * sizeof(uint32_t)
                       + 2 * N_WARPS * sizeof(float)
                       + N_WARPS * kDim * sizeof(float);
    size_t smem_bytes = std::max(smem_p1, std::max(smem_p2, smem_p3));

    auto stream = at::cuda::getCurrentCUDAStream();

    const half*    q_ptr  = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half*    c_ptr  = reinterpret_cast<const half*>(centers.data_ptr<at::Half>());
    const int32_t* doff_p = dim_offsets.data_ptr<int32_t>();
    const int32_t* dwid_p = dim_widths.data_ptr<int32_t>();
    const int64_t* kv_ptr = has_map ? q_head_to_kv.data_ptr<int64_t>() : nullptr;
    const float*   th_ptr = threshold.data_ptr<float>();
    const half*    k_ptr  = reinterpret_cast<const half*>(keys.data_ptr<at::Half>());
    const half*    v_ptr  = reinterpret_cast<const half*>(values.data_ptr<at::Half>());
    const half*    bk_ptr = reinterpret_cast<const half*>(buffer_keys.data_ptr<at::Half>());
    const half*    bv_ptr = reinterpret_cast<const half*>(buffer_values.data_ptr<at::Half>());
    const int64_t* ap_ptr = assigns_packed.data_ptr<int64_t>();
    float*         ts_ptr = top_scores.data_ptr<float>();
    int32_t*       ti_ptr = top_indices.data_ptr<int32_t>();
    int32_t*       dp_ptr = depth.data_ptr<int32_t>();
    uint32_t*      bm_ptr = reinterpret_cast<uint32_t*>(parent_alive_bitmap.data_ptr<int32_t>());
    float*         pm_ptr = partial_m.data_ptr<float>();
    float*         pl_ptr = partial_l.data_ptr<float>();
    float*         po_ptr = partial_o.data_ptr<float>();
    int*           cn_ptr = counters.data_ptr<int>();
    half*          out_ptr= reinterpret_cast<half*>(out.data_ptr<at::Half>());
    int Nu = (int)N_used; int LB = (int)l_buf; int NS = splits; int hm_i = (int)has_map;
    float scale_log2e = (float)(scale * 1.4426950408889634);

    void* args[] = {
        (void*)&q_ptr, (void*)&c_ptr, (void*)&doff_p, (void*)&dwid_p,
        (void*)&kv_ptr, (void*)&th_ptr,
        (void*)&k_ptr, (void*)&v_ptr, (void*)&bk_ptr, (void*)&bv_ptr,
        (void*)&ap_ptr,
        (void*)&ts_ptr, (void*)&ti_ptr, (void*)&dp_ptr, (void*)&bm_ptr,
        (void*)&pm_ptr, (void*)&pl_ptr, (void*)&po_ptr, (void*)&cn_ptr,
        (void*)&out_ptr,
        (void*)&Hq, (void*)&Hkv, (void*)&K, (void*)&max_w, (void*)&D,
        (void*)&Npad, (void*)&Bmax, (void*)&Nu, (void*)&K_words, (void*)&LB,
        (void*)&NS, (void*)&scale_log2e, (void*)&hm_i,
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)attend_fused_v1_kernel,
        grid, block, args, smem_bytes, stream);
    TORCH_CHECK(err == cudaSuccess, "coop launch failed: ", cudaGetErrorString(err));
}
