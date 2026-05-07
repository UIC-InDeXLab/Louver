// v2.5: v2.4 + buffer-aware tail loop.
//
// Adds optional ``buffer_keys`` / ``buffer_values`` tensors (H_kv, B, kDim)
// and an ``l_buf`` count.  Total per-head workload is ``live_count[hq] +
// l_buf``; for the first ``live_count[hq]`` keys we look up ``live_idx``;
// for the trailing ``l_buf`` keys we fetch the buffer slot directly.
//
// The split scheduler divides the combined total over num_splits so each
// split still owns a contiguous run of keys; per-warp branching is uniform
// across the 32 lanes (each warp processes one key), so divergence is nil.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps   = 8;
constexpr int kDim     = 128;
constexpr int kPerLane = kDim / 32;

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

// Load (k0, k1) half2 pair for combined index `i` in the (live_idx ∪ buffer)
// space.  Returns by reference into k0/k1.
__device__ __forceinline__ void load_kv_pair(
    int i, int live_count_eff,
    const int32_t* __restrict__ li_h_full,    // live_idx + hq * Npad
    const half*    __restrict__ kh_base,      // keys[kvh] base
    const half*    __restrict__ vh_base,      // values[kvh] base
    const half*    __restrict__ buf_kh_base,  // buffer_keys[kvh] base
    const half*    __restrict__ buf_vh_base,  // buffer_values[kvh] base
    int            lane,
    half2&         k0_out, half2& k1_out,
    half2&         v0_out, half2& v1_out)
{
    int n;
    const half* kp;
    const half* vp;
    if (i < live_count_eff) {
        n = li_h_full[i];
        kp = kh_base + (int64_t)n * kDim;
        vp = vh_base + (int64_t)n * kDim;
    } else {
        int b = i - live_count_eff;
        kp = buf_kh_base + (int64_t)b * kDim;
        vp = buf_vh_base + (int64_t)b * kDim;
    }
    const half2* kh2 = reinterpret_cast<const half2*>(kp);
    const half2* vh2 = reinterpret_cast<const half2*>(vp);
    k0_out = kh2[lane];
    k1_out = kh2[lane + 32];
    v0_out = vh2[lane];
    v1_out = vh2[lane + 32];
}

__global__ void sdpa_cuda_sparse_v2_5_kernel(
    const half*    __restrict__ q,
    const half*    __restrict__ keys,
    const half*    __restrict__ values,
    const half*    __restrict__ buffer_keys,    // (H_kv, B_max, kDim)
    const half*    __restrict__ buffer_values,  // (H_kv, B_max, kDim)
    const int32_t* __restrict__ live_idx,
    const int32_t* __restrict__ live_count,
    float*         __restrict__ partial_m,
    float*         __restrict__ partial_l,
    float*         __restrict__ partial_o,
    int*           __restrict__ counters,
    half*          __restrict__ out,
    int Hq, int Hkv, int Npad, int Bmax, int l_buf,
    int num_splits, float scale_log2e)
{
  int hq    = blockIdx.x;
  int split = blockIdx.y;
  int tid   = threadIdx.x;
  int lane  = tid & 31;
  int warp  = tid >> 5;
  int kvh   = hq / (Hq / Hkv);

  int live_count_eff = live_count[hq];
  int total_count    = live_count_eff + l_buf;
  int span_per_split = (total_count + num_splits - 1) / num_splits;
  int start = split * span_per_split;
  int end   = start + span_per_split;
  if (end > total_count) end = total_count;
  int span = (start < end) ? (end - start) : 0;

  const half*    qh           = q + hq * kDim;
  const half*    kh_base      = keys   + (int64_t)kvh * Npad * kDim;
  const half*    vh_base      = values + (int64_t)kvh * Npad * kDim;
  const half*    buf_kh_base  = buffer_keys   + (int64_t)kvh * Bmax * kDim;
  const half*    buf_vh_base  = buffer_values + (int64_t)kvh * Bmax * kDim;
  const int32_t* li_h_full    = live_idx + (int64_t)hq * Npad;

  const half2* qh2 = reinterpret_cast<const half2*>(qh);
  half2 q_reg[2];
  q_reg[0] = qh2[lane];
  q_reg[1] = qh2[lane + 32];

  float o_acc[kPerLane];
  float m_w = -CUDART_INF_F;
  float l_w = 0.0f;

  // ── First key peel ──
  int first_i = warp;
  if (first_i < span) {
    half2 k0, k1, v0, v1;
    load_kv_pair(start + first_i, live_count_eff,
                 li_h_full, kh_base, vh_base, buf_kh_base, buf_vh_base,
                 lane, k0, k1, v0, v1);
    half2 p0 = __hmul2(q_reg[0], k0);
    half2 p1 = __hmul2(q_reg[1], k1);
    float2 f0 = __half22float2(p0);
    float2 f1 = __half22float2(p1);
    float partial = (f0.x + f0.y) + (f1.x + f1.y);
    float s = warp_reduce_sum(partial) * scale_log2e;
    m_w = s;
    l_w = 1.0f;
    float2 fv0 = __half22float2(v0);
    float2 fv1 = __half22float2(v1);
    o_acc[0] = fv0.x;
    o_acc[1] = fv0.y;
    o_acc[2] = fv1.x;
    o_acc[3] = fv1.y;
  } else {
    #pragma unroll
    for (int d = 0; d < kPerLane; d++) o_acc[d] = 0.0f;
  }

  // ── Hot loop ──
  for (int i = first_i + kWarps; i < span; i += kWarps) {
    half2 k0, k1, v0, v1;
    load_kv_pair(start + i, live_count_eff,
                 li_h_full, kh_base, vh_base, buf_kh_base, buf_vh_base,
                 lane, k0, k1, v0, v1);
    half2 p0 = __hmul2(q_reg[0], k0);
    half2 p1 = __hmul2(q_reg[1], k1);
    float2 f0 = __half22float2(p0);
    float2 f1 = __half22float2(p1);
    float partial = (f0.x + f0.y) + (f1.x + f1.y);
    float s = warp_reduce_sum(partial) * scale_log2e;

    float m_new = fmaxf(m_w, s);
    float alpha = exp2f(m_w - m_new);
    float p     = exp2f(s - m_new);
    l_w = l_w * alpha + p;

    float2 fv0 = __half22float2(v0);
    float2 fv1 = __half22float2(v1);
    o_acc[0] = o_acc[0] * alpha + p * fv0.x;
    o_acc[1] = o_acc[1] * alpha + p * fv0.y;
    o_acc[2] = o_acc[2] * alpha + p * fv1.x;
    o_acc[3] = o_acc[3] * alpha + p * fv1.y;
    m_w = m_new;
  }

  // ── Cross-warp combine. ──
  __shared__ float smem_m[kWarps];
  __shared__ float smem_l[kWarps];
  __shared__ float smem_o[kWarps][kDim];
  __shared__ float s_mg, s_lg;

  if (lane == 0) {
    smem_m[warp] = m_w;
    smem_l[warp] = l_w;
  }
  smem_o[warp][2 * lane]      = o_acc[0];
  smem_o[warp][2 * lane + 1]  = o_acc[1];
  smem_o[warp][2 * lane + 64] = o_acc[2];
  smem_o[warp][2 * lane + 65] = o_acc[3];
  __syncthreads();

  if (warp == 0) {
    float v = (lane < kWarps) ? smem_m[lane] : -CUDART_INF_F;
    v = warp_reduce_max(v);
    if (lane == 0) s_mg = v;
  }
  __syncthreads();
  float m_global = s_mg;

  float alpha_w;
  if (lane == 0) {
    float mw = smem_m[warp];
    alpha_w = (mw == -CUDART_INF_F) ? 0.0f : exp2f(mw - m_global);
  }
  alpha_w = __shfl_sync(0xffffffffu, alpha_w, 0);

  if (warp == 0) {
    float mw = (lane < kWarps) ? smem_m[lane] : -CUDART_INF_F;
    float a = (mw == -CUDART_INF_F) ? 0.0f : exp2f(mw - m_global);
    float v = (lane < kWarps) ? a * smem_l[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) s_lg = v;
  }
  smem_o[warp][2 * lane]      *= alpha_w;
  smem_o[warp][2 * lane + 1]  *= alpha_w;
  smem_o[warp][2 * lane + 64] *= alpha_w;
  smem_o[warp][2 * lane + 65] *= alpha_w;
  __syncthreads();

  float l_global = s_lg;

  if (tid < kDim) {
    float acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWarps; w++) {
      acc += smem_o[w][tid];
    }
    partial_o[(hq * num_splits + split) * kDim + tid] = acc;
  }

  if (tid == 0) {
    partial_m[hq * num_splits + split] = m_global;
    partial_l[hq * num_splits + split] = l_global;
  }
  __syncthreads();

  // ── Cross-split last-block combine. ──
  __shared__ bool is_last;
  if (tid == 0) {
    int old;
    asm volatile("atom.release.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(old) : "l"(counters + hq) : "memory");
    is_last = (old == num_splits - 1);
  }
  __syncthreads();
  if (!is_last) return;

  asm volatile("fence.acquire.gpu;" ::: "memory");

  float m_split = -CUDART_INF_F;
  for (int s = tid; s < num_splits; s += blockDim.x)
    m_split = fmaxf(m_split, partial_m[hq * num_splits + s]);
  m_split = warp_reduce_max(m_split);
  if (lane == 0) smem_m[warp] = m_split;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < kWarps) ? smem_m[lane] : -CUDART_INF_F;
    v = warp_reduce_max(v);
    if (lane == 0) s_mg = v;
  }
  __syncthreads();
  float m_g = s_mg;

  float l_split_local = 0.0f;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    float ms = partial_m[hq * num_splits + s];
    float a = (ms == -CUDART_INF_F) ? 0.0f : exp2f(ms - m_g);
    l_split_local += a * partial_l[hq * num_splits + s];
  }
  l_split_local = warp_reduce_sum(l_split_local);
  if (lane == 0) smem_l[warp] = l_split_local;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < kWarps) ? smem_l[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) s_lg = v;
  }
  __syncthreads();
  float l_g = s_lg;
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
  if (tid == 0) counters[hq] = 0;
}

}  // namespace


void sdpa_cuda_sparse_v2_5_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor buffer_keys,
    torch::Tensor buffer_values,
    torch::Tensor live_idx,
    torch::Tensor live_count,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t l_buf,
    int64_t num_splits)
{
  TORCH_CHECK(q.is_cuda() && keys.is_cuda() && values.is_cuda()
              && live_idx.is_cuda() && live_count.is_cuda());
  TORCH_CHECK(buffer_keys.is_cuda() && buffer_values.is_cuda());
  TORCH_CHECK(q.scalar_type()             == torch::kFloat16);
  TORCH_CHECK(keys.scalar_type()          == torch::kFloat16);
  TORCH_CHECK(values.scalar_type()        == torch::kFloat16);
  TORCH_CHECK(buffer_keys.scalar_type()   == torch::kFloat16);
  TORCH_CHECK(buffer_values.scalar_type() == torch::kFloat16);
  TORCH_CHECK(live_idx.scalar_type()      == torch::kInt32);
  TORCH_CHECK(live_count.scalar_type()    == torch::kInt32);
  TORCH_CHECK(q.size(1) == kDim
              && keys.size(2) == kDim && values.size(2) == kDim
              && buffer_keys.size(2) == kDim && buffer_values.size(2) == kDim);

  int Hq      = (int)q.size(0);
  int Hkv     = (int)keys.size(0);
  int Npad    = (int)live_idx.size(1);
  int Bmax    = (int)buffer_keys.size(1);
  int splits  = (int)num_splits;

  dim3 grid(Hq, splits);
  auto stream = at::cuda::getCurrentCUDAStream();

  sdpa_cuda_sparse_v2_5_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(buffer_keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(buffer_values.data_ptr<at::Half>()),
      live_idx.data_ptr<int32_t>(),
      live_count.data_ptr<int32_t>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq, Hkv, Npad, Bmax, (int)l_buf, splits,
      (float)(scale * 1.4426950408889634));
}
