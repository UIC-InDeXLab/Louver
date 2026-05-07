#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = 8;

__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float smem[kThreads];
  int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  return smem[0];
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float smem[kThreads];
  int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();
  for (int stride = kThreads >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  return smem[0];
}

__device__ __forceinline__ float dot128(
    const half* __restrict__ q,
    const half* __restrict__ k) {
  float acc = 0.0f;
  const half2* q2 = reinterpret_cast<const half2*>(q);
  const half2* k2 = reinterpret_cast<const half2*>(k);
#pragma unroll
  for (int d = 0; d < 64; ++d) {
    half2 prod = __hmul2(q2[d], k2[d]);
    float2 f = __half22float2(prod);
    acc += f.x + f.y;
  }
  return acc;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void sdpa_cuda_atomic_kernel(
    const half* __restrict__ q,        // Hq, 128
    const half* __restrict__ keys,     // Hkv, N, 128
    const half* __restrict__ values,   // Hkv, N, 128
    const int8_t* __restrict__ mask,   // Hq, N
    float* __restrict__ partial_m,     // Hq, S
    float* __restrict__ partial_l,     // Hq, S
    float* __restrict__ partial_o,     // Hq, S, 128
    float* __restrict__ scores,        // Hq, S, cols_per_split
    int* __restrict__ counters,        // Hq
    half* __restrict__ out,            // Hq, 128
    int Hq,
  int Hkv,
  int N,
  int num_splits,
  int cols_per_split,
  float scale_log2e) {
  int hq = blockIdx.x;
  int split = blockIdx.y;
  int tid = threadIdx.x;
  int groups = Hq / Hkv;
  int kvh = hq / groups;
  int n_start = split * cols_per_split;
  int n_end = min(n_start + cols_per_split, N);
  int span = max(0, n_end - n_start);

  const half* qh = q + hq * 128;
  const half* kh_base = keys + kvh * N * 128;
  const half* vh_base = values + kvh * N * 128;
  const int8_t* mask_h = mask + hq * N;
  float* score_base = scores + (hq * num_splits + split) * cols_per_split;

  float local_m = -CUDART_INF_F;
  int lane = tid & 31;
  int warp = tid >> 5;
  for (int i = warp; i < span; i += kWarps) {
    int n = n_start + i;
    float part = 0.0f;
    if (mask_h[n] != 0) {
      const half* kh = kh_base + n * 128;
#pragma unroll
      for (int d = lane; d < 128; d += 32) {
        part += __half2float(qh[d]) * __half2float(kh[d]);
      }
    } else {
      part = -CUDART_INF_F;
    }
    float s = warp_reduce_sum(part);
    if (lane == 0) {
      if (mask_h[n] == 0) {
        s = -CUDART_INF_F;
      } else {
        s *= scale_log2e;
      }
      score_base[i] = s;
      local_m = fmaxf(local_m, s);
    }
  }
  float m = block_reduce_max(local_m);

  float local_l = 0.0f;
  if (m != -CUDART_INF_F) {
    for (int i = tid; i < span; i += blockDim.x) {
      float p = exp2f(score_base[i] - m);
      score_base[i] = p;
      local_l += p;
    }
  }
  float l = block_reduce_sum(local_l);

  float* po = partial_o + (hq * num_splits + split) * 128;
  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < span; ++i) {
      int n = n_start + i;
      acc += score_base[i] * __half2float(vh_base[n * 128 + dv]);
    }
    po[dv] = acc;
  }
  if (tid == 0) {
    partial_m[hq * num_splits + split] = m;
    partial_l[hq * num_splits + split] = l;
  }

  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (tid == 0) {
    int old = atomicAdd(counters + hq, 1);
    is_last = (old == num_splits - 1);
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  float m_global = -CUDART_INF_F;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    m_global = fmaxf(m_global, partial_m[hq * num_splits + s]);
  }
  m_global = block_reduce_max(m_global);

  float l_global_local = 0.0f;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
    l_global_local += alpha * partial_l[hq * num_splits + s];
  }
  float l_global = block_reduce_sum(l_global_local);
  float inv_l = l_global > 0.0f ? 1.0f / l_global : 0.0f;

  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
      acc += alpha * partial_o[(hq * num_splits + s) * 128 + dv];
    }
    out[hq * 128 + dv] = __float2half(acc * inv_l);
  }
}

__global__ void build_sparse_indices_kernel(
    const int8_t* __restrict__ mask,
    int* __restrict__ indices,
    int* __restrict__ counts,
    int Hq,
    int N) {
  int hq = blockIdx.x;
  int tid = blockIdx.y * blockDim.x + threadIdx.x;
  if (tid >= N) {
    return;
  }
  int m = mask[hq * N + tid] != 0;
  if (m) {
    int slot = atomicAdd(counts + hq, 1);
    indices[hq * N + slot] = tid;
  }
}

__global__ void sdpa_cuda_sparse_v1_kernel(
    const half* __restrict__ q,        // Hq, 128
    const half* __restrict__ keys,     // Hkv, N, 128
    const half* __restrict__ values,   // Hkv, N, 128
    const int* __restrict__ indices,   // Hq, N_live_max
    const int* __restrict__ counts,    // Hq
    float* __restrict__ partial_m,     // Hq, S
    float* __restrict__ partial_l,     // Hq, S
    float* __restrict__ partial_o,     // Hq, S, 128
    float* __restrict__ scores,        // Hq, S, cols_per_split
    int* __restrict__ counters,        // Hq
    half* __restrict__ out,            // Hq, 128
    int Hq,
    int Hkv,
    int N,
    int num_splits,
    int cols_per_split,
    float scale_log2e) {
  int hq = blockIdx.x;
  int split = blockIdx.y;
  int tid = threadIdx.x;
  int groups = Hq / Hkv;
  int kvh = hq / groups;
  int live_n = counts[hq];
  int live_cols = (live_n + num_splits - 1) / num_splits;
  int live_start = split * live_cols;
  int live_end = min(live_start + live_cols, live_n);
  int span = max(0, live_end - live_start);

  const half* qh = q + hq * 128;
  const half* kh_base = keys + kvh * N * 128;
  const half* vh_base = values + kvh * N * 128;
  const int* idx_h = indices + hq * N;
  float* score_base = scores + (hq * num_splits + split) * cols_per_split;

  float local_m = -CUDART_INF_F;
  int lane = tid & 31;
  int warp = tid >> 5;
  for (int i = warp; i < span; i += kWarps) {
    int n = idx_h[live_start + i];
    const half* kh = kh_base + n * 128;
    float part = 0.0f;
#pragma unroll
    for (int d = lane; d < 128; d += 32) {
      part += __half2float(qh[d]) * __half2float(kh[d]);
    }
    float s = warp_reduce_sum(part);
    if (lane == 0) {
      s *= scale_log2e;
      score_base[i] = s;
      local_m = fmaxf(local_m, s);
    }
  }
  float m = block_reduce_max(local_m);

  float local_l = 0.0f;
  if (m != -CUDART_INF_F) {
    for (int i = tid; i < span; i += blockDim.x) {
      float p = exp2f(score_base[i] - m);
      score_base[i] = p;
      local_l += p;
    }
  }
  float l = block_reduce_sum(local_l);

  float* po = partial_o + (hq * num_splits + split) * 128;
  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int i = 0; i < span; ++i) {
      int n = idx_h[live_start + i];
      acc += score_base[i] * __half2float(vh_base[n * 128 + dv]);
    }
    po[dv] = acc;
  }
  if (tid == 0) {
    partial_m[hq * num_splits + split] = m;
    partial_l[hq * num_splits + split] = l;
  }

  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (tid == 0) {
    int old = atomicAdd(counters + hq, 1);
    is_last = (old == num_splits - 1);
  }
  __syncthreads();
  if (!is_last) {
    return;
  }

  float m_global = -CUDART_INF_F;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    m_global = fmaxf(m_global, partial_m[hq * num_splits + s]);
  }
  m_global = block_reduce_max(m_global);

  float l_global_local = 0.0f;
  for (int s = tid; s < num_splits; s += blockDim.x) {
    float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
    l_global_local += alpha * partial_l[hq * num_splits + s];
  }
  float l_global = block_reduce_sum(l_global_local);
  float inv_l = l_global > 0.0f ? 1.0f / l_global : 0.0f;

  for (int dv = tid; dv < 128; dv += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      float alpha = exp2f(partial_m[hq * num_splits + s] - m_global);
      acc += alpha * partial_o[(hq * num_splits + s) * 128 + dv];
    }
    out[hq * 128 + dv] = __float2half(acc * inv_l);
  }
}

} // namespace

void sdpa_cuda_atomic_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor scores,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && keys.is_cuda() && values.is_cuda() && mask.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(keys.scalar_type() == torch::kFloat16, "keys must be fp16");
  TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be fp16");
  TORCH_CHECK(mask.scalar_type() == torch::kInt8, "mask must be int8");
  TORCH_CHECK(q.size(1) == 128 && keys.size(2) == 128 && values.size(2) == 128, "D/Dv must be 128");

  int Hq = static_cast<int>(q.size(0));
  int Hkv = static_cast<int>(keys.size(0));
  int N = static_cast<int>(keys.size(1));
  int splits = static_cast<int>(num_splits);
  int cols = (N + splits - 1) / splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(counters.data_ptr<int>(), 0, Hq * sizeof(int), stream);

  dim3 grid(Hq, splits);
  sdpa_cuda_atomic_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      reinterpret_cast<const int8_t*>(mask.data_ptr<int8_t>()),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      scores.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq,
      Hkv,
      N,
      splits,
      cols,
      static_cast<float>(scale * 1.4426950408889634));
}

void sdpa_cuda_sparse_v1_launch(
    torch::Tensor q,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor mask,
    torch::Tensor indices,
    torch::Tensor counts,
    torch::Tensor partial_m,
    torch::Tensor partial_l,
    torch::Tensor partial_o,
    torch::Tensor scores,
    torch::Tensor counters,
    torch::Tensor out,
    double scale,
    int64_t num_splits) {
  TORCH_CHECK(q.is_cuda() && keys.is_cuda() && values.is_cuda() && mask.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16, "q must be fp16");
  TORCH_CHECK(keys.scalar_type() == torch::kFloat16, "keys must be fp16");
  TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be fp16");
  TORCH_CHECK(mask.scalar_type() == torch::kInt8, "mask must be int8");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(counts.scalar_type() == torch::kInt32, "counts must be int32");
  TORCH_CHECK(q.size(1) == 128 && keys.size(2) == 128 && values.size(2) == 128, "D/Dv must be 128");

  int Hq = static_cast<int>(q.size(0));
  int Hkv = static_cast<int>(keys.size(0));
  int N = static_cast<int>(keys.size(1));
  int splits = static_cast<int>(num_splits);
  int cols = (N + splits - 1) / splits;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(counts.data_ptr<int>(), 0, Hq * sizeof(int), stream);
  cudaMemsetAsync(counters.data_ptr<int>(), 0, Hq * sizeof(int), stream);

  dim3 build_grid(Hq, (N + kThreads - 1) / kThreads);
  build_sparse_indices_kernel<<<build_grid, kThreads, 0, stream>>>(
      reinterpret_cast<const int8_t*>(mask.data_ptr<int8_t>()),
      indices.data_ptr<int>(),
      counts.data_ptr<int>(),
      Hq,
      N);

  dim3 grid(Hq, splits);
  sdpa_cuda_sparse_v1_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(keys.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(values.data_ptr<at::Half>()),
      indices.data_ptr<int>(),
      counts.data_ptr<int>(),
      partial_m.data_ptr<float>(),
      partial_l.data_ptr<float>(),
      partial_o.data_ptr<float>(),
      scores.data_ptr<float>(),
      counters.data_ptr<int>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      Hq,
      Hkv,
      N,
      splits,
      cols,
      static_cast<float>(scale * 1.4426950408889634));
}
