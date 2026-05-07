#include <torch/extension.h>

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
    int64_t num_splits);

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
    int64_t num_splits);

torch::Tensor sdpa_cuda_atomic_forward(
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
  sdpa_cuda_atomic_launch(
      q,
      keys,
      values,
      mask,
      partial_m,
      partial_l,
      partial_o,
      scores,
      counters,
      out,
      scale,
      num_splits);
  return out;
}

torch::Tensor sdpa_cuda_sparse_v1_forward(
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
  sdpa_cuda_sparse_v1_launch(
      q,
      keys,
      values,
      mask,
      indices,
      counts,
      partial_m,
      partial_l,
      partial_o,
      scores,
      counters,
      out,
      scale,
      num_splits);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdpa_cuda_atomic_forward, "Masked decode SDPA fp16 CUDA");
  m.def("forward_sparse_v1", &sdpa_cuda_sparse_v1_forward, "Sparse masked decode SDPA fp16 CUDA v1.0");
}
