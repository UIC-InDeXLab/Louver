#include <torch/extension.h>

void sdpa_cuda_sparse_v2_6_launch(
    torch::Tensor q, torch::Tensor keys, torch::Tensor values,
    torch::Tensor buffer_keys, torch::Tensor buffer_values,
    torch::Tensor parent_alive_bitmap, torch::Tensor assigns_packed,
    torch::Tensor partial_m, torch::Tensor partial_l, torch::Tensor partial_o,
    torch::Tensor counters, torch::Tensor out,
    double scale, int64_t N_used, int64_t l_buf, int64_t num_splits);

torch::Tensor sdpa_cuda_sparse_v2_6_forward(
    torch::Tensor q, torch::Tensor keys, torch::Tensor values,
    torch::Tensor buffer_keys, torch::Tensor buffer_values,
    torch::Tensor parent_alive_bitmap, torch::Tensor assigns_packed,
    torch::Tensor partial_m, torch::Tensor partial_l, torch::Tensor partial_o,
    torch::Tensor counters, torch::Tensor out,
    double scale, int64_t N_used, int64_t l_buf, int64_t num_splits) {
  sdpa_cuda_sparse_v2_6_launch(
      q, keys, values, buffer_keys, buffer_values,
      parent_alive_bitmap, assigns_packed,
      partial_m, partial_l, partial_o, counters, out,
      scale, N_used, l_buf, num_splits);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdpa_cuda_sparse_v2_6_forward,
        "v2.6 — bitmap-driven sparse SDPA (no live_idx)");
}
