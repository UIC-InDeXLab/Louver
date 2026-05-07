#include <torch/extension.h>

void sdpa_cuda_sparse_v2_5_launch(
    torch::Tensor q, torch::Tensor keys, torch::Tensor values,
    torch::Tensor buffer_keys, torch::Tensor buffer_values,
    torch::Tensor live_idx, torch::Tensor live_count,
    torch::Tensor partial_m, torch::Tensor partial_l, torch::Tensor partial_o,
    torch::Tensor counters, torch::Tensor out,
    double scale, int64_t l_buf, int64_t num_splits);

torch::Tensor sdpa_cuda_sparse_v2_5_forward(
    torch::Tensor q, torch::Tensor keys, torch::Tensor values,
    torch::Tensor buffer_keys, torch::Tensor buffer_values,
    torch::Tensor live_idx, torch::Tensor live_count,
    torch::Tensor partial_m, torch::Tensor partial_l, torch::Tensor partial_o,
    torch::Tensor counters, torch::Tensor out,
    double scale, int64_t l_buf, int64_t num_splits) {
  sdpa_cuda_sparse_v2_5_launch(
      q, keys, values, buffer_keys, buffer_values, live_idx, live_count,
      partial_m, partial_l, partial_o, counters, out,
      scale, l_buf, num_splits);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sdpa_cuda_sparse_v2_5_forward,
        "Online softmax sparse decode SDPA fp16 v2.5 (buffer-aware)");
}
