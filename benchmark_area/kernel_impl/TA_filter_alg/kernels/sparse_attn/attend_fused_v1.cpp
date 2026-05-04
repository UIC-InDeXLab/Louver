#include <torch/extension.h>

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
    int64_t N_used, int64_t l_buf, int64_t num_splits);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attend_fused_v1_launch,
          "fused filter+sparse_attn coop kernel (single launch)");
}
