#include <torch/extension.h>

void ta_filter_v10_0_launch(
    torch::Tensor q, torch::Tensor centers,
    torch::Tensor dim_offsets, torch::Tensor dim_widths,
    torch::Tensor q_head_to_kv,
    torch::Tensor threshold,
    torch::Tensor top_scores, torch::Tensor top_indices,
    torch::Tensor depth, torch::Tensor parent_alive_bitmap,
    int64_t k_clusters);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter", &ta_filter_v10_0_launch,
          "v10.0 — split filter (score | depth+bitmap), no cooperative grid_sync");
}
