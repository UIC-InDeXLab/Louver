#include <torch/extension.h>

void ta_filter_v7_10_fused_launch(
    torch::Tensor q, torch::Tensor centers,
    torch::Tensor dim_offsets, torch::Tensor dim_widths,
    torch::Tensor q_head_to_kv,
    torch::Tensor threshold,
    torch::Tensor assigns_packed,
    torch::Tensor top_scores, torch::Tensor top_indices,
    torch::Tensor depth, torch::Tensor live_idx, torch::Tensor live_count,
    int64_t k_clusters,
    int64_t k_stride,
    int64_t n_eff,
    int64_t tile_n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_pipeline", &ta_filter_v7_10_fused_launch,
          "v7.10/v7.11 single-launch fused pipeline (cooperative groups, CUDA)");
}
