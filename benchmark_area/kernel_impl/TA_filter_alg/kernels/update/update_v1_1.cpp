#include <torch/extension.h>

void update_v1_1_cluster_launch(
    torch::Tensor buffer_keys,
    torch::Tensor dim_offsets,
    torch::Tensor dim_widths,
    torch::Tensor centers_arena,
    torch::Tensor assigns_arena,
    int64_t K_used,
    int64_t N_used);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &update_v1_1_cluster_launch,
          "update_v1.1: cluster 256 buffer keys -> centers + assigns (S=4, bf=4, K_BUF=64)");
}
