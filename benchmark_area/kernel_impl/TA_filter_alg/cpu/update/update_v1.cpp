// Incremental update kernel — clusters a fixed buffer of 256 keys per
// (s, h_kv) into 64 groups of 4 by sorted projection on a per-subspace axis.
// Mirrors the .cu update_v1.1 logic on CPU (parallel over s × h_kv).
//
// Layout written into:
//   centers[s, h_kv, K_used:K_used+64, :w_s]  — group means
//   assigns[s, h_kv, N_used:N_used+256]       — parent ids in [K_used, K_used+64)
//   keys/values arena tail filled by Python via .copy_().

#include "../_common.h"

namespace py = pybind11;
using namespace ta_cpu;

namespace {
constexpr int S_FIXED = 4;
constexpr int B = 256;
constexpr int K_BUF = 64;
constexpr int BF = 4;
}

void update_cluster(
    torch::Tensor buffer_keys,         // (H_kv, B, D) fp32
    torch::Tensor dim_offsets,         // (S,) int32
    torch::Tensor dim_widths,          // (S,) int32
    torch::Tensor centers,             // (S, H_kv, K_cap, max_w) fp32
    torch::Tensor assigns,             // (S, H_kv, N_pad) int32
    torch::Tensor parent_children,     // (S, H_kv, K_cap, bf) int32
    torch::Tensor parent_counts,       // (S, H_kv, K_cap) int32
    int64_t k_used,
    int64_t n_used) {
    TORCH_CHECK(buffer_keys.dtype() == torch::kFloat32);
    TORCH_CHECK(buffer_keys.size(1) == B, "buffer must be 256");

    const int64_t H_kv = buffer_keys.size(0);
    const int64_t D = buffer_keys.size(2);
    const int64_t K_cap = centers.size(2);
    const int64_t MW = centers.size(3);
    const int64_t N_pad = assigns.size(2);
    const int64_t S = centers.size(0);
    TORCH_CHECK(S == S_FIXED);

    const float* buf_ptr   = buffer_keys.data_ptr<float>();
    const int32_t* offs    = dim_offsets.data_ptr<int32_t>();
    const int32_t* widths  = dim_widths.data_ptr<int32_t>();
    float* centers_ptr     = centers.data_ptr<float>();
    int32_t* assigns_ptr   = assigns.data_ptr<int32_t>();
    int32_t* pc_ptr        = parent_children.data_ptr<int32_t>();
    int32_t* pcnt_ptr      = parent_counts.data_ptr<int32_t>();

    const int64_t buf_stride_h = B * D;
    const int64_t centers_stride_s = H_kv * K_cap * MW;
    const int64_t centers_stride_h = K_cap * MW;
    const int64_t assigns_stride_s = H_kv * N_pad;
    const int64_t assigns_stride_h = N_pad;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int s = 0; s < S_FIXED; ++s) {
        for (int64_t h = 0; h < H_kv; ++h) {
            int w = widths[s];
            int off = offs[s];
            const float* base = buf_ptr + h * buf_stride_h;

            // Project each key onto a single per-subspace axis = sum of dims.
            float proj[B];
            int idx[B];
            for (int i = 0; i < B; ++i) {
                const float* p = base + (int64_t)i * D + off;
                float sum = 0.f;
                for (int d = 0; d < w; ++d) sum += p[d];
                proj[i] = sum;
                idx[i] = i;
            }
            std::sort(idx, idx + B,
                      [&](int a, int b) { return proj[a] < proj[b]; });

            float* centers_block = centers_ptr + s * centers_stride_s
                                   + h * centers_stride_h
                                   + (int64_t)k_used * MW;
            int32_t* assigns_block = assigns_ptr + s * assigns_stride_s
                                     + h * assigns_stride_h
                                     + n_used;
            // parent_children: shape (S, H_kv, K_cap, BF)
            int32_t* pc_block = pc_ptr
                + s * H_kv * (int64_t)centers.size(2) * BF
                + h * (int64_t)centers.size(2) * BF
                + (int64_t)k_used * BF;
            int32_t* pcnt_block = pcnt_ptr
                + s * H_kv * (int64_t)centers.size(2)
                + h * (int64_t)centers.size(2)
                + k_used;

            for (int g = 0; g < K_BUF; ++g) {
                float* mean = centers_block + (int64_t)g * MW;
                std::memset(mean, 0, sizeof(float) * (size_t)MW);
                int32_t* pc_row = pc_block + (int64_t)g * BF;
                for (int j = 0; j < BF; ++j) {
                    int key_idx = idx[g * BF + j];
                    const float* p = base + (int64_t)key_idx * D + off;
                    for (int d = 0; d < w; ++d) mean[d] += p[d];
                    int32_t arena_key = (int32_t)(n_used + key_idx);
                    assigns_block[key_idx] = (int32_t)(k_used + g);
                    pc_row[j] = arena_key;
                }
                pcnt_block[g] = (int32_t)BF;
                float inv = 1.f / (float)BF;
                for (int d = 0; d < w; ++d) mean[d] *= inv;
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster", &update_cluster,
          py::arg("buffer_keys"), py::arg("dim_offsets"), py::arg("dim_widths"),
          py::arg("centers"), py::arg("assigns"),
          py::arg("parent_children"), py::arg("parent_counts"),
          py::arg("k_used"), py::arg("n_used"));
}
