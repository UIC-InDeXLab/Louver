// attend_v2 — h_q-parallel TA-filter + sparse SDPA with **parent-walk**.
//
// Same per-(h_q) phase 1-4 as v1; phase 5 replaces the O(N_eff) key scan with
// a direct walk over selected_parents × bf children. Dedupe via per-query
// visited[N_pad] uint8 (memset on each call). Wins big when depth is small.
//
// Specialised for S=4, bf=4, D=Dv=128.

#include "../_common.h"

namespace py = pybind11;
using namespace ta_cpu;

namespace {

constexpr int S_FIXED = 4;
constexpr int BF_FIXED = 4;
constexpr int L_MAX = 256;
constexpr float EXP_PRUNE = -16.f;

inline void update_online_softmax_d128(
    float& m, float& l, float* __restrict o,
    float score, const float* __restrict v) {
    if (score > m) {
        float diff = m - score;
        float a = (diff > EXP_PRUNE) ? std::exp(diff) : 0.f;
        l = l * a + 1.0f;
        if (a == 0.f) std::memset(o, 0, sizeof(float) * 128);
        else scale_d128(o, a);
        axpy_d128(o, v, 1.0f);
        m = score;
    } else {
        float diff = score - m;
        if (diff < EXP_PRUNE) return;
        float w = std::exp(diff);
        l += w;
        axpy_d128(o, v, w);
    }
}

}  // namespace

torch::Tensor attend_v2_forward(
    torch::Tensor q,
    torch::Tensor centers,
    torch::Tensor parent_children,
    torch::Tensor parent_counts,
    torch::Tensor keys,
    torch::Tensor values,
    torch::Tensor invalid_mask,
    torch::Tensor dim_offsets,
    torch::Tensor dim_widths,
    torch::Tensor threshold,
    torch::Tensor q_head_to_kv,
    torch::Tensor buffer_keys,
    torch::Tensor buffer_values,
    int64_t k_used,
    int64_t n_used,
    int64_t l_buf,
    double scale_d) {
    TORCH_CHECK(q.dtype() == torch::kFloat32);
    TORCH_CHECK(centers.dtype() == torch::kFloat32);
    TORCH_CHECK(parent_children.dtype() == torch::kInt32);
    TORCH_CHECK(parent_counts.dtype() == torch::kInt32);
    TORCH_CHECK(keys.dtype() == torch::kFloat32);
    TORCH_CHECK(values.dtype() == torch::kFloat32);

    const int64_t h_q = q.size(0);
    const int64_t D = q.size(1);
    const int64_t H_kv = centers.size(1);
    const int64_t K_cap = centers.size(2);
    const int64_t MW = centers.size(3);
    const int64_t N_pad = keys.size(1);
    const int64_t Dv = values.size(2);
    const int64_t B_max = buffer_keys.size(1);
    TORCH_CHECK(D == 128 && Dv == 128, "v2 specialised for D=Dv=128");

    const float scale = (float)scale_d;
    const auto q2kv = resolve_q_head_to_kv(q_head_to_kv, h_q, H_kv);

    auto out = torch::empty({h_q, Dv}, q.options());
    const int64_t K_eff = k_used;

    const float* q_ptr        = q.data_ptr<float>();
    const float* centers_ptr  = centers.data_ptr<float>();
    const int32_t* pc_ptr     = parent_children.data_ptr<int32_t>();
    const int32_t* pcnt_ptr   = parent_counts.data_ptr<int32_t>();
    const float* keys_ptr     = keys.data_ptr<float>();
    const float* values_ptr   = values.data_ptr<float>();
    const uint8_t* inv_ptr    = invalid_mask.data_ptr<uint8_t>();
    const int32_t* offs_ptr   = dim_offsets.data_ptr<int32_t>();
    const int32_t* w_ptr      = dim_widths.data_ptr<int32_t>();
    const float* th_ptr       = threshold.data_ptr<float>();
    const float* buf_k_ptr    = buffer_keys.data_ptr<float>();
    const float* buf_v_ptr    = buffer_values.data_ptr<float>();
    float* out_ptr            = out.data_ptr<float>();

    const int64_t centers_stride_s = H_kv * K_cap * MW;
    const int64_t centers_stride_h = K_cap * MW;
    const int64_t centers_stride_k = MW;
    const int64_t pc_stride_s = H_kv * K_cap * BF_FIXED;
    const int64_t pc_stride_h = K_cap * BF_FIXED;
    const int64_t pcnt_stride_s = H_kv * K_cap;
    const int64_t pcnt_stride_h = K_cap;
    const int64_t keys_stride_h = N_pad * D;
    const int64_t values_stride_h = N_pad * Dv;
    const int64_t buf_k_stride_h = B_max * D;
    const int64_t buf_v_stride_h = B_max * Dv;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<float>   scores(K_eff > 0 ? (size_t)K_eff : 1, 0.f);
        std::vector<int32_t> idxs(K_eff > 0 ? (size_t)K_eff : 1, 0);
        std::vector<float>   sorted_top((size_t)(L_MAX * S_FIXED), 0.f);
        std::vector<int32_t> top_idx((size_t)(L_MAX * S_FIXED), 0);
        std::vector<uint8_t> visited((size_t)N_pad, 0);
        float o[128];

#ifdef _OPENMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int64_t hq = 0; hq < h_q; ++hq) {
            const int64_t hkv = q2kv[(size_t)hq];
            const float th = th_ptr[hq];
            const float* qrow = q_ptr + hq * D;
            const int L_force = (int)std::min<int64_t>(L_MAX, K_eff);

            for (int s = 0; s < S_FIXED; ++s) {
                int w = w_ptr[s];
                int off = offs_ptr[s];
                const float* qs = qrow + off;
                const float* base = centers_ptr + s * centers_stride_s
                                    + hkv * centers_stride_h;
                if (w == 32) {
                    for (int64_t k = 0; k < K_eff; ++k) {
                        const float* c = base + k * centers_stride_k;
                        __m512 lo = _mm512_mul_ps(_mm512_loadu_ps(qs), _mm512_loadu_ps(c));
                        __m512 hi = _mm512_fmadd_ps(_mm512_loadu_ps(qs + 16),
                                                    _mm512_loadu_ps(c + 16), lo);
                        scores[(size_t)k] = reduce_zmm(hi);
                        idxs[(size_t)k] = (int32_t)k;
                    }
                } else {
                    for (int64_t k = 0; k < K_eff; ++k) {
                        scores[(size_t)k] = dot_generic(qs, base + k * centers_stride_k, w);
                        idxs[(size_t)k] = (int32_t)k;
                    }
                }
                int L = L_force;
                if (K_eff > L) {
                    std::partial_sort(idxs.begin(), idxs.begin() + L,
                                      idxs.begin() + (size_t)K_eff,
                                      [&](int32_t a, int32_t b) {
                                          return scores[(size_t)a] > scores[(size_t)b];
                                      });
                } else {
                    std::sort(idxs.begin(), idxs.begin() + (size_t)K_eff,
                              [&](int32_t a, int32_t b) {
                                  return scores[(size_t)a] > scores[(size_t)b];
                              });
                }
                float* row = sorted_top.data() + s * L_MAX;
                int32_t* tidx = top_idx.data() + s * L_MAX;
                for (int t = 0; t < L; ++t) {
                    int idx = idxs[(size_t)t];
                    row[t] = scores[(size_t)idx];
                    tidx[t] = idx;
                }
            }

            int depth = L_force;
            for (int t = 0; t < L_force; ++t) {
                float row_sum = sorted_top[0 * L_MAX + t]
                              + sorted_top[1 * L_MAX + t]
                              + sorted_top[2 * L_MAX + t]
                              + sorted_top[3 * L_MAX + t];
                if (row_sum < th) { depth = t; break; }
            }
            if (depth == 0) depth = 1;

            std::memset(o, 0, sizeof(float) * 128);
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;
            std::memset(visited.data(), 0, (size_t)N_pad);

            const float* keys_row = keys_ptr + hkv * keys_stride_h;
            const float* vals_row = values_ptr + hkv * values_stride_h;
            const uint8_t* inv_row = inv_ptr + hkv * N_pad;

            // Phase 5: parent-walk. For each (s, selected parent), iterate
            // bf children, dedupe via visited[].
            for (int s = 0; s < S_FIXED; ++s) {
                const int32_t* tidx = top_idx.data() + s * L_MAX;
                const int32_t* pc_base = pc_ptr + s * pc_stride_s + hkv * pc_stride_h;
                const int32_t* pcnt_base = pcnt_ptr + s * pcnt_stride_s + hkv * pcnt_stride_h;
                int L = std::min(depth, L_force);
                for (int t = 0; t < L; ++t) {
                    int p = tidx[t];
                    int cnt = pcnt_base[p];
                    const int32_t* children = pc_base + (int64_t)p * BF_FIXED;
                    for (int j = 0; j < cnt; ++j) {
                        int32_t k = children[j];
                        if (k < 0) continue;
                        if (visited[(size_t)k]) continue;
                        visited[(size_t)k] = 1;
                        if (inv_row[k]) continue;
                        const float* kr = keys_row + (int64_t)k * D;
                        __builtin_prefetch(kr + D);
                        float dot = dot_d128(qrow, kr);
                        float sc = dot * scale;
                        update_online_softmax_d128(m, l, o, sc, vals_row + (int64_t)k * Dv);
                    }
                }
            }

            // Phase 6: buffer tail.
            const float* bk = buf_k_ptr + hkv * buf_k_stride_h;
            const float* bv = buf_v_ptr + hkv * buf_v_stride_h;
            for (int64_t k = 0; k < l_buf; ++k) {
                const float* kr = bk + k * D;
                float dot = dot_d128(qrow, kr);
                float sc = dot * scale;
                update_online_softmax_d128(m, l, o, sc, bv + k * Dv);
            }

            float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            float* orow = out_ptr + hq * Dv;
            for (int i = 0; i < 128; ++i) orow[i] = o[i] * inv_l;
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v2_forward,
          py::arg("q"), py::arg("centers"),
          py::arg("parent_children"), py::arg("parent_counts"),
          py::arg("keys"), py::arg("values"), py::arg("invalid_mask"),
          py::arg("dim_offsets"), py::arg("dim_widths"),
          py::arg("threshold"), py::arg("q_head_to_kv"),
          py::arg("buffer_keys"), py::arg("buffer_values"),
          py::arg("k_used"), py::arg("n_used"), py::arg("l_buf"),
          py::arg("scale"));
}
