// attention v1.3 — anchor-only gate (mirrors GPU v5.14).
//
// Why this matters: profiling v1.2 shows the per-row gate (S=8 subspace dot
// products of width=16 per subspace) dominates wall-clock. Only ~1.4% of rows
// survive the gate, so the full D=128 dot is rarely executed — speeding up
// the survivor path doesn't help.
//
// v1.3 collapses the gate to a *single* subspace check (the anchor). The
// build kernel already arranges keys so that anchor is the contiguous
// subspace with the strongest discriminative power; the other S-1 subspaces
// were used at threshold-derivation time but are not re-checked online.
//
// Net effect: ~8× less gate work per row. Inner kernels stay AVX-512.
#include "_cpu_common.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define HIRA_HAS_AVX512 1
#else
#define HIRA_HAS_AVX512 0
#endif

namespace py = pybind11;
using namespace hira_cpu;

namespace {

constexpr int64_t PARENTS_PER_TILE = 32;

#if HIRA_HAS_AVX512
inline float reduce_zmm(__m512 v) { return _mm512_reduce_add_ps(v); }

inline float dot_d128_avx512(const float* __restrict a, const float* __restrict b) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (int i = 0; i < 128; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i +  0), _mm512_loadu_ps(b + i +  0), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    return reduce_zmm(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

inline float dot_w16_avx512(const float* __restrict a, const float* __restrict b) {
    return reduce_zmm(_mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b)));
}

inline float dot_generic_avx512(const float* __restrict a, const float* __restrict b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float r = reduce_zmm(acc);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

inline void axpy_d128_avx512(float* __restrict dst, const float* __restrict src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vw, _mm512_loadu_ps(dst + i +  0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vw, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vw, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vw, _mm512_loadu_ps(dst + i + 48)));
    }
}

inline void axpy_generic_avx512(float* __restrict dst, const float* __restrict src, float w, int n) {
    const __m512 vw = _mm512_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(src + i), vw, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; ++i) dst[i] += w * src[i];
}

inline void scale_axpy_d128_avx512(
    float* __restrict dst, const float* __restrict src, float a, float b) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va)));
    }
}

inline void scale_axpy_generic_avx512(
    float* __restrict dst, const float* __restrict src, float a, float b, int n) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(src + i), vb,
                                                  _mm512_mul_ps(_mm512_loadu_ps(dst + i), va)));
    }
    for (; i < n; ++i) dst[i] = a * dst[i] + b * src[i];
}

inline void scale_d128_avx512(float* __restrict dst, float a) {
    const __m512 va = _mm512_set1_ps(a);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va));
    }
}

inline void scale_generic_avx512(float* __restrict dst, float a, int n) {
    const __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), va));
    }
    for (; i < n; ++i) dst[i] *= a;
}
#endif  // HIRA_HAS_AVX512

inline float dot_dispatch(const float* a, const float* b, int n) {
#if HIRA_HAS_AVX512
    if (n == 128) return dot_d128_avx512(a, b);
    if (n == 16)  return dot_w16_avx512(a, b);
    return dot_generic_avx512(a, b, n);
#else
    float r = 0.0f;
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
#endif
}

inline void axpy_dispatch(float* dst, const float* src, float w, int n) {
#if HIRA_HAS_AVX512
    if (n == 128) axpy_d128_avx512(dst, src, w);
    else axpy_generic_avx512(dst, src, w, n);
#else
    for (int i = 0; i < n; ++i) dst[i] += w * src[i];
#endif
}

inline void scale_axpy_dispatch(float* dst, const float* src, float a, float b, int n) {
#if HIRA_HAS_AVX512
    if (n == 128) scale_axpy_d128_avx512(dst, src, a, b);
    else scale_axpy_generic_avx512(dst, src, a, b, n);
#else
    for (int i = 0; i < n; ++i) dst[i] = a * dst[i] + b * src[i];
#endif
}

inline void scale_dispatch(float* dst, float a, int n) {
#if HIRA_HAS_AVX512
    if (n == 128) scale_d128_avx512(dst, a);
    else scale_generic_avx512(dst, a, n);
#else
    for (int i = 0; i < n; ++i) dst[i] *= a;
#endif
}

inline void merge_online(
    float& m_dst, float& l_dst, float* o_dst,
    float m_src, float l_src, const float* o_src, int64_t d_v) {
    if (l_src == 0.0f) return;
    if (l_dst == 0.0f) {
        m_dst = m_src; l_dst = l_src;
        std::copy(o_src, o_src + d_v, o_dst);
        return;
    }
    const float new_m = std::max(m_dst, m_src);
    const float a = std::exp(m_dst - new_m);
    const float b = std::exp(m_src - new_m);
    l_dst = l_dst * a + l_src * b;
    scale_axpy_dispatch(o_dst, o_src, a, b, static_cast<int>(d_v));
    m_dst = new_m;
}

}  // namespace

torch::Tensor attend_v1_3(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    auto q = as_cpu_float(q_in, "q");
    auto th = as_cpu_float(th_in, "th_per_subspace");
    TORCH_CHECK(q.dim() == 2, "q must be (H_q, D)");
    TORCH_CHECK(th.dim() == 2, "th_per_subspace must be (S, H_q) or (2S, H_q)");

    auto keys_reord = as_cpu_float(state["keys_reord"].cast<torch::Tensor>(),
                                   "state['keys_reord']");
    auto invalid_mask = state["invalid_mask"].cast<torch::Tensor>().contiguous().to(torch::kBool);
    auto values_reord = state.contains("values_reord")
        ? as_cpu_float(state["values_reord"].cast<torch::Tensor>(), "state['values_reord']")
        : keys_reord;

    auto centers = list_float_tensors(state["centers"], "state['centers']");
    auto radii = list_float_tensors(state["radii"], "state['radii']");
    auto assigns = list_int_tensors(state["assigns_reord"], "state['assigns_reord']");
    auto slices = slices_from_state(state);

    const int64_t s_count = static_cast<int64_t>(centers.size());
    TORCH_CHECK(th.size(0) >= s_count && th.size(1) == q.size(0),
                "threshold shape mismatch");

    // Anchor: build_v1_0 reorders subspaces so the anchor is index 0.
    const int64_t anchor_s = 0;

    torch::Tensor buffer_keys = object_to_tensor_or_empty(buffer_keys_obj);
    torch::Tensor buffer_values = object_to_tensor_or_empty(buffer_values_obj);
    const bool has_buffer = buffer_keys.defined() && buffer_keys.numel() > 0;
    if (has_buffer) {
        buffer_keys = as_cpu_float(buffer_keys, "buffer_keys");
        if (buffer_values.defined() && buffer_values.numel() > 0)
            buffer_values = as_cpu_float(buffer_values, "buffer_values");
        else
            buffer_values = buffer_keys;
    }

    const int64_t h_q = q.size(0);
    const int64_t d = q.size(1);
    const int64_t h_kv = keys_reord.size(0);
    const int64_t n_pad = keys_reord.size(1);
    const int64_t d_v = values_reord.size(2);
    const int64_t bf = state["bf"].cast<int64_t>();
    const int64_t k_used = state.contains("K_used")
        ? state["K_used"].cast<int64_t>()
        : state["K"].cast<int64_t>();

    auto q2kv = resolve_q_head_to_kv(q_head_to_kv_obj, h_q, h_kv);

    auto out = torch::zeros({h_q, d_v}, q.options().dtype(torch::kFloat32));
    const float* qp = q.data_ptr<float>();
    const float* thp = th.data_ptr<float>();
    const float* krp = keys_reord.data_ptr<float>();
    const float* vrp = values_reord.data_ptr<float>();
    const bool* invp = invalid_mask.data_ptr<bool>();
    float* outp = out.data_ptr<float>();
    const float* bkp = has_buffer ? buffer_keys.data_ptr<float>() : nullptr;
    const float* bvp = has_buffer ? buffer_values.data_ptr<float>() : nullptr;
    const int64_t n_buf = has_buffer ? buffer_keys.size(1) : 0;

    const float* center_anchor = centers[anchor_s].data_ptr<float>();
    const float* radius_anchor = radii[anchor_s].data_ptr<float>();
    const int32_t* assign_anchor = assigns[anchor_s].data_ptr<int32_t>();
    const int64_t anchor_start = slices[anchor_s].first;
    const int64_t anchor_width = slices[anchor_s].second - slices[anchor_s].first;

    const int64_t n_tiles = (k_used + PARENTS_PER_TILE - 1) / PARENTS_PER_TILE;
    const int64_t total_tiles = h_q * n_tiles;

    std::vector<float> tile_m(static_cast<size_t>(total_tiles),
                              -std::numeric_limits<float>::infinity());
    std::vector<float> tile_l(static_cast<size_t>(total_tiles), 0.0f);
    std::vector<float> tile_o(static_cast<size_t>(total_tiles * d_v), 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles; ++task) {
        const int64_t qh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const int64_t kvh = q2kv[qh];

        // Only the anchor q-norm is needed.
        const float qn_anchor = std::sqrt(std::max(
            dot_dispatch(qp + qh * d + anchor_start,
                         qp + qh * d + anchor_start,
                         static_cast<int>(anchor_width)), 0.0f));
        const float th_anchor = thp[anchor_s * h_q + qh];

        const int64_t parent_lo = tile * PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + PARENTS_PER_TILE, k_used);

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        float* o_run = &tile_o[task * d_v];

        // Walk parent-by-parent so the (parent, gate) cost is paid once for
        // its bf children — bf children share a parent in build's reorder.
        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            // Anchor bound per parent (anchor assignment is identical for
            // children sharing a parent — that's the build invariant).
            const float* cp = center_anchor + (kvh * k_used + parent) * anchor_width;
            const float bound_dot = dot_dispatch(
                qp + qh * d + anchor_start, cp, static_cast<int>(anchor_width));
            const float bound = bound_dot + qn_anchor * radius_anchor[kvh * k_used + parent];
            if (bound < th_anchor) continue;  // entire bf-block pruned

            for (int64_t child = 0; child < bf; ++child) {
                const int64_t j = parent * bf + child;
                if (j >= n_pad) break;
                if (invp[kvh * n_pad + j]) continue;

                const float* key = krp + (kvh * n_pad + j) * d;
                const float score = dot_dispatch(qp + qh * d, key, static_cast<int>(d))
                                    * static_cast<float>(scale);

                const float* val = vrp + (kvh * n_pad + j) * d_v;
                if (score > m_run) {
                    if (l_run == 0.0f) {
                        m_run = score; l_run = 1.0f;
                        std::copy(val, val + d_v, o_run);
                    } else {
                        const float a = std::exp(m_run - score);
                        l_run = l_run * a + 1.0f;
                        scale_dispatch(o_run, a, static_cast<int>(d_v));
                        axpy_dispatch(o_run, val, 1.0f, static_cast<int>(d_v));
                        m_run = score;
                    }
                } else {
                    const float w = std::exp(score - m_run);
                    l_run += w;
                    axpy_dispatch(o_run, val, w, static_cast<int>(d_v));
                }
            }
        }

        tile_m[task] = m_run;
        tile_l[task] = l_run;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < h_q; ++qh) {
        const int64_t kvh = q2kv[qh];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        std::vector<float> o_acc(static_cast<size_t>(d_v), 0.0f);

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t task = qh * n_tiles + tile;
            merge_online(m_acc, l_acc, o_acc.data(),
                         tile_m[task], tile_l[task], &tile_o[task * d_v], d_v);
        }

        for (int64_t j = 0; j < n_buf; ++j) {
            const float* key = bkp + (kvh * n_buf + j) * d;
            const float score = dot_dispatch(qp + qh * d, key, static_cast<int>(d))
                                * static_cast<float>(scale);
            const float* val = bvp + (kvh * n_buf + j) * d_v;
            if (score > m_acc) {
                if (l_acc == 0.0f) {
                    m_acc = score; l_acc = 1.0f;
                    std::copy(val, val + d_v, o_acc.data());
                } else {
                    const float a = std::exp(m_acc - score);
                    l_acc = l_acc * a + 1.0f;
                    scale_dispatch(o_acc.data(), a, static_cast<int>(d_v));
                    axpy_dispatch(o_acc.data(), val, 1.0f, static_cast<int>(d_v));
                    m_acc = score;
                }
            } else {
                const float w = std::exp(score - m_acc);
                l_acc += w;
                axpy_dispatch(o_acc.data(), val, w, static_cast<int>(d_v));
            }
        }

        if (l_acc > 0.0f) {
            const float inv = 1.0f / l_acc;
            for (int64_t x = 0; x < d_v; ++x) outp[qh * d_v + x] = o_acc[x] * inv;
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v1_3,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
