// Shared implementation for attention v2.x exact full-AND CPU kernels.
//
// v2 restores the documented algorithmic semantics:
//   1. compute per-subspace parent pass/fail tables for the query,
//   2. scan the reordered children and keep only rows whose parent passes in
//      every subspace,
//   3. scan the decode buffer unconditionally,
//   4. fuse scoring, online softmax, and value accumulation.
#pragma once

#include "_cpu_common.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define HIRA_V2_HAS_AVX512 1
#else
#define HIRA_V2_HAS_AVX512 0
#endif

#include <memory>

namespace py = pybind11;
using namespace hira_cpu;

namespace {

constexpr int64_t V2_PARENTS_PER_TILE = 32;
constexpr int64_t V2_D_MAX = 512;
constexpr int64_t V2_MAX_GROUPS = 16;

#if HIRA_V2_HAS_AVX512
inline float v2_reduce_zmm(__m512 v) { return _mm512_reduce_add_ps(v); }

inline float v2_dot_d128_avx512(const float* __restrict a, const float* __restrict b) {
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
    return v2_reduce_zmm(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

inline float v2_dot_w16_avx512(const float* __restrict a, const float* __restrict b) {
    return v2_reduce_zmm(_mm512_mul_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b)));
}

inline float v2_dot_generic_avx512(const float* __restrict a, const float* __restrict b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float r = v2_reduce_zmm(acc);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

inline void v2_axpy_d128_avx512(float* __restrict dst, const float* __restrict src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vw, _mm512_loadu_ps(dst + i +  0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vw, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vw, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vw, _mm512_loadu_ps(dst + i + 48)));
    }
}

inline void v2_axpy_generic_avx512(float* __restrict dst, const float* __restrict src, float w, int n) {
    const __m512 vw = _mm512_set1_ps(w);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_loadu_ps(src + i), vw, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; ++i) dst[i] += w * src[i];
}

inline void v2_scale_d128_avx512(float* __restrict dst, float a) {
    const __m512 va = _mm512_set1_ps(a);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va));
    }
}

inline void v2_scale_generic_avx512(float* __restrict dst, float a, int n) {
    const __m512 va = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), va));
    }
    for (; i < n; ++i) dst[i] *= a;
}

inline void v2_scale_axpy_d128_avx512(
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

inline void v2_scale_axpy_generic_avx512(
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
#endif

inline float v2_dot_dispatch(const float* a, const float* b, int n) {
#if HIRA_V2_HAS_AVX512
    if (n == 128) return v2_dot_d128_avx512(a, b);
    if (n == 16) return v2_dot_w16_avx512(a, b);
    return v2_dot_generic_avx512(a, b, n);
#else
    float r = 0.0f;
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
#endif
}

inline void v2_axpy_dispatch(float* dst, const float* src, float w, int n) {
#if HIRA_V2_HAS_AVX512
    if (n == 128) v2_axpy_d128_avx512(dst, src, w);
    else v2_axpy_generic_avx512(dst, src, w, n);
#else
    for (int i = 0; i < n; ++i) dst[i] += w * src[i];
#endif
}

inline void v2_scale_dispatch(float* dst, float a, int n) {
#if HIRA_V2_HAS_AVX512
    if (n == 128) v2_scale_d128_avx512(dst, a);
    else v2_scale_generic_avx512(dst, a, n);
#else
    for (int i = 0; i < n; ++i) dst[i] *= a;
#endif
}

inline void v2_scale_axpy_dispatch(float* dst, const float* src, float a, float b, int n) {
#if HIRA_V2_HAS_AVX512
    if (n == 128) v2_scale_axpy_d128_avx512(dst, src, a, b);
    else v2_scale_axpy_generic_avx512(dst, src, a, b, n);
#else
    for (int i = 0; i < n; ++i) dst[i] = a * dst[i] + b * src[i];
#endif
}

inline void v2_online_update(
    float score, const float* __restrict val, float& m_run, float& l_run,
    float* __restrict o_run, int64_t d_v) {
    if (score > m_run) {
        if (l_run == 0.0f) {
            m_run = score;
            l_run = 1.0f;
            std::copy(val, val + d_v, o_run);
        } else {
            const float a = std::exp(m_run - score);
            l_run = l_run * a + 1.0f;
            v2_scale_dispatch(o_run, a, static_cast<int>(d_v));
            v2_axpy_dispatch(o_run, val, 1.0f, static_cast<int>(d_v));
            m_run = score;
        }
    } else {
        const float w = std::exp(score - m_run);
        l_run += w;
        v2_axpy_dispatch(o_run, val, w, static_cast<int>(d_v));
    }
}

inline void v2_merge_online(
    float& m_dst, float& l_dst, float* __restrict o_dst,
    float m_src, float l_src, const float* __restrict o_src, int64_t d_v) {
    if (l_src == 0.0f) return;
    if (l_dst == 0.0f) {
        m_dst = m_src;
        l_dst = l_src;
        std::copy(o_src, o_src + d_v, o_dst);
        return;
    }
    const float new_m = std::max(m_dst, m_src);
    const float a = std::exp(m_dst - new_m);
    const float b = std::exp(m_src - new_m);
    l_dst = l_dst * a + l_src * b;
    v2_scale_axpy_dispatch(o_dst, o_src, a, b, static_cast<int>(d_v));
    m_dst = new_m;
}

struct V2Prepared {
    torch::Tensor q;
    torch::Tensor th;
    torch::Tensor keys_reord;
    torch::Tensor values_reord;
    torch::Tensor invalid_mask;
    torch::Tensor buffer_keys;
    torch::Tensor buffer_values;
    std::vector<torch::Tensor> centers;
    std::vector<torch::Tensor> radii;
    std::vector<torch::Tensor> assigns;
    std::vector<std::pair<int64_t, int64_t>> slices;
    std::vector<int64_t> q2kv;
    int64_t h_q = 0;
    int64_t h_kv = 0;
    int64_t n_pad = 0;
    int64_t d = 0;
    int64_t d_v = 0;
    int64_t bf = 0;
    int64_t k_total = 0;
    int64_t k_scan = 0;
    int64_t n_scan = 0;
    int64_t n_buf = 0;
    int64_t s_count = 0;
    bool has_buffer = false;
};

inline V2Prepared v2_prepare(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj) {
    V2Prepared p;
    p.q = as_cpu_float(q_in, "q");
    p.th = as_cpu_float(th_in, "th_per_subspace");
    TORCH_CHECK(p.q.dim() == 2, "q must be (H_q, D)");
    TORCH_CHECK(p.th.dim() == 2, "th_per_subspace must be (S, H_q) or (2S, H_q)");

    p.keys_reord = as_cpu_float(state["keys_reord"].cast<torch::Tensor>(),
                                "state['keys_reord']");
    p.invalid_mask = state["invalid_mask"].cast<torch::Tensor>().contiguous().to(torch::kBool);
    p.values_reord = state.contains("values_reord")
        ? as_cpu_float(state["values_reord"].cast<torch::Tensor>(), "state['values_reord']")
        : p.keys_reord;

    p.centers = list_float_tensors(state["centers"], "state['centers']");
    p.radii = list_float_tensors(state["radii"], "state['radii']");
    p.assigns = list_int_tensors(state["assigns_reord"], "state['assigns_reord']");
    p.slices = slices_from_state(state);
    p.s_count = static_cast<int64_t>(p.centers.size());
    TORCH_CHECK(p.assigns.size() == p.centers.size(), "assigns/centers size mismatch");
    TORCH_CHECK(p.radii.size() == p.centers.size(), "radii/centers size mismatch");
    TORCH_CHECK(p.th.size(0) >= p.s_count && p.th.size(1) == p.q.size(0),
                "threshold shape mismatch");

    p.buffer_keys = object_to_tensor_or_empty(buffer_keys_obj);
    p.buffer_values = object_to_tensor_or_empty(buffer_values_obj);
    p.has_buffer = p.buffer_keys.defined() && p.buffer_keys.numel() > 0;
    if (p.has_buffer) {
        p.buffer_keys = as_cpu_float(p.buffer_keys, "buffer_keys");
        if (p.buffer_values.defined() && p.buffer_values.numel() > 0) {
            p.buffer_values = as_cpu_float(p.buffer_values, "buffer_values");
        } else {
            p.buffer_values = p.buffer_keys;
        }
    }

    p.h_q = p.q.size(0);
    p.d = p.q.size(1);
    p.h_kv = p.keys_reord.size(0);
    p.n_pad = p.keys_reord.size(1);
    p.d_v = p.values_reord.size(2);
    p.bf = state["bf"].cast<int64_t>();
    p.k_total = p.centers[0].size(1);
    const int64_t k_visible = state.contains("K_used")
        ? state["K_used"].cast<int64_t>()
        : state["K"].cast<int64_t>();
    p.k_scan = std::min<int64_t>(k_visible, p.k_total);
    p.n_scan = std::min<int64_t>(p.n_pad, p.k_scan * p.bf);
    p.n_buf = p.has_buffer ? p.buffer_keys.size(1) : 0;
    TORCH_CHECK(p.d <= V2_D_MAX && p.d_v <= V2_D_MAX, "v2 D limit ", V2_D_MAX);
    TORCH_CHECK(p.invalid_mask.size(0) == p.h_kv && p.invalid_mask.size(1) == p.n_pad,
                "invalid_mask shape mismatch");
    for (int64_t s = 0; s < p.s_count; ++s) {
        TORCH_CHECK(p.centers[s].size(0) == p.h_kv && p.centers[s].size(1) == p.k_total,
                    "center shape mismatch at subspace ", s);
        TORCH_CHECK(p.radii[s].size(0) == p.h_kv && p.radii[s].size(1) == p.k_total,
                    "radius shape mismatch at subspace ", s);
        TORCH_CHECK(p.assigns[s].size(0) == p.h_kv && p.assigns[s].size(1) == p.n_pad,
                    "assign shape mismatch at subspace ", s);
    }
    p.q2kv = resolve_q_head_to_kv(q_head_to_kv_obj, p.h_q, p.h_kv);
    return p;
}

inline bool v2_child_passes_local(
    int64_t kvh, int64_t j, int64_t n_pad, int64_t bf, int64_t k_total,
    int64_t s_count, const uint8_t* __restrict pass,
    const std::vector<const int32_t*>& assign_ptrs) {
    for (int64_t s = 0; s < s_count; ++s) {
        const int64_t parent = (s == 0)
            ? (j / bf)
            : static_cast<int64_t>(assign_ptrs[static_cast<size_t>(s)][kvh * n_pad + j]);
        if (parent < 0 || parent >= k_total) return false;
        if (pass[s * k_total + parent] == 0) return false;
    }
    return true;
}

inline bool v2_child_passes_global(
    int64_t qh, int64_t kvh, int64_t j, int64_t n_pad, int64_t bf,
    int64_t k_total, int64_t s_count, const uint8_t* __restrict pass,
    const std::vector<const int32_t*>& assign_ptrs) {
    const int64_t base = qh * s_count * k_total;
    return v2_child_passes_local(
        kvh, j, n_pad, bf, k_total, s_count, pass + base, assign_ptrs);
}

torch::Tensor attend_v2_exact_serial(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    V2Prepared p = v2_prepare(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj, q_head_to_kv_obj);

    auto out = torch::empty({p.h_q, p.d_v}, p.q.options().dtype(torch::kFloat32));
    const float* qp = p.q.data_ptr<float>();
    const float* thp = p.th.data_ptr<float>();
    const float* krp = p.keys_reord.data_ptr<float>();
    const float* vrp = p.values_reord.data_ptr<float>();
    const bool* invp = p.invalid_mask.data_ptr<bool>();
    float* outp = out.data_ptr<float>();
    const float* bkp = p.has_buffer ? p.buffer_keys.data_ptr<float>() : nullptr;
    const float* bvp = p.has_buffer ? p.buffer_values.data_ptr<float>() : nullptr;

    std::vector<const float*> center_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const float*> radius_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const int32_t*> assign_ptrs(static_cast<size_t>(p.s_count));
    for (int64_t s = 0; s < p.s_count; ++s) {
        center_ptrs[static_cast<size_t>(s)] = p.centers[s].data_ptr<float>();
        radius_ptrs[static_cast<size_t>(s)] = p.radii[s].data_ptr<float>();
        assign_ptrs[static_cast<size_t>(s)] = p.assigns[s].data_ptr<int32_t>();
    }

    #pragma omp parallel for schedule(dynamic)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        TORCH_CHECK(kvh >= 0 && kvh < p.h_kv, "q_head_to_kv entry out of range");

        std::vector<float> q_norms(static_cast<size_t>(p.s_count), 0.0f);
        std::vector<uint8_t> pass(static_cast<size_t>(p.s_count * p.k_total), 0);
        for (int64_t s = 0; s < p.s_count; ++s) {
            const auto [start, end] = p.slices[static_cast<size_t>(s)];
            const int64_t width = end - start;
            q_norms[static_cast<size_t>(s)] = std::sqrt(std::max(
                v2_dot_dispatch(qp + qh * p.d + start, qp + qh * p.d + start,
                                static_cast<int>(width)),
                0.0f));
            const float qn = q_norms[static_cast<size_t>(s)];
            const float th = thp[s * p.h_q + qh];
            const float* cp_base = center_ptrs[static_cast<size_t>(s)] + kvh * p.k_total * width;
            const float* rp = radius_ptrs[static_cast<size_t>(s)] + kvh * p.k_total;
            uint8_t* pass_s = pass.data() + s * p.k_total;
            for (int64_t parent = 0; parent < p.k_total; ++parent) {
                const float dot = v2_dot_dispatch(
                    qp + qh * p.d + start, cp_base + parent * width,
                    static_cast<int>(width));
                pass_s[parent] = static_cast<uint8_t>(dot + qn * rp[parent] >= th);
            }
        }

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        alignas(64) float o_run[V2_D_MAX];

        for (int64_t j = 0; j < p.n_scan; ++j) {
            if (invp[kvh * p.n_pad + j]) continue;
            if (!v2_child_passes_local(
                    kvh, j, p.n_pad, p.bf, p.k_total, p.s_count,
                    pass.data(), assign_ptrs)) {
                continue;
            }
            const float* key = krp + (kvh * p.n_pad + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
            v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
        }

        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = bvp + (kvh * p.n_buf + j) * p.d_v;
            v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
        }

        if (l_run > 0.0f) {
            const float inv = 1.0f / l_run;
            for (int64_t x = 0; x < p.d_v; ++x) outp[qh * p.d_v + x] = o_run[x] * inv;
        } else {
            std::fill(outp + qh * p.d_v, outp + (qh + 1) * p.d_v, 0.0f);
        }
    }
    return out;
}

torch::Tensor attend_v2_exact_tiled(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    V2Prepared p = v2_prepare(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj, q_head_to_kv_obj);

    auto out = torch::empty({p.h_q, p.d_v}, p.q.options().dtype(torch::kFloat32));
    const float* qp = p.q.data_ptr<float>();
    const float* thp = p.th.data_ptr<float>();
    const float* krp = p.keys_reord.data_ptr<float>();
    const float* vrp = p.values_reord.data_ptr<float>();
    const bool* invp = p.invalid_mask.data_ptr<bool>();
    float* outp = out.data_ptr<float>();
    const float* bkp = p.has_buffer ? p.buffer_keys.data_ptr<float>() : nullptr;
    const float* bvp = p.has_buffer ? p.buffer_values.data_ptr<float>() : nullptr;

    std::vector<const float*> center_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const float*> radius_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const int32_t*> assign_ptrs(static_cast<size_t>(p.s_count));
    for (int64_t s = 0; s < p.s_count; ++s) {
        center_ptrs[static_cast<size_t>(s)] = p.centers[s].data_ptr<float>();
        radius_ptrs[static_cast<size_t>(s)] = p.radii[s].data_ptr<float>();
        assign_ptrs[static_cast<size_t>(s)] = p.assigns[s].data_ptr<int32_t>();
    }

    const int64_t total_pass = p.h_q * p.s_count * p.k_total;
    std::unique_ptr<uint8_t[]> pass(new uint8_t[static_cast<size_t>(total_pass)]);
    std::unique_ptr<float[]> q_norms(new float[static_cast<size_t>(p.h_q * p.s_count)]);

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;
        q_norms[task] = std::sqrt(std::max(
            v2_dot_dispatch(qp + qh * p.d + start, qp + qh * p.d + start,
                            static_cast<int>(width)),
            0.0f));
    }

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;
        const float qn = q_norms[task];
        const float th = thp[s * p.h_q + qh];
        const float* cp_base = center_ptrs[static_cast<size_t>(s)] + kvh * p.k_total * width;
        const float* rp = radius_ptrs[static_cast<size_t>(s)] + kvh * p.k_total;
        uint8_t* pass_s = pass.get() + (qh * p.s_count + s) * p.k_total;
        for (int64_t parent = 0; parent < p.k_total; ++parent) {
            const float dot = v2_dot_dispatch(
                qp + qh * p.d + start, cp_base + parent * width,
                static_cast<int>(width));
            pass_s[parent] = static_cast<uint8_t>(dot + qn * rp[parent] >= th);
        }
    }

    const int64_t n_tiles = (p.k_scan + V2_PARENTS_PER_TILE - 1) / V2_PARENTS_PER_TILE;
    const int64_t total_tiles = p.h_q * n_tiles;
    std::unique_ptr<float[]> tile_m(new float[static_cast<size_t>(total_tiles)]);
    std::unique_ptr<float[]> tile_l(new float[static_cast<size_t>(total_tiles)]);
    std::unique_ptr<float[]> tile_o(new float[static_cast<size_t>(total_tiles * p.d_v)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles; ++task) {
        const int64_t qh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const int64_t parent_lo = tile * V2_PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + V2_PARENTS_PER_TILE, p.k_scan);

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        float* o_run = tile_o.get() + task * p.d_v;

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;
                if (!v2_child_passes_global(
                        qh, kvh, j, p.n_pad, p.bf, p.k_total, p.s_count,
                        pass.get(), assign_ptrs)) {
                    continue;
                }
                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                    * static_cast<float>(scale);
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
            }
        }
        tile_m[task] = m_run;
        tile_l[task] = l_run;
    }

    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        alignas(64) float o_acc[V2_D_MAX];

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t task = qh * n_tiles + tile;
            v2_merge_online(m_acc, l_acc, o_acc,
                            tile_m[task], tile_l[task],
                            tile_o.get() + task * p.d_v, p.d_v);
        }

        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = bvp + (kvh * p.n_buf + j) * p.d_v;
            v2_online_update(score, val, m_acc, l_acc, o_acc, p.d_v);
        }

        if (l_acc > 0.0f) {
            const float inv = 1.0f / l_acc;
            for (int64_t x = 0; x < p.d_v; ++x) outp[qh * p.d_v + x] = o_acc[x] * inv;
        } else {
            std::fill(outp + qh * p.d_v, outp + (qh + 1) * p.d_v, 0.0f);
        }
    }
    return out;
}

torch::Tensor attend_v2_exact_gqa_tiled(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    V2Prepared p = v2_prepare(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj, q_head_to_kv_obj);

    std::vector<std::vector<int64_t>> qh_per_kv(static_cast<size_t>(p.h_kv));
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        TORCH_CHECK(kvh >= 0 && kvh < p.h_kv, "q_head_to_kv entry out of range");
        qh_per_kv[static_cast<size_t>(kvh)].push_back(qh);
    }
    int64_t max_groups = 0;
    for (const auto& v : qh_per_kv) max_groups = std::max<int64_t>(max_groups, v.size());
    TORCH_CHECK(max_groups <= V2_MAX_GROUPS, "v2.2 max GQA group size ", V2_MAX_GROUPS);

    auto out = torch::empty({p.h_q, p.d_v}, p.q.options().dtype(torch::kFloat32));
    const float* qp = p.q.data_ptr<float>();
    const float* thp = p.th.data_ptr<float>();
    const float* krp = p.keys_reord.data_ptr<float>();
    const float* vrp = p.values_reord.data_ptr<float>();
    const bool* invp = p.invalid_mask.data_ptr<bool>();
    float* outp = out.data_ptr<float>();
    const float* bkp = p.has_buffer ? p.buffer_keys.data_ptr<float>() : nullptr;
    const float* bvp = p.has_buffer ? p.buffer_values.data_ptr<float>() : nullptr;

    std::vector<const float*> center_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const float*> radius_ptrs(static_cast<size_t>(p.s_count));
    std::vector<const int32_t*> assign_ptrs(static_cast<size_t>(p.s_count));
    for (int64_t s = 0; s < p.s_count; ++s) {
        center_ptrs[static_cast<size_t>(s)] = p.centers[s].data_ptr<float>();
        radius_ptrs[static_cast<size_t>(s)] = p.radii[s].data_ptr<float>();
        assign_ptrs[static_cast<size_t>(s)] = p.assigns[s].data_ptr<int32_t>();
    }

    const int64_t total_pass = p.h_q * p.s_count * p.k_total;
    std::unique_ptr<uint8_t[]> pass(new uint8_t[static_cast<size_t>(total_pass)]);
    std::unique_ptr<float[]> q_norms(new float[static_cast<size_t>(p.h_q * p.s_count)]);

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;
        q_norms[task] = std::sqrt(std::max(
            v2_dot_dispatch(qp + qh * p.d + start, qp + qh * p.d + start,
                            static_cast<int>(width)),
            0.0f));
    }

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;
        const float qn = q_norms[task];
        const float th = thp[s * p.h_q + qh];
        const float* cp_base = center_ptrs[static_cast<size_t>(s)] + kvh * p.k_total * width;
        const float* rp = radius_ptrs[static_cast<size_t>(s)] + kvh * p.k_total;
        uint8_t* pass_s = pass.get() + (qh * p.s_count + s) * p.k_total;
        for (int64_t parent = 0; parent < p.k_total; ++parent) {
            const float dot = v2_dot_dispatch(
                qp + qh * p.d + start, cp_base + parent * width,
                static_cast<int>(width));
            pass_s[parent] = static_cast<uint8_t>(dot + qn * rp[parent] >= th);
        }
    }

    const int64_t n_tiles = (p.k_scan + V2_PARENTS_PER_TILE - 1) / V2_PARENTS_PER_TILE;
    const int64_t total_tiles_kv = p.h_kv * n_tiles;
    const int64_t total_tiles_q = p.h_q * n_tiles;
    std::unique_ptr<float[]> tile_m(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_l(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_o(new float[static_cast<size_t>(total_tiles_q * p.d_v)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles_kv; ++task) {
        const int64_t kvh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        const int64_t G = static_cast<int64_t>(qh_list.size());
        if (G == 0) continue;

        float m_run[V2_MAX_GROUPS];
        float l_run[V2_MAX_GROUPS];
        alignas(64) float o_run[V2_MAX_GROUPS * V2_D_MAX];
        for (int64_t g = 0; g < G; ++g) {
            m_run[g] = -std::numeric_limits<float>::infinity();
            l_run[g] = 0.0f;
        }

        const int64_t parent_lo = tile * V2_PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + V2_PARENTS_PER_TILE, p.k_scan);

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                uint32_t group_pass = 0;
                for (int64_t g = 0; g < G; ++g) {
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    if (v2_child_passes_global(
                            qh, kvh, j, p.n_pad, p.bf, p.k_total, p.s_count,
                            pass.get(), assign_ptrs)) {
                        group_pass |= (1u << g);
                    }
                }
                if (group_pass == 0) continue;

                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                for (int64_t g = 0; g < G; ++g) {
                    if (!((group_pass >> g) & 1u)) continue;
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                        * static_cast<float>(scale);
                    v2_online_update(score, val, m_run[g], l_run[g],
                                     o_run + g * V2_D_MAX, p.d_v);
                }
            }
        }

        for (int64_t g = 0; g < G; ++g) {
            const int64_t qh = qh_list[static_cast<size_t>(g)];
            const int64_t flat = qh * n_tiles + tile;
            tile_m[flat] = m_run[g];
            tile_l[flat] = l_run[g];
            if (l_run[g] > 0.0f) {
                std::copy(o_run + g * V2_D_MAX, o_run + g * V2_D_MAX + p.d_v,
                          tile_o.get() + flat * p.d_v);
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        alignas(64) float o_acc[V2_D_MAX];

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t task = qh * n_tiles + tile;
            v2_merge_online(m_acc, l_acc, o_acc,
                            tile_m[task], tile_l[task],
                            tile_o.get() + task * p.d_v, p.d_v);
        }

        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = bvp + (kvh * p.n_buf + j) * p.d_v;
            v2_online_update(score, val, m_acc, l_acc, o_acc, p.d_v);
        }

        if (l_acc > 0.0f) {
            const float inv = 1.0f / l_acc;
            for (int64_t x = 0; x < p.d_v; ++x) outp[qh * p.d_v + x] = o_acc[x] * inv;
        } else {
            std::fill(outp + qh * p.d_v, outp + (qh + 1) * p.d_v, 0.0f);
        }
    }
    return out;
}

}  // namespace
