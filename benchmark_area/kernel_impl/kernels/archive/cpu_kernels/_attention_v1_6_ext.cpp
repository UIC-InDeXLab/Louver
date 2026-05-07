// attention v1.6 — final fused fast kernel: BF16 storage + AVX-512 BF16 dot
// + anchor-only gate + GQA-aware query batching.
//
// This is the union of every winning idea from v1.1..v1.5:
//   - online softmax single-pass (v1.1)
//   - parent-tile parallelism (v1.1)
//   - anchor-only subspace gate (v1.3)
//   - bf16 keys/values + VDPBF16PS dot (v1.4)
//   - GQA-aware key reuse: parallelize over (kvh, tile) and inner-loop
//     over the G query heads sharing a kv head (v1.5)
//
// Inputs are bf16 (q, th_packed, buffer); state's fp32 keys/values are
// converted to bf16 once on first call and cached on the state dict.
#include "_cpu_common.h"

#if defined(__AVX512BF16__)
#include <immintrin.h>
#define HIRA_HAS_AVX512_BF16 1
#else
#define HIRA_HAS_AVX512_BF16 0
#endif

namespace py = pybind11;
using namespace hira_cpu;

namespace {

constexpr int64_t PARENTS_PER_TILE = 32;
constexpr int64_t MAX_GROUPS = 16;
constexpr int64_t D_MAX = 256;

#if HIRA_HAS_AVX512_BF16
inline float reduce_zmm(__m512 v) { return _mm512_reduce_add_ps(v); }

inline float dot_d128_bf16(const uint16_t* q, const uint16_t* k) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    acc0 = _mm512_dpbf16_ps(acc0,
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(q +  0)),
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(k +  0)));
    acc1 = _mm512_dpbf16_ps(acc1,
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(q + 32)),
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(k + 32)));
    acc2 = _mm512_dpbf16_ps(acc2,
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(q + 64)),
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(k + 64)));
    acc3 = _mm512_dpbf16_ps(acc3,
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(q + 96)),
        (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(k + 96)));
    return reduce_zmm(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

inline float dot_w16_bf16_to_fp32(const float* q_fp32, const uint16_t* k_bf16) {
    __m256i k16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_bf16));
    __m512 kf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(k16), 16));
    return reduce_zmm(_mm512_mul_ps(_mm512_loadu_ps(q_fp32), kf));
}

inline void axpy_d128_bf16src(float* dst, const uint16_t* src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 16) {
        __m256i s16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m512 sf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(s16), 16));
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(sf, vw, _mm512_loadu_ps(dst + i)));
    }
}

inline void scale_d128_avx512(float* dst, float a) {
    const __m512 va = _mm512_set1_ps(a);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va));
    }
}

inline void scale_axpy_d128_avx512_bf16src(float* dst, const uint16_t* src, float a, float b) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    for (int i = 0; i < 128; i += 16) {
        __m256i s16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m512 sf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(s16), 16));
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(sf, vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i), va)));
    }
}

inline void scale_axpy_d128_avx512(float* dst, const float* src, float a, float b) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va)));
    }
}

inline void copy_bf16_to_fp32_d128(float* dst, const uint16_t* src) {
    for (int i = 0; i < 128; i += 16) {
        __m256i s16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m512 sf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(s16), 16));
        _mm512_storeu_ps(dst + i, sf);
    }
}
#endif

inline uint16_t fp32_to_bf16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) {
        return static_cast<uint16_t>((u >> 16) | 0x40);
    }
    uint32_t rounded = u + 0x7fff + ((u >> 16) & 1);
    return static_cast<uint16_t>(rounded >> 16);
}

void fp32_to_bf16(const float* src, uint16_t* dst, int64_t n) {
#if defined(__AVX512BF16__)
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512 a = _mm512_loadu_ps(src + i +  0);
        __m512 b = _mm512_loadu_ps(src + i + 16);
        __m512bh r = _mm512_cvtne2ps_pbh(b, a);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(dst + i), (__m512i)r);
    }
    for (; i + 16 <= n; i += 16) {
        __m512 a = _mm512_loadu_ps(src + i);
        __m256bh r = _mm512_cvtneps_pbh(a);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), (__m256i)r);
    }
    for (; i < n; ++i) dst[i] = fp32_to_bf16_rne(src[i]);
#else
    for (int64_t i = 0; i < n; ++i) dst[i] = fp32_to_bf16_rne(src[i]);
#endif
}

inline void merge_online_d128(
    float& m_dst, float& l_dst, float* o_dst,
    float m_src, float l_src, const float* o_src) {
    if (l_src == 0.0f) return;
    if (l_dst == 0.0f) {
        m_dst = m_src; l_dst = l_src;
        std::copy(o_src, o_src + 128, o_dst);
        return;
    }
    const float new_m = std::max(m_dst, m_src);
    const float a = std::exp(m_dst - new_m);
    const float b = std::exp(m_src - new_m);
    l_dst = l_dst * a + l_src * b;
#if HIRA_HAS_AVX512_BF16
    scale_axpy_d128_avx512(o_dst, o_src, a, b);
#else
    for (int x = 0; x < 128; ++x) o_dst[x] = a * o_dst[x] + b * o_src[x];
#endif
    m_dst = new_m;
}

}  // namespace

torch::Tensor attend_v1_6(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    TORCH_CHECK(q_in.device().is_cpu(), "q must be CPU");
    TORCH_CHECK(q_in.dtype() == torch::kBFloat16, "v1.6 expects bf16 q");
    TORCH_CHECK(th_in.dtype() == torch::kBFloat16, "v1.6 expects bf16 thresholds");
    TORCH_CHECK(q_in.dim() == 2, "q must be (H_q, D)");

    auto q_bf16 = q_in.contiguous();
    auto th_fp32 = th_in.contiguous().to(torch::kFloat32);

    auto keys_reord_fp32 = as_cpu_float(state["keys_reord"].cast<torch::Tensor>(),
                                        "state['keys_reord']");
    auto invalid_mask = state["invalid_mask"].cast<torch::Tensor>().contiguous().to(torch::kBool);
    auto values_reord_fp32 = state.contains("values_reord")
        ? as_cpu_float(state["values_reord"].cast<torch::Tensor>(), "state['values_reord']")
        : keys_reord_fp32;

    auto centers = list_float_tensors(state["centers"], "state['centers']");
    auto radii = list_float_tensors(state["radii"], "state['radii']");
    auto slices = slices_from_state(state);

    const int64_t s_count = static_cast<int64_t>(centers.size());
    const int64_t anchor_s = 0;
    TORCH_CHECK(th_in.dim() == 2 && th_in.size(0) >= s_count && th_in.size(1) == q_in.size(0),
                "threshold shape mismatch");

    // bf16 mirror cache.
    py::dict cache;
    if (state.contains("_v1_6_cache")) cache = state["_v1_6_cache"].cast<py::dict>();
    else { cache = py::dict(); state["_v1_6_cache"] = cache; }
    const int64_t key_signature[3] = {
        reinterpret_cast<int64_t>(keys_reord_fp32.data_ptr<float>()),
        reinterpret_cast<int64_t>(values_reord_fp32.data_ptr<float>()),
        reinterpret_cast<int64_t>(centers[anchor_s].data_ptr<float>()),
    };
    bool valid = cache.contains("sig") && cache.contains("keys_bf16")
                 && cache.contains("values_bf16") && cache.contains("center_anchor_bf16");
    if (valid) {
        auto sig = cache["sig"].cast<py::tuple>();
        valid = sig[0].cast<int64_t>() == key_signature[0]
             && sig[1].cast<int64_t>() == key_signature[1]
             && sig[2].cast<int64_t>() == key_signature[2];
    }
    torch::Tensor keys_bf16, values_bf16, center_anchor_bf16;
    if (!valid) {
        keys_bf16 = torch::empty(keys_reord_fp32.sizes(),
                                 keys_reord_fp32.options().dtype(torch::kBFloat16));
        values_bf16 = torch::empty(values_reord_fp32.sizes(),
                                   values_reord_fp32.options().dtype(torch::kBFloat16));
        fp32_to_bf16(keys_reord_fp32.data_ptr<float>(),
                     reinterpret_cast<uint16_t*>(keys_bf16.data_ptr()),
                     keys_reord_fp32.numel());
        fp32_to_bf16(values_reord_fp32.data_ptr<float>(),
                     reinterpret_cast<uint16_t*>(values_bf16.data_ptr()),
                     values_reord_fp32.numel());
        center_anchor_bf16 = torch::empty(centers[anchor_s].sizes(),
                                          centers[anchor_s].options().dtype(torch::kBFloat16));
        fp32_to_bf16(centers[anchor_s].data_ptr<float>(),
                     reinterpret_cast<uint16_t*>(center_anchor_bf16.data_ptr()),
                     centers[anchor_s].numel());
        cache["keys_bf16"] = keys_bf16;
        cache["values_bf16"] = values_bf16;
        cache["center_anchor_bf16"] = center_anchor_bf16;
        cache["sig"] = py::make_tuple(key_signature[0], key_signature[1], key_signature[2]);
    } else {
        keys_bf16 = cache["keys_bf16"].cast<torch::Tensor>();
        values_bf16 = cache["values_bf16"].cast<torch::Tensor>();
        center_anchor_bf16 = cache["center_anchor_bf16"].cast<torch::Tensor>();
    }

    torch::Tensor buffer_keys = object_to_tensor_or_empty(buffer_keys_obj);
    torch::Tensor buffer_values = object_to_tensor_or_empty(buffer_values_obj);
    const bool has_buffer = buffer_keys.defined() && buffer_keys.numel() > 0;
    torch::Tensor buffer_keys_bf16, buffer_values_bf16;
    if (has_buffer) {
        TORCH_CHECK(buffer_keys.dtype() == torch::kBFloat16, "v1.6 expects bf16 buffer_keys");
        buffer_keys_bf16 = buffer_keys.contiguous();
        if (buffer_values.defined() && buffer_values.numel() > 0) {
            TORCH_CHECK(buffer_values.dtype() == torch::kBFloat16, "v1.6 expects bf16 buffer_values");
            buffer_values_bf16 = buffer_values.contiguous();
        } else {
            buffer_values_bf16 = buffer_keys_bf16;
        }
    }

    const int64_t h_q = q_bf16.size(0);
    const int64_t d = q_bf16.size(1);
    const int64_t h_kv = keys_reord_fp32.size(0);
    const int64_t n_pad = keys_reord_fp32.size(1);
    const int64_t d_v = values_reord_fp32.size(2);
    const int64_t bf = state["bf"].cast<int64_t>();
    const int64_t k_used = state.contains("K_used")
        ? state["K_used"].cast<int64_t>()
        : state["K"].cast<int64_t>();
    TORCH_CHECK(d == 128 && d_v == 128, "v1.6 specialized for D=128/D_v=128");

    auto q2kv = resolve_q_head_to_kv(q_head_to_kv_obj, h_q, h_kv);
    auto q_fp32 = q_bf16.to(torch::kFloat32);

    std::vector<std::vector<int64_t>> qh_per_kv(static_cast<size_t>(h_kv));
    for (int64_t qh = 0; qh < h_q; ++qh) {
        const int64_t kvh = q2kv[qh];
        TORCH_CHECK(kvh >= 0 && kvh < h_kv, "q_head_to_kv out of range");
        qh_per_kv[static_cast<size_t>(kvh)].push_back(qh);
    }
    int64_t max_groups = 0;
    for (const auto& v : qh_per_kv) max_groups = std::max<int64_t>(max_groups, v.size());
    TORCH_CHECK(max_groups <= MAX_GROUPS, "v1.6: max GQA group size ", MAX_GROUPS);

    auto out = torch::zeros({h_q, d_v}, q_bf16.options().dtype(torch::kFloat32));
    const uint16_t* qp_bf16 = reinterpret_cast<const uint16_t*>(q_bf16.data_ptr());
    const float* qp_fp32 = q_fp32.data_ptr<float>();
    const float* thp = th_fp32.data_ptr<float>();
    const uint16_t* krp_bf16 = reinterpret_cast<const uint16_t*>(keys_bf16.data_ptr());
    const uint16_t* vrp_bf16 = reinterpret_cast<const uint16_t*>(values_bf16.data_ptr());
    const bool* invp = invalid_mask.data_ptr<bool>();
    float* outp = out.data_ptr<float>();
    const uint16_t* bkp_bf16 = has_buffer ? reinterpret_cast<const uint16_t*>(buffer_keys_bf16.data_ptr()) : nullptr;
    const uint16_t* bvp_bf16 = has_buffer ? reinterpret_cast<const uint16_t*>(buffer_values_bf16.data_ptr()) : nullptr;
    const int64_t n_buf = has_buffer ? buffer_keys_bf16.size(1) : 0;

    const uint16_t* center_anchor_bf16_p = reinterpret_cast<const uint16_t*>(center_anchor_bf16.data_ptr());
    const float* radius_anchor = radii[anchor_s].data_ptr<float>();
    const int64_t anchor_start = slices[anchor_s].first;
    const int64_t anchor_width = slices[anchor_s].second - slices[anchor_s].first;
    TORCH_CHECK(anchor_width == 16, "v1.6 specialized for anchor width = 16");

    const int64_t n_tiles = (k_used + PARENTS_PER_TILE - 1) / PARENTS_PER_TILE;
    const int64_t total_tiles_kv = h_kv * n_tiles;
    const int64_t total_tiles_q = h_q * n_tiles;

    std::vector<float> tile_m(static_cast<size_t>(total_tiles_q),
                              -std::numeric_limits<float>::infinity());
    std::vector<float> tile_l(static_cast<size_t>(total_tiles_q), 0.0f);
    std::vector<float> tile_o(static_cast<size_t>(total_tiles_q * d_v), 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles_kv; ++task) {
        const int64_t kvh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        const int64_t G = static_cast<int64_t>(qh_list.size());
        if (G == 0) continue;

        float m_run[MAX_GROUPS];
        float l_run[MAX_GROUPS];
        alignas(64) float o_run[MAX_GROUPS * D_MAX];
        for (int64_t g = 0; g < G; ++g) {
            m_run[g] = -std::numeric_limits<float>::infinity();
            l_run[g] = 0.0f;
        }

        float qn_anchor[MAX_GROUPS];
        float th_anchor[MAX_GROUPS];
        for (int64_t g = 0; g < G; ++g) {
            const int64_t qh = qh_list[g];
            const float* p = qp_fp32 + qh * d + anchor_start;
            float acc = 0.0f;
            for (int x = 0; x < 16; ++x) acc += p[x] * p[x];
            qn_anchor[g] = std::sqrt(std::max(acc, 0.0f));
            th_anchor[g] = thp[anchor_s * h_q + qh];
        }

        const int64_t parent_lo = tile * PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + PARENTS_PER_TILE, k_used);

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            const uint16_t* cp_bf16 = center_anchor_bf16_p + (kvh * k_used + parent) * 16;
            const float radius = radius_anchor[kvh * k_used + parent];

            uint16_t group_pass = 0;
            for (int64_t g = 0; g < G; ++g) {
                const int64_t qh = qh_list[g];
#if HIRA_HAS_AVX512_BF16
                const float bound_dot = dot_w16_bf16_to_fp32(qp_fp32 + qh * d + anchor_start, cp_bf16);
#else
                float bound_dot = 0.0f;
                for (int x = 0; x < 16; ++x) {
                    uint32_t u = static_cast<uint32_t>(cp_bf16[x]) << 16;
                    float cv; std::memcpy(&cv, &u, 4);
                    bound_dot += qp_fp32[qh * d + anchor_start + x] * cv;
                }
#endif
                const float bound = bound_dot + qn_anchor[g] * radius;
                if (bound >= th_anchor[g]) group_pass |= (1u << g);
            }
            if (group_pass == 0) continue;

            for (int64_t child = 0; child < bf; ++child) {
                const int64_t j = parent * bf + child;
                if (j >= n_pad) break;
                if (invp[kvh * n_pad + j]) continue;

                const uint16_t* key = krp_bf16 + (kvh * n_pad + j) * d;
                const uint16_t* val = vrp_bf16 + (kvh * n_pad + j) * d_v;

                for (int64_t g = 0; g < G; ++g) {
                    if (!((group_pass >> g) & 1u)) continue;
                    const int64_t qh = qh_list[g];
#if HIRA_HAS_AVX512_BF16
                    const float score = dot_d128_bf16(qp_bf16 + qh * d, key)
                                        * static_cast<float>(scale);
#else
                    float score = 0.0f;
                    for (int x = 0; x < d; ++x) {
                        uint32_t uq = static_cast<uint32_t>(qp_bf16[qh * d + x]) << 16;
                        uint32_t uk = static_cast<uint32_t>(key[x]) << 16;
                        float qf, kf;
                        std::memcpy(&qf, &uq, 4); std::memcpy(&kf, &uk, 4);
                        score += qf * kf;
                    }
                    score *= static_cast<float>(scale);
#endif
                    float* o_g = o_run + g * D_MAX;
                    if (score > m_run[g]) {
                        if (l_run[g] == 0.0f) {
                            m_run[g] = score; l_run[g] = 1.0f;
#if HIRA_HAS_AVX512_BF16
                            copy_bf16_to_fp32_d128(o_g, val);
#else
                            for (int x = 0; x < d_v; ++x) {
                                uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                                std::memcpy(o_g + x, &u, 4);
                            }
#endif
                        } else {
                            const float a = std::exp(m_run[g] - score);
                            l_run[g] = l_run[g] * a + 1.0f;
#if HIRA_HAS_AVX512_BF16
                            scale_axpy_d128_avx512_bf16src(o_g, val, a, 1.0f);
#else
                            for (int x = 0; x < d_v; ++x) {
                                uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                                float vf; std::memcpy(&vf, &u, 4);
                                o_g[x] = a * o_g[x] + vf;
                            }
#endif
                            m_run[g] = score;
                        }
                    } else {
                        const float w = std::exp(score - m_run[g]);
                        l_run[g] += w;
#if HIRA_HAS_AVX512_BF16
                        axpy_d128_bf16src(o_g, val, w);
#else
                        for (int x = 0; x < d_v; ++x) {
                            uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                            float vf; std::memcpy(&vf, &u, 4);
                            o_g[x] += w * vf;
                        }
#endif
                    }
                }
            }
        }

        for (int64_t g = 0; g < G; ++g) {
            const int64_t qh = qh_list[g];
            const int64_t flat = qh * n_tiles + tile;
            tile_m[flat] = m_run[g];
            tile_l[flat] = l_run[g];
            std::copy(o_run + g * D_MAX, o_run + g * D_MAX + d_v,
                      &tile_o[flat * d_v]);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < h_q; ++qh) {
        const int64_t kvh = q2kv[qh];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        alignas(64) float o_acc[D_MAX];
        std::fill(o_acc, o_acc + d_v, 0.0f);

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t task = qh * n_tiles + tile;
            merge_online_d128(m_acc, l_acc, o_acc,
                              tile_m[task], tile_l[task], &tile_o[task * d_v]);
        }

        for (int64_t j = 0; j < n_buf; ++j) {
            const uint16_t* key = bkp_bf16 + (kvh * n_buf + j) * d;
#if HIRA_HAS_AVX512_BF16
            const float score = dot_d128_bf16(qp_bf16 + qh * d, key) * static_cast<float>(scale);
#else
            float score = 0.0f;
            for (int x = 0; x < d; ++x) {
                uint32_t uq = static_cast<uint32_t>(qp_bf16[qh * d + x]) << 16;
                uint32_t uk = static_cast<uint32_t>(key[x]) << 16;
                float qf, kf; std::memcpy(&qf, &uq, 4); std::memcpy(&kf, &uk, 4);
                score += qf * kf;
            }
            score *= static_cast<float>(scale);
#endif
            const uint16_t* val = bvp_bf16 + (kvh * n_buf + j) * d_v;
            if (score > m_acc) {
                if (l_acc == 0.0f) {
                    m_acc = score; l_acc = 1.0f;
#if HIRA_HAS_AVX512_BF16
                    copy_bf16_to_fp32_d128(o_acc, val);
#else
                    for (int x = 0; x < d_v; ++x) {
                        uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                        std::memcpy(o_acc + x, &u, 4);
                    }
#endif
                } else {
                    const float a = std::exp(m_acc - score);
                    l_acc = l_acc * a + 1.0f;
#if HIRA_HAS_AVX512_BF16
                    scale_axpy_d128_avx512_bf16src(o_acc, val, a, 1.0f);
#else
                    for (int x = 0; x < d_v; ++x) {
                        uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                        float vf; std::memcpy(&vf, &u, 4);
                        o_acc[x] = a * o_acc[x] + vf;
                    }
#endif
                    m_acc = score;
                }
            } else {
                const float w = std::exp(score - m_acc);
                l_acc += w;
#if HIRA_HAS_AVX512_BF16
                axpy_d128_bf16src(o_acc, val, w);
#else
                for (int x = 0; x < d_v; ++x) {
                    uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                    float vf; std::memcpy(&vf, &u, 4);
                    o_acc[x] += w * vf;
                }
#endif
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
    m.def("attend", &attend_v1_6,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
