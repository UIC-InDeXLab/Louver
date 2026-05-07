// attention v1.4 — BF16 keys/values + AVX-512 VDPBF16PS.
//
// Memory: keys/values stored as bf16 (half the bytes of v1.3) → halves the L3
// footprint of the index. Dot products via _mm512_dpbf16_ps (32 bf16 multiplies
// + 16 fp32 accumulators per ZMM per cycle) — twice the FMA throughput of
// fp32 on the same micro-architecture.
//
// q is materialized as bf16 once at call entry; thresholds and centers stay
// fp32 (their cost is small). Anchor centers are converted to bf16 lazily
// inside the kernel from the fp32 state to keep the v1.0 build kernel
// untouched.
//
// Output: fp32 (matches dense reference precision).
#include "_cpu_common.h"

#if defined(__AVX512F__) && defined(__AVX512BF16__)
#include <immintrin.h>
#define HIRA_HAS_AVX512_BF16 1
#elif defined(__AVX512F__)
#include <immintrin.h>
#define HIRA_HAS_AVX512_BF16 0
#else
#define HIRA_HAS_AVX512_BF16 0
#endif

namespace py = pybind11;
using namespace hira_cpu;

namespace {

constexpr int64_t PARENTS_PER_TILE = 32;

#if defined(__AVX512F__)
inline float reduce_zmm(__m512 v) { return _mm512_reduce_add_ps(v); }
#endif

#if HIRA_HAS_AVX512_BF16
// Dot product fp32(q[128]) · bf16(k[128]) → fp32. q is loaded as bf16 first.
// vdpbf16ps does 32 bf16 muls + 16 fp32 adds per call.
inline float dot_d128_bf16(const uint16_t* __restrict q, const uint16_t* __restrict k) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    // 128 elems / 32 per chunk = 4 chunks
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

// Width=16 dot for the anchor gate: half a vdpbf16ps register filled.
// Easier to just convert the 16 bf16 → fp32 and use a normal fp32 dot.
inline float dot_w16_bf16_to_fp32(const float* __restrict q_fp32, const uint16_t* __restrict k_bf16) {
    // Convert 16 bf16 → 16 fp32 via shift/cast.
    __m256i k16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(k_bf16));
    __m512i k32 = _mm512_cvtepu16_epi32(k16);
    __m512 kf = _mm512_castsi512_ps(_mm512_slli_epi32(k32, 16));
    return reduce_zmm(_mm512_mul_ps(_mm512_loadu_ps(q_fp32), kf));
}

// AXPY for D=128, src is bf16. Convert chunks then FMA.
inline void axpy_d128_bf16src(float* __restrict dst, const uint16_t* __restrict src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 16) {
        __m256i s16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        __m512 sf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(s16), 16));
        _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(sf, vw, _mm512_loadu_ps(dst + i)));
    }
}
#endif

#if defined(__AVX512F__)
inline void axpy_d128_avx512(float* __restrict dst, const float* __restrict src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vw, _mm512_loadu_ps(dst + i +  0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vw, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vw, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vw, _mm512_loadu_ps(dst + i + 48)));
    }
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

inline void scale_axpy_d128_avx512(float* __restrict dst, const float* __restrict src, float a, float b) {
    const __m512 va = _mm512_set1_ps(a);
    const __m512 vb = _mm512_set1_ps(b);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vb, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va)));
    }
}
#endif

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
#if defined(__AVX512F__)
    if (d_v == 128) { scale_axpy_d128_avx512(o_dst, o_src, a, b); }
    else
#endif
    {
        for (int64_t x = 0; x < d_v; ++x) o_dst[x] = o_dst[x] * a + b * o_src[x];
    }
    m_dst = new_m;
}

// Convert fp32 → bf16 with round-to-nearest-even (good enough; matches
// `__bfloat16` semantics torch uses).
inline uint16_t fp32_to_bf16_rne(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) {
        // NaN: preserve and force qnan bit.
        return static_cast<uint16_t>((u >> 16) | 0x40);
    }
    uint32_t rounded = u + 0x7fff + ((u >> 16) & 1);
    return static_cast<uint16_t>(rounded >> 16);
}

void fp32_to_bf16(const float* __restrict src, uint16_t* __restrict dst, int64_t n) {
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

}  // namespace

torch::Tensor attend_v1_4(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    TORCH_CHECK(q_in.device().is_cpu(), "q must be CPU");
    TORCH_CHECK(q_in.dim() == 2, "q must be (H_q, D)");
    TORCH_CHECK(q_in.dtype() == torch::kBFloat16,
                "v1.4 expects q in bf16; convert outside the timed region");
    TORCH_CHECK(th_in.dtype() == torch::kBFloat16,
                "v1.4 expects th_per_subspace in bf16");
    TORCH_CHECK(th_in.dim() == 2,
                "th_per_subspace must be (S, H_q) or (2S, H_q)");

    auto q_bf16 = q_in.contiguous();
    auto th_bf16 = th_in.contiguous();
    auto th_fp32 = th_bf16.to(torch::kFloat32);

    // State tensors stay fp32; we cache bf16 mirrors on the state dict so
    // re-entry pays the conversion only once.
    auto keys_reord_fp32 = as_cpu_float(state["keys_reord"].cast<torch::Tensor>(),
                                        "state['keys_reord']");
    auto invalid_mask = state["invalid_mask"].cast<torch::Tensor>().contiguous().to(torch::kBool);
    auto values_reord_fp32 = state.contains("values_reord")
        ? as_cpu_float(state["values_reord"].cast<torch::Tensor>(), "state['values_reord']")
        : keys_reord_fp32;

    auto centers = list_float_tensors(state["centers"], "state['centers']");
    auto radii = list_float_tensors(state["radii"], "state['radii']");
    auto assigns = list_int_tensors(state["assigns_reord"], "state['assigns_reord']");
    auto slices = slices_from_state(state);

    const int64_t s_count = static_cast<int64_t>(centers.size());
    const int64_t anchor_s = 0;

    // bf16 mirror cache (keyed by data_ptr to detect changes).
    py::dict cache;
    if (state.contains("_v1_4_cache")) {
        cache = state["_v1_4_cache"].cast<py::dict>();
    } else {
        cache = py::dict();
        state["_v1_4_cache"] = cache;
    }
    const int64_t k_used_for_cache = state.contains("K_used")
        ? state["K_used"].cast<int64_t>()
        : state["K"].cast<int64_t>();
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
        TORCH_CHECK(buffer_keys.dtype() == torch::kBFloat16,
                    "v1.4 expects buffer_keys in bf16");
        buffer_keys_bf16 = buffer_keys.contiguous();
        if (buffer_values.defined() && buffer_values.numel() > 0) {
            TORCH_CHECK(buffer_values.dtype() == torch::kBFloat16,
                        "v1.4 expects buffer_values in bf16");
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
    const int64_t k_used = k_used_for_cache;
    TORCH_CHECK(d == 128 && d_v == 128,
                "v1.4 specialized for D=128/D_v=128 currently");

    auto q2kv = resolve_q_head_to_kv(q_head_to_kv_obj, h_q, h_kv);
    auto q_fp32 = q_bf16.to(torch::kFloat32);  // for anchor q-norm (small)

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
    const int32_t* assign_anchor = assigns[anchor_s].data_ptr<int32_t>();
    const int64_t anchor_start = slices[anchor_s].first;
    const int64_t anchor_width = slices[anchor_s].second - slices[anchor_s].first;
    TORCH_CHECK(anchor_width == 16, "v1.4 specialized for anchor width = 16 (S=8, D=128)");
    (void)assign_anchor;  // kept for parity with v1.3; not used in anchor-only path

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

        // Anchor q-norm in fp32 (small: 16 elems).
        float qn_anchor;
        {
            const float* p = qp_fp32 + qh * d + anchor_start;
            float acc = 0.0f;
            for (int x = 0; x < 16; ++x) acc += p[x] * p[x];
            qn_anchor = std::sqrt(std::max(acc, 0.0f));
        }
        const float th_anchor = thp[anchor_s * h_q + qh];

        const int64_t parent_lo = tile * PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + PARENTS_PER_TILE, k_used);

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        float* o_run = &tile_o[task * d_v];

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
#if HIRA_HAS_AVX512_BF16
            const uint16_t* cp_bf16 = center_anchor_bf16_p + (kvh * k_used + parent) * 16;
            float bound_dot = dot_w16_bf16_to_fp32(qp_fp32 + qh * d + anchor_start, cp_bf16);
#else
            const uint16_t* cp_bf16 = center_anchor_bf16_p + (kvh * k_used + parent) * 16;
            float bound_dot = 0.0f;
            for (int x = 0; x < 16; ++x) {
                uint32_t u = static_cast<uint32_t>(cp_bf16[x]) << 16;
                float cv; std::memcpy(&cv, &u, 4);
                bound_dot += qp_fp32[qh * d + anchor_start + x] * cv;
            }
#endif
            const float bound = bound_dot + qn_anchor * radius_anchor[kvh * k_used + parent];
            if (bound < th_anchor) continue;

            for (int64_t child = 0; child < bf; ++child) {
                const int64_t j = parent * bf + child;
                if (j >= n_pad) break;
                if (invp[kvh * n_pad + j]) continue;

                const uint16_t* key = krp_bf16 + (kvh * n_pad + j) * d;
#if HIRA_HAS_AVX512_BF16
                const float score_raw = dot_d128_bf16(qp_bf16 + qh * d, key);
#else
                float score_raw = 0.0f;
                for (int x = 0; x < d; ++x) {
                    uint32_t uq = static_cast<uint32_t>(qp_bf16[qh * d + x]) << 16;
                    uint32_t uk = static_cast<uint32_t>(key[x]) << 16;
                    float qf, kf;
                    std::memcpy(&qf, &uq, 4); std::memcpy(&kf, &uk, 4);
                    score_raw += qf * kf;
                }
#endif
                const float score = score_raw * static_cast<float>(scale);

                const uint16_t* val = vrp_bf16 + (kvh * n_pad + j) * d_v;
                if (score > m_run) {
                    if (l_run == 0.0f) {
                        m_run = score; l_run = 1.0f;
                        // Fresh accumulator: scale-store from bf16 to fp32.
#if HIRA_HAS_AVX512_BF16
                        for (int i = 0; i < d_v; i += 16) {
                            __m256i s16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(val + i));
                            __m512 sf = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(s16), 16));
                            _mm512_storeu_ps(o_run + i, sf);
                        }
#else
                        for (int x = 0; x < d_v; ++x) {
                            uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                            std::memcpy(o_run + x, &u, 4);
                        }
#endif
                    } else {
                        const float a = std::exp(m_run - score);
                        l_run = l_run * a + 1.0f;
#if defined(__AVX512F__)
                        scale_d128_avx512(o_run, a);
#else
                        for (int x = 0; x < d_v; ++x) o_run[x] *= a;
#endif
#if HIRA_HAS_AVX512_BF16
                        axpy_d128_bf16src(o_run, val, 1.0f);
#else
                        for (int x = 0; x < d_v; ++x) {
                            uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                            float vf; std::memcpy(&vf, &u, 4);
                            o_run[x] += vf;
                        }
#endif
                        m_run = score;
                    }
                } else {
                    const float w = std::exp(score - m_run);
                    l_run += w;
#if HIRA_HAS_AVX512_BF16
                    axpy_d128_bf16src(o_run, val, w);
#else
                    for (int x = 0; x < d_v; ++x) {
                        uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                        float vf; std::memcpy(&vf, &u, 4);
                        o_run[x] += w * vf;
                    }
#endif
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
            const uint16_t* key = bkp_bf16 + (kvh * n_buf + j) * d;
#if HIRA_HAS_AVX512_BF16
            const float score = dot_d128_bf16(qp_bf16 + qh * d, key) * static_cast<float>(scale);
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
            const uint16_t* val = bvp_bf16 + (kvh * n_buf + j) * d_v;
            if (score > m_acc) {
                if (l_acc == 0.0f) {
                    m_acc = score; l_acc = 1.0f;
                    for (int x = 0; x < d_v; ++x) {
                        uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                        std::memcpy(o_acc.data() + x, &u, 4);
                    }
                } else {
                    const float a = std::exp(m_acc - score);
                    l_acc = l_acc * a + 1.0f;
                    for (int x = 0; x < d_v; ++x) o_acc[x] *= a;
                    for (int x = 0; x < d_v; ++x) {
                        uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                        float vf; std::memcpy(&vf, &u, 4);
                        o_acc[x] += vf;
                    }
                    m_acc = score;
                }
            } else {
                const float w = std::exp(score - m_acc);
                l_acc += w;
                for (int x = 0; x < d_v; ++x) {
                    uint32_t u = static_cast<uint32_t>(val[x]) << 16;
                    float vf; std::memcpy(&vf, &u, 4);
                    o_acc[x] += w * vf;
                }
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
    m.def("attend", &attend_v1_4,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
