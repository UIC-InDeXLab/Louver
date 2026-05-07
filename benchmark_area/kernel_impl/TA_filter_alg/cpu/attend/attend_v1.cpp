// Fused TA-filter + sparse SDPA — CPU.
//
// One kernel does the entire decoding-time TA pipeline per query head:
//   1. Per subspace s in [0..S): score q_s · centers[s, h_kv, k, :w_s]
//      for k in [0..K_used).
//   2. Partial-sort each subspace's K_used scores descending; keep top L.
//   3. TA depth: smallest L* where Σ_s sorted[s][L] < threshold.
//   4. Build a per-subspace bitset of selected parent ids (top L*).
//   5. Online-softmax sweep over keys [0..N_used): a key is alive if any
//      subspace's parent (assigns[s, h_kv, k]) is in selected[s]. Full-D dot
//      and (m, l, o) update for alive keys.
//   6. Sweep buffer keys [0..l_buf) unconditionally.
//   7. out[h_q, :] = o / l.
//
// Specialised for S=4, bf=4, D=Dv=128, max_w=32. Other (D, w) fall back to
// generic dot.

#include "../_common.h"

namespace py = pybind11;
using namespace ta_cpu;

namespace {

constexpr int S_FIXED = 4;
constexpr int L_MAX = 256;
constexpr float EXP_PRUNE = -16.f;  // exp(-16) ≈ 1e-7, ignore.

#if HIRA_HAS_AVX512
// w=32 dot — two 16-wide loads, fused into one reduce.
inline float dot_w32(const float* __restrict a, const float* __restrict b) {
    __m512 lo = _mm512_mul_ps(_mm512_loadu_ps(a),      _mm512_loadu_ps(b));
    __m512 hi = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), _mm512_loadu_ps(b + 16), lo);
    return reduce_zmm(hi);
}
#else
inline float dot_w32(const float* a, const float* b) {
    float s = 0.f;
    for (int i = 0; i < 32; ++i) s += a[i] * b[i];
    return s;
}
#endif

inline float dot_w(const float* qs, const float* c, int w) {
    if (w == 32) return dot_w32(qs, c);
    return dot_generic(qs, c, w);
}

inline void update_online_softmax(
    float& m, float& l, float* __restrict o,
    float score, const float* __restrict v, int dv) {
    if (score > m) {
        float diff = m - score;
        float a = (diff > EXP_PRUNE) ? std::exp(diff) : 0.f;
        l = l * a + 1.0f;
        if (dv == 128) {
            if (a == 0.f) std::memset(o, 0, sizeof(float) * 128);
            else scale_d128(o, a);
            axpy_d128(o, v, 1.0f);
        } else {
            if (a == 0.f) std::memset(o, 0, sizeof(float) * dv);
            else for (int i = 0; i < dv; ++i) o[i] *= a;
            for (int i = 0; i < dv; ++i) o[i] += v[i];
        }
        m = score;
    } else {
        float diff = score - m;
        if (diff < EXP_PRUNE) return;  // weight ≈ 0
        float w = std::exp(diff);
        l += w;
        if (dv == 128) axpy_d128(o, v, w);
        else for (int i = 0; i < dv; ++i) o[i] += w * v[i];
    }
}

}  // namespace

torch::Tensor attend_forward(
    torch::Tensor q,                 // (h_q, D) fp32
    torch::Tensor centers,           // (S, H_kv, K_cap, max_w) fp32
    torch::Tensor assigns,           // (S, H_kv, N_pad) int32
    torch::Tensor keys,              // (H_kv, N_pad, D) fp32
    torch::Tensor values,            // (H_kv, N_pad, Dv) fp32
    torch::Tensor invalid_mask,      // (H_kv, N_pad) uint8
    torch::Tensor dim_offsets,       // (S,) int32
    torch::Tensor dim_widths,        // (S,) int32
    torch::Tensor threshold,         // (h_q,) fp32, raw dot threshold
    torch::Tensor q_head_to_kv,
    torch::Tensor buffer_keys,       // (H_kv, B_max, D) fp32
    torch::Tensor buffer_values,     // (H_kv, B_max, Dv) fp32
    int64_t k_used,
    int64_t n_used,
    int64_t l_buf,
    double scale_d) {
    TORCH_CHECK(q.is_contiguous() && q.dtype() == torch::kFloat32);
    TORCH_CHECK(centers.dtype() == torch::kFloat32);
    TORCH_CHECK(assigns.dtype() == torch::kInt32);
    TORCH_CHECK(keys.dtype() == torch::kFloat32);
    TORCH_CHECK(values.dtype() == torch::kFloat32);
    TORCH_CHECK(threshold.dtype() == torch::kFloat32);

    const int64_t h_q = q.size(0);
    const int64_t D = q.size(1);
    const int64_t S = centers.size(0);
    const int64_t H_kv = centers.size(1);
    const int64_t K_cap = centers.size(2);
    const int64_t MW = centers.size(3);
    const int64_t N_pad = keys.size(1);
    const int64_t Dv = values.size(2);
    const int64_t B_max = buffer_keys.size(1);
    TORCH_CHECK(S == S_FIXED);

    const float scale = (float)scale_d;
    const auto q2kv = resolve_q_head_to_kv(q_head_to_kv, h_q, H_kv);

    auto out = torch::empty({h_q, Dv}, q.options());
    const int64_t K_eff = k_used;
    const int64_t N_eff = n_used;

    const float* q_ptr        = q.data_ptr<float>();
    const float* centers_ptr  = centers.data_ptr<float>();
    const int32_t* assigns_ptr = assigns.data_ptr<int32_t>();
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
    const int64_t assigns_stride_s = H_kv * N_pad;
    const int64_t assigns_stride_h = N_pad;
    const int64_t keys_stride_h = N_pad * D;
    const int64_t values_stride_h = N_pad * Dv;
    const int64_t buf_k_stride_h = B_max * D;
    const int64_t buf_v_stride_h = B_max * Dv;
    const int64_t inv_stride_h = N_pad;

    const int64_t bm_words = (K_cap + 63) / 64;

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        std::vector<float>    scores(K_eff > 0 ? (size_t)K_eff : 1, 0.f);
        std::vector<int32_t>  idxs(K_eff > 0 ? (size_t)K_eff : 1, 0);
        std::vector<float>    sorted_top((size_t)(L_MAX * S_FIXED), 0.f);
        std::vector<int32_t>  top_idx((size_t)(L_MAX * S_FIXED), 0);
        std::vector<uint64_t> bm((size_t)(bm_words * S_FIXED), 0ULL);
        std::vector<float>    o((size_t)Dv, 0.f);

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
                        scores[(size_t)k] = dot_w32(qs, base + k * centers_stride_k);
                        idxs[(size_t)k] = (int32_t)k;
                    }
                } else {
                    for (int64_t k = 0; k < K_eff; ++k) {
                        scores[(size_t)k] = dot_w(qs, base + k * centers_stride_k, w);
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
                if (row_sum < th) { depth = t + 1; break; }
            }

            std::memset(bm.data(), 0, sizeof(uint64_t) * (size_t)(bm_words * S_FIXED));
            for (int s = 0; s < S_FIXED; ++s) {
                int L = std::min(depth, L_force);
                uint64_t* bms = bm.data() + s * bm_words;
                int32_t* tidx = top_idx.data() + s * L_MAX;
                for (int t = 0; t < L; ++t) {
                    int p = tidx[t];
                    bms[p >> 6] |= (1ULL << (p & 63));
                }
            }

            std::memset(o.data(), 0, sizeof(float) * (size_t)Dv);
            float m = -std::numeric_limits<float>::infinity();
            float l = 0.f;

            const int32_t* a0 = assigns_ptr + 0 * assigns_stride_s + hkv * assigns_stride_h;
            const int32_t* a1 = assigns_ptr + 1 * assigns_stride_s + hkv * assigns_stride_h;
            const int32_t* a2 = assigns_ptr + 2 * assigns_stride_s + hkv * assigns_stride_h;
            const int32_t* a3 = assigns_ptr + 3 * assigns_stride_s + hkv * assigns_stride_h;
            const uint64_t* bm0 = bm.data() + 0 * bm_words;
            const uint64_t* bm1 = bm.data() + 1 * bm_words;
            const uint64_t* bm2 = bm.data() + 2 * bm_words;
            const uint64_t* bm3 = bm.data() + 3 * bm_words;
            const uint8_t* inv_row = inv_ptr + hkv * inv_stride_h;
            const float* keys_row = keys_ptr + hkv * keys_stride_h;
            const float* vals_row = values_ptr + hkv * values_stride_h;

            for (int64_t k = 0; k < N_eff; ++k) {
                if (inv_row[k]) continue;
                int p0 = a0[k];
                bool alive = ((bm0[p0 >> 6] >> (p0 & 63)) & 1ULL) != 0;
                if (!alive) {
                    int p1 = a1[k];
                    alive = ((bm1[p1 >> 6] >> (p1 & 63)) & 1ULL) != 0;
                }
                if (!alive) {
                    int p2 = a2[k];
                    alive = ((bm2[p2 >> 6] >> (p2 & 63)) & 1ULL) != 0;
                }
                if (!alive) {
                    int p3 = a3[k];
                    alive = ((bm3[p3 >> 6] >> (p3 & 63)) & 1ULL) != 0;
                }
                if (!alive) continue;
                const float* kr = keys_row + k * D;
                __builtin_prefetch(keys_row + (k + 4) * D);
                float dot = (D == 128) ? dot_d128(qrow, kr)
                                       : dot_generic(qrow, kr, (int)D);
                float sc = dot * scale;
                update_online_softmax(m, l, o.data(), sc,
                                      vals_row + k * Dv, (int)Dv);
            }

            const float* bk = buf_k_ptr + hkv * buf_k_stride_h;
            const float* bv = buf_v_ptr + hkv * buf_v_stride_h;
            for (int64_t k = 0; k < l_buf; ++k) {
                const float* kr = bk + k * D;
                float dot = (D == 128) ? dot_d128(qrow, kr)
                                       : dot_generic(qrow, kr, (int)D);
                float sc = dot * scale;
                update_online_softmax(m, l, o.data(), sc,
                                      bv + k * Dv, (int)Dv);
            }

            float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            float* orow = out_ptr + hq * Dv;
            for (int64_t i = 0; i < Dv; ++i) orow[i] = o[(size_t)i] * inv_l;
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_forward,
          py::arg("q"), py::arg("centers"), py::arg("assigns"),
          py::arg("keys"), py::arg("values"), py::arg("invalid_mask"),
          py::arg("dim_offsets"), py::arg("dim_widths"),
          py::arg("threshold"), py::arg("q_head_to_kv"),
          py::arg("buffer_keys"), py::arg("buffer_values"),
          py::arg("k_used"), py::arg("n_used"), py::arg("l_buf"),
          py::arg("scale"));
}
