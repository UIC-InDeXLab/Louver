// attend_v3 — v2 (parent-walk) + bf16 storage + AVX-512 BF16 (vdpbf16ps).
//
// Cuts centroid + inner-sweep dots roughly 2× by storing centers/keys/values
// as bf16 and using vdpbf16ps. q is converted to bf16 once per call.
//
// Specialised for S=4, bf=4, D=Dv=128, w=32.

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
    float score, const uint16_t* __restrict v_bf16) {
    // axpy from bf16 src to fp32 dst with weight w.
    const __m512 vw = _mm512_set1_ps(score > m ? 1.0f : 0.f);  // placeholder, replaced below
    (void)vw;
    if (score > m) {
        float diff = m - score;
        float a = (diff > EXP_PRUNE) ? std::exp(diff) : 0.f;
        l = l * a + 1.0f;
        if (a == 0.f) std::memset(o, 0, sizeof(float) * 128);
        else scale_d128(o, a);
        // Add v_bf16 with weight 1.0
        const __m512 wv = _mm512_set1_ps(1.0f);
        for (int i = 0; i < 128; i += 32) {
            __m256i bh = _mm256_loadu_si256((const __m256i*)(v_bf16 + i));
            __m512i ext = _mm512_cvtepu16_epi32(bh);
            __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(ext, 16));
            _mm512_storeu_ps(o + i, _mm512_fmadd_ps(v, wv, _mm512_loadu_ps(o + i)));
            __m256i bh2 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i + 16));
            __m512i ext2 = _mm512_cvtepu16_epi32(bh2);
            __m512 v2 = _mm512_castsi512_ps(_mm512_slli_epi32(ext2, 16));
            _mm512_storeu_ps(o + i + 16, _mm512_fmadd_ps(v2, wv, _mm512_loadu_ps(o + i + 16)));
        }
        m = score;
    } else {
        float diff = score - m;
        if (diff < EXP_PRUNE) return;
        float wf = std::exp(diff);
        l += wf;
        const __m512 wv = _mm512_set1_ps(wf);
        for (int i = 0; i < 128; i += 32) {
            __m256i bh = _mm256_loadu_si256((const __m256i*)(v_bf16 + i));
            __m512i ext = _mm512_cvtepu16_epi32(bh);
            __m512 v = _mm512_castsi512_ps(_mm512_slli_epi32(ext, 16));
            _mm512_storeu_ps(o + i, _mm512_fmadd_ps(v, wv, _mm512_loadu_ps(o + i)));
            __m256i bh2 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i + 16));
            __m512i ext2 = _mm512_cvtepu16_epi32(bh2);
            __m512 v2 = _mm512_castsi512_ps(_mm512_slli_epi32(ext2, 16));
            _mm512_storeu_ps(o + i + 16, _mm512_fmadd_ps(v2, wv, _mm512_loadu_ps(o + i + 16)));
        }
    }
}

}  // namespace

torch::Tensor attend_v3_forward(
    torch::Tensor q,                  // (h_q, D) fp32
    torch::Tensor centers_bf16,       // (S, H_kv, K_cap, max_w) bf16
    torch::Tensor parent_children,    // (S, H_kv, K_cap, BF) int32
    torch::Tensor parent_counts,      // (S, H_kv, K_cap) int32
    torch::Tensor keys_bf16,          // (H_kv, N_pad, D) bf16
    torch::Tensor values_bf16,        // (H_kv, N_pad, Dv) bf16
    torch::Tensor invalid_mask,
    torch::Tensor dim_offsets,
    torch::Tensor dim_widths,
    torch::Tensor threshold,
    torch::Tensor q_head_to_kv,
    torch::Tensor buffer_keys_bf16,   // (H_kv, B_max, D) bf16
    torch::Tensor buffer_values_bf16, // (H_kv, B_max, Dv) bf16
    int64_t k_used,
    int64_t n_used,
    int64_t l_buf,
    double scale_d) {
    TORCH_CHECK(q.dtype() == torch::kFloat32);
    TORCH_CHECK(centers_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(keys_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(values_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(buffer_keys_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(buffer_values_bf16.dtype() == torch::kBFloat16);

    const int64_t h_q = q.size(0);
    const int64_t D = q.size(1);
    const int64_t H_kv = centers_bf16.size(1);
    const int64_t K_cap = centers_bf16.size(2);
    const int64_t MW = centers_bf16.size(3);
    const int64_t N_pad = keys_bf16.size(1);
    const int64_t Dv = values_bf16.size(2);
    const int64_t B_max = buffer_keys_bf16.size(1);
    TORCH_CHECK(D == 128 && Dv == 128, "v3 specialised for D=Dv=128");
    TORCH_CHECK(MW == 32, "v3 specialised for max_w=32");

    const float scale = (float)scale_d;
    const auto q2kv = resolve_q_head_to_kv(q_head_to_kv, h_q, H_kv);

    // Convert q → bf16 once.
    auto q_bf16 = q.to(torch::kBFloat16).contiguous();
    const uint16_t* q_ptr = (const uint16_t*)q_bf16.data_ptr();

    auto out = torch::empty({h_q, Dv}, q.options());
    const int64_t K_eff = k_used;

    const uint16_t* centers_ptr = (const uint16_t*)centers_bf16.data_ptr();
    const int32_t* pc_ptr     = parent_children.data_ptr<int32_t>();
    const int32_t* pcnt_ptr   = parent_counts.data_ptr<int32_t>();
    const uint16_t* keys_ptr  = (const uint16_t*)keys_bf16.data_ptr();
    const uint16_t* values_ptr = (const uint16_t*)values_bf16.data_ptr();
    const uint8_t* inv_ptr    = invalid_mask.data_ptr<uint8_t>();
    const int32_t* offs_ptr   = dim_offsets.data_ptr<int32_t>();
    const int32_t* w_ptr      = dim_widths.data_ptr<int32_t>();
    const float* th_ptr       = threshold.data_ptr<float>();
    const uint16_t* buf_k_ptr = (const uint16_t*)buffer_keys_bf16.data_ptr();
    const uint16_t* buf_v_ptr = (const uint16_t*)buffer_values_bf16.data_ptr();
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
            const uint16_t* qrow = q_ptr + hq * D;
            const int L_force = (int)std::min<int64_t>(L_MAX, K_eff);

            for (int s = 0; s < S_FIXED; ++s) {
                int off = offs_ptr[s];
                const uint16_t* qs = qrow + off;
                const uint16_t* base = centers_ptr + s * centers_stride_s
                                       + hkv * centers_stride_h;
#if defined(__AVX512BF16__)
                __m512bh va = (__m512bh)_mm512_loadu_si512((const __m512i*)qs);
                for (int64_t k = 0; k < K_eff; ++k) {
                    const uint16_t* c = base + k * centers_stride_k;
                    __m512bh vb = (__m512bh)_mm512_loadu_si512((const __m512i*)c);
                    __m512 acc = _mm512_setzero_ps();
                    acc = _mm512_dpbf16_ps(acc, va, vb);
                    scores[(size_t)k] = reduce_zmm(acc);
                    idxs[(size_t)k] = (int32_t)k;
                }
#else
                for (int64_t k = 0; k < K_eff; ++k) {
                    scores[(size_t)k] = dot_bf16_w32(qs, base + k * centers_stride_k);
                    idxs[(size_t)k] = (int32_t)k;
                }
#endif
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

            const uint16_t* keys_row = keys_ptr + hkv * keys_stride_h;
            const uint16_t* vals_row = values_ptr + hkv * values_stride_h;
            const uint8_t* inv_row = inv_ptr + hkv * N_pad;

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
                        const uint16_t* kr = keys_row + (int64_t)k * D;
                        __builtin_prefetch(kr + D);
#if defined(__AVX512BF16__)
                        float dot = dot_bf16_d128(qrow, kr);
#else
                        float dot = 0.f; (void)kr;
#endif
                        float sc = dot * scale;
                        update_online_softmax_d128(m, l, o, sc, vals_row + (int64_t)k * Dv);
                    }
                }
            }

            const uint16_t* bk = buf_k_ptr + hkv * buf_k_stride_h;
            const uint16_t* bv = buf_v_ptr + hkv * buf_v_stride_h;
            for (int64_t k = 0; k < l_buf; ++k) {
                const uint16_t* kr = bk + k * D;
#if defined(__AVX512BF16__)
                float dot = dot_bf16_d128(qrow, kr);
#else
                float dot = 0.f; (void)kr;
#endif
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
    m.def("attend", &attend_v3_forward,
          py::arg("q"), py::arg("centers_bf16"),
          py::arg("parent_children"), py::arg("parent_counts"),
          py::arg("keys_bf16"), py::arg("values_bf16"),
          py::arg("invalid_mask"),
          py::arg("dim_offsets"), py::arg("dim_widths"),
          py::arg("threshold"), py::arg("q_head_to_kv"),
          py::arg("buffer_keys_bf16"), py::arg("buffer_values_bf16"),
          py::arg("k_used"), py::arg("n_used"), py::arg("l_buf"),
          py::arg("scale"));
}
