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

// Packed (score, idx) — 8 bytes — for branch-free sort comparator inlining.
struct ScoreIdx {
    float   score;
    int32_t idx;
    bool operator>(const ScoreIdx& o) const { return score > o.score; }
};

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
    torch::Tensor centers_f32,        // (S, H_kv, K_cap, max_w) fp32 — KEPT FP32 for alive-set parity
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
    TORCH_CHECK(centers_f32.dtype() == torch::kFloat32);
    TORCH_CHECK(keys_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(values_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(buffer_keys_bf16.dtype() == torch::kBFloat16);
    TORCH_CHECK(buffer_values_bf16.dtype() == torch::kBFloat16);

    const int64_t h_q = q.size(0);
    const int64_t D = q.size(1);
    const int64_t H_kv = centers_f32.size(1);
    const int64_t K_cap = centers_f32.size(2);
    const int64_t MW = centers_f32.size(3);
    const int64_t N_pad = keys_bf16.size(1);
    const int64_t Dv = values_bf16.size(2);
    const int64_t B_max = buffer_keys_bf16.size(1);
    TORCH_CHECK(D == 128 && Dv == 128, "v3 specialised for D=Dv=128");
    TORCH_CHECK(MW == 32, "v3 specialised for max_w=32");

    const float scale = (float)scale_d;
    const auto q2kv = resolve_q_head_to_kv(q_head_to_kv, h_q, H_kv);

    // Convert q → bf16 once for the inner-sweep sparse SDPA hot loop.
    auto q_bf16 = q.to(torch::kBFloat16).contiguous();
    const uint16_t* q_ptr_bf16 = (const uint16_t*)q_bf16.data_ptr();
    const float* q_ptr_f32 = q.data_ptr<float>();

    auto out = torch::empty({h_q, Dv}, q.options());
    const int64_t K_eff = k_used;

    const float* centers_ptr  = centers_f32.data_ptr<float>();
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

    // Persistent per-thread scratch arena (sized once per process).
    // `static` so the heap allocation happens on first call only — subsequent
    // calls reuse the same buffers. Avoids per-call vector allocation cost.
#ifdef _OPENMP
    int max_threads = omp_get_max_threads();
#else
    int max_threads = 1;
#endif
    static std::vector<ScoreIdx> g_si;       // packed (score, idx) sort buf
    static std::vector<float>   g_sorted_top;
    static std::vector<int32_t> g_top_idx;
    static std::vector<uint8_t> g_visited;
    static std::vector<int32_t> g_alive_buf;
    static std::vector<float>   g_alive_scores;
    static int64_t g_K_cap = 0, g_N_cap = 0, g_threads = 0;
    if (K_eff > g_K_cap || max_threads > g_threads) {
        g_K_cap = std::max(K_eff, g_K_cap);
        g_threads = std::max(max_threads, (int)g_threads);
        g_si.resize((size_t)(g_threads * g_K_cap));
        g_sorted_top.resize((size_t)(g_threads * L_MAX * S_FIXED));
        g_top_idx.resize((size_t)(g_threads * L_MAX * S_FIXED));
    }
    if (N_pad > g_N_cap || max_threads > g_threads) {
        g_N_cap = std::max(N_pad, g_N_cap);
        g_visited.resize((size_t)(g_threads * g_N_cap));
        g_alive_buf.resize((size_t)(g_threads * g_N_cap));
        g_alive_scores.resize((size_t)(g_threads * g_N_cap));
    }
    ScoreIdx* g_si_ptr         = g_si.data();
    float* g_sorted_top_ptr    = g_sorted_top.data();
    int32_t* g_top_idx_ptr     = g_top_idx.data();
    uint8_t* g_visited_ptr     = g_visited.data();
    int32_t* g_alive_buf_ptr   = g_alive_buf.data();
    float* g_alive_scores_ptr  = g_alive_scores.data();
    const int64_t scratch_K = g_K_cap;
    const int64_t scratch_N = g_N_cap;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t hq = 0; hq < h_q; ++hq) {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        ScoreIdx* si        = g_si_ptr + tid * scratch_K;
        float* sorted_top   = g_sorted_top_ptr + tid * (L_MAX * S_FIXED);
        int32_t* top_idx    = g_top_idx_ptr + tid * (L_MAX * S_FIXED);
        uint8_t* visited    = g_visited_ptr + tid * scratch_N;
        int32_t* alive_buf_g    = g_alive_buf_ptr + tid * scratch_N;
        float*   alive_scores_g = g_alive_scores_ptr + tid * scratch_N;
        float o[128];
        {
            const int64_t hkv = q2kv[(size_t)hq];
            const float th = th_ptr[hq];
            const uint16_t* qrow_bf16 = q_ptr_bf16 + hq * D;
            const float* qrow_f32 = q_ptr_f32 + hq * D;
            const int L_force = (int)std::min<int64_t>(L_MAX, K_eff);

            // Centroid scoring: full fp32. Centers are fp32-stored.
            // Bandwidth = K_eff * 32 fp32 = small, fits in L1/L2 per (hkv, s).
            // Keeping fp32 here preserves alive-set parity with v1/v2.
            for (int s = 0; s < S_FIXED; ++s) {
                int off = offs_ptr[s];
                const float* qs = qrow_f32 + off;
                const float* base = centers_ptr + s * centers_stride_s
                                    + hkv * centers_stride_h;
                for (int64_t k = 0; k < K_eff; ++k) {
                    const float* c = base + k * centers_stride_k;
                    __m512 lo = _mm512_mul_ps(_mm512_loadu_ps(qs),
                                              _mm512_loadu_ps(c));
                    __m512 hi = _mm512_fmadd_ps(_mm512_loadu_ps(qs + 16),
                                                _mm512_loadu_ps(c + 16), lo);
                    si[k].score = reduce_zmm(hi);
                    si[k].idx   = (int32_t)k;
                }
                int L = L_force;
                if (K_eff > L) {
                    std::partial_sort(si, si + L, si + (size_t)K_eff,
                                      std::greater<ScoreIdx>{});
                } else {
                    std::sort(si, si + (size_t)K_eff, std::greater<ScoreIdx>{});
                }
                float* row = sorted_top + s * L_MAX;
                int32_t* tidx = top_idx + s * L_MAX;
                for (int t = 0; t < L; ++t) {
                    row[t] = si[t].score;
                    tidx[t] = si[t].idx;
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

            std::memset(visited, 0, (size_t)N_pad);

            const uint16_t* keys_row = keys_ptr + hkv * keys_stride_h;
            const uint16_t* vals_row = values_ptr + hkv * values_stride_h;
            const uint8_t* inv_row = inv_ptr + hkv * N_pad;

            // Collect unique alive keys via parent-walk (no scoring yet).
            int32_t* alive_buf = alive_buf_g;
            int n_alive = 0;
            for (int s = 0; s < S_FIXED; ++s) {
                const int32_t* tidx = top_idx + s * L_MAX;
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
                        alive_buf[n_alive++] = k;
                    }
                }
            }

            // Score alive keys (dedicated scratch sized to N_pad).
            float* alive_scores = alive_scores_g;
            for (int t = 0; t < n_alive; ++t) {
                int32_t k = alive_buf[t];
                const uint16_t* kr = keys_row + (int64_t)k * D;
                if (t + 4 < n_alive) {
                    __builtin_prefetch(keys_row + (int64_t)alive_buf[t + 4] * D);
                }
                alive_scores[t] = dot_bf16_d128(qrow_bf16, kr) * scale;
            }

            // Buffer scoring (also batched).
            const uint16_t* bk = buf_k_ptr + hkv * buf_k_stride_h;
            const uint16_t* bv = buf_v_ptr + hkv * buf_v_stride_h;
            float bscores[256];
            for (int64_t k = 0; k < l_buf; ++k) {
                if (k + 1 < l_buf) __builtin_prefetch(bk + (k + 1) * D);
                bscores[k] = dot_bf16_d128(qrow_bf16, bk + k * D) * scale;
            }

            // Combined max over alive scores ∪ buffer scores.
            float maxv = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < n_alive; ++t) if (alive_scores[t] > maxv) maxv = alive_scores[t];
            for (int64_t k = 0; k < l_buf;  ++k) if (bscores[k]      > maxv) maxv = bscores[k];

            // exp() + sum (single pass) + axpy. Initialise o = 0 first.
            std::memset(o, 0, sizeof(float) * 128);
            float sum_w = 0.f;
            for (int t = 0; t < n_alive; ++t) {
                float w = std::exp(alive_scores[t] - maxv);
                alive_scores[t] = w;
                sum_w += w;
            }
            for (int64_t k = 0; k < l_buf; ++k) {
                float w = std::exp(bscores[k] - maxv);
                bscores[k] = w;
                sum_w += w;
            }
            // axpy: o += w[i] * v[i] (bf16 src, fp32 dst).
            for (int t = 0; t < n_alive; ++t) {
                float w = alive_scores[t];
                if (w == 0.f) continue;
                int32_t k = alive_buf[t];
                const uint16_t* v_bf16 = vals_row + (int64_t)k * Dv;
                const __m512 wv = _mm512_set1_ps(w);
                for (int i = 0; i < 128; i += 32) {
                    __m256i bh1 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i));
                    __m512  v1  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bh1), 16));
                    _mm512_storeu_ps(o + i, _mm512_fmadd_ps(v1, wv, _mm512_loadu_ps(o + i)));
                    __m256i bh2 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i + 16));
                    __m512  v2  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bh2), 16));
                    _mm512_storeu_ps(o + i + 16, _mm512_fmadd_ps(v2, wv, _mm512_loadu_ps(o + i + 16)));
                }
            }
            for (int64_t k = 0; k < l_buf; ++k) {
                float w = bscores[k];
                if (w == 0.f) continue;
                const uint16_t* v_bf16 = bv + k * Dv;
                const __m512 wv = _mm512_set1_ps(w);
                for (int i = 0; i < 128; i += 32) {
                    __m256i bh1 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i));
                    __m512  v1  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bh1), 16));
                    _mm512_storeu_ps(o + i, _mm512_fmadd_ps(v1, wv, _mm512_loadu_ps(o + i)));
                    __m256i bh2 = _mm256_loadu_si256((const __m256i*)(v_bf16 + i + 16));
                    __m512  v2  = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bh2), 16));
                    _mm512_storeu_ps(o + i + 16, _mm512_fmadd_ps(v2, wv, _mm512_loadu_ps(o + i + 16)));
                }
            }
            float l = sum_w;

            float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            float* orow = out_ptr + hq * Dv;
            for (int i = 0; i < 128; ++i) orow[i] = o[i] * inv_l;
        }
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v3_forward,
          py::arg("q"), py::arg("centers_f32"),
          py::arg("parent_children"), py::arg("parent_counts"),
          py::arg("keys_bf16"), py::arg("values_bf16"),
          py::arg("invalid_mask"),
          py::arg("dim_offsets"), py::arg("dim_widths"),
          py::arg("threshold"), py::arg("q_head_to_kv"),
          py::arg("buffer_keys_bf16"), py::arg("buffer_values_bf16"),
          py::arg("k_used"), py::arg("n_used"), py::arg("l_buf"),
          py::arg("scale"));
}
