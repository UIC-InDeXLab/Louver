// attention v1.1 — online single-pass softmax + (h_q × parent-block) parallelism.
//
// v1.0 ran two passes (max-pass then sum-pass) with full per-subspace gates in
// both, and parallelized only over H_q (8-24 work units, leaving the
// 32-thread default pool half-idle on real captures).
//
// v1.1 changes:
//   - One pass per row: maintain (m_i, l_i, o_i) and rescale on new max
//     (FlashAttention-style online softmax).
//   - Splits work along parents: PARENTS_PER_TILE consecutive parent blocks
//     per task; outer parallel over (h_q × n_tiles) so we can saturate ≥32
//     threads even with H_q=8.
//   - Per-thread partial (m, l, o) merged with the standard online-softmax
//     reduction at the end. Buffer rows are folded into the final reduction
//     after the per-tile loop closes.
//
// Inner dot products are still scalar; v1.2 will hand-vectorize for D=128.
#include "_cpu_common.h"

namespace py = pybind11;
using namespace hira_cpu;

namespace {

constexpr int64_t PARENTS_PER_TILE = 32;  // 32 parents × bf=4 = 128 rows / tile

inline void merge_online(
    float& m_dst, float& l_dst, float* o_dst,
    float m_src, float l_src, const float* o_src,
    int64_t d_v) {
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
    for (int64_t x = 0; x < d_v; ++x) o_dst[x] = o_dst[x] * a + o_src[x] * b;
    m_dst = new_m;
}

}  // namespace

torch::Tensor attend_v1_1(
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

    std::vector<const float*> center_ptrs(centers.size()), radius_ptrs(radii.size());
    std::vector<const int32_t*> assign_ptrs(assigns.size());
    std::vector<int64_t> sub_starts(s_count), sub_widths(s_count);
    for (int64_t s = 0; s < s_count; ++s) {
        center_ptrs[s] = centers[s].data_ptr<float>();
        radius_ptrs[s] = radii[s].data_ptr<float>();
        assign_ptrs[s] = assigns[s].data_ptr<int32_t>();
        sub_starts[s] = slices[s].first;
        sub_widths[s] = slices[s].second - slices[s].first;
    }

    const int64_t n_tiles = (k_used + PARENTS_PER_TILE - 1) / PARENTS_PER_TILE;
    const int64_t total_tiles = h_q * n_tiles;

    // Per-(h_q, tile) partial running m/l/o. Storing flat for cheap aggregation.
    std::vector<float> tile_m(static_cast<size_t>(total_tiles),
                              -std::numeric_limits<float>::infinity());
    std::vector<float> tile_l(static_cast<size_t>(total_tiles), 0.0f);
    std::vector<float> tile_o(static_cast<size_t>(total_tiles * d_v), 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles; ++task) {
        const int64_t qh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const int64_t kvh = q2kv[qh];

        // Per-subspace q-norms for the bound check.
        float q_norms[64];
        for (int64_t s = 0; s < s_count; ++s) {
            const int64_t st = sub_starts[s];
            const int64_t w = sub_widths[s];
            float acc = 0.0f;
            for (int64_t x = 0; x < w; ++x) {
                const float v = qp[qh * d + st + x];
                acc += v * v;
            }
            q_norms[s] = std::sqrt(std::max(acc, 0.0f));
        }

        const int64_t parent_lo = tile * PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + PARENTS_PER_TILE, k_used);
        const int64_t row_lo = parent_lo * bf;
        const int64_t row_hi = std::min(parent_hi * bf, n_pad);

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        float* o_run = &tile_o[task * d_v];

        for (int64_t j = row_lo; j < row_hi; ++j) {
            if (invp[kvh * n_pad + j]) continue;

            // Subspace-bound gate (same as v1.0 but the gate-then-score logic
            // is now folded into a single pass).
            bool pass = true;
            for (int64_t s = 0; s < s_count; ++s) {
                const int64_t st = sub_starts[s];
                const int64_t w = sub_widths[s];
                const int64_t parent = assign_ptrs[s][kvh * n_pad + j];
                const float* cp = center_ptrs[s] + (kvh * k_used + parent) * w;
                float bound = 0.0f;
                for (int64_t x = 0; x < w; ++x) bound += qp[qh * d + st + x] * cp[x];
                bound += q_norms[s] * radius_ptrs[s][kvh * k_used + parent];
                if (bound < thp[s * h_q + qh]) { pass = false; break; }
            }
            if (!pass) continue;

            float score = 0.0f;
            const float* key = krp + (kvh * n_pad + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);

            const float* val = vrp + (kvh * n_pad + j) * d_v;
            if (score > m_run) {
                if (l_run == 0.0f) {
                    m_run = score; l_run = 1.0f;
                    std::copy(val, val + d_v, o_run);
                } else {
                    const float a = std::exp(m_run - score);
                    l_run = l_run * a + 1.0f;
                    for (int64_t x = 0; x < d_v; ++x) o_run[x] = o_run[x] * a + val[x];
                    m_run = score;
                }
            } else {
                const float w = std::exp(score - m_run);
                l_run += w;
                for (int64_t x = 0; x < d_v; ++x) o_run[x] += w * val[x];
            }
        }

        tile_m[task] = m_run;
        tile_l[task] = l_run;
    }

    // Reduce tiles per query head, then fold buffer.
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

        // Buffer (always scanned, no gate). Done sequentially per-head; cheap.
        for (int64_t j = 0; j < n_buf; ++j) {
            float score = 0.0f;
            const float* key = bkp + (kvh * n_buf + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);
            const float* val = bvp + (kvh * n_buf + j) * d_v;
            if (score > m_acc) {
                if (l_acc == 0.0f) {
                    m_acc = score; l_acc = 1.0f;
                    std::copy(val, val + d_v, o_acc.data());
                } else {
                    const float a = std::exp(m_acc - score);
                    l_acc = l_acc * a + 1.0f;
                    for (int64_t x = 0; x < d_v; ++x) o_acc[x] = o_acc[x] * a + val[x];
                    m_acc = score;
                }
            } else {
                const float w = std::exp(score - m_acc);
                l_acc += w;
                for (int64_t x = 0; x < d_v; ++x) o_acc[x] += w * val[x];
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
    m.def("attend", &attend_v1_1,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
