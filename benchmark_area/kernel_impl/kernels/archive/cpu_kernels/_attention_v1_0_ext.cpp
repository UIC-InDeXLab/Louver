// attention v1.0: scalar reference; parallelizes only over H_q. Two-pass
// softmax (max-pass then sum-pass), bound check on every subspace. This is the
// correctness baseline — clean and trivially auditable; v1.1+ are tuned.
#include "_cpu_common.h"

namespace py = pybind11;
using namespace hira_cpu;

torch::Tensor attend_v1_0(
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
    const int64_t n_scan = std::min<int64_t>(n_pad, k_used * bf);

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
    for (int64_t s = 0; s < s_count; ++s) {
        center_ptrs[s] = centers[s].data_ptr<float>();
        radius_ptrs[s] = radii[s].data_ptr<float>();
        assign_ptrs[s] = assigns[s].data_ptr<int32_t>();
    }

    #pragma omp parallel for schedule(dynamic)
    for (int64_t qh = 0; qh < h_q; ++qh) {
        const int64_t kvh = q2kv[qh];
        TORCH_CHECK(kvh >= 0 && kvh < h_kv, "q_head_to_kv entry out of range");
        std::vector<float> q_norms(s_count, 0.0f);
        for (int64_t s = 0; s < s_count; ++s) {
            const auto [start, end] = slices[s];
            float acc = 0.0f;
            for (int64_t x = start; x < end; ++x) acc += qp[qh * d + x] * qp[qh * d + x];
            q_norms[s] = std::sqrt(std::max(acc, 0.0f));
        }

        float max_score = -std::numeric_limits<float>::infinity();
        int64_t kept = 0;
        for (int64_t j = 0; j < n_scan; ++j) {
            if (invp[kvh * n_pad + j]) continue;
            bool pass = true;
            for (int64_t s = 0; s < s_count; ++s) {
                const auto [start, end] = slices[s];
                const int64_t width = end - start;
                const int64_t parent = assign_ptrs[s][kvh * n_pad + j];
                const float* cp = center_ptrs[s] + (kvh * k_used + parent) * width;
                float bound = 0.0f;
                for (int64_t x = 0; x < width; ++x) bound += qp[qh * d + start + x] * cp[x];
                bound += q_norms[s] * radius_ptrs[s][kvh * k_used + parent];
                if (bound < thp[s * h_q + qh]) { pass = false; break; }
            }
            if (!pass) continue;
            float score = 0.0f;
            const float* key = krp + (kvh * n_pad + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);
            max_score = std::max(max_score, score);
            ++kept;
        }
        for (int64_t j = 0; j < n_buf; ++j) {
            float score = 0.0f;
            const float* key = bkp + (kvh * n_buf + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);
            max_score = std::max(max_score, score);
            ++kept;
        }
        if (kept == 0) continue;

        float denom = 0.0f;
        for (int64_t j = 0; j < n_scan; ++j) {
            if (invp[kvh * n_pad + j]) continue;
            bool pass = true;
            for (int64_t s = 0; s < s_count; ++s) {
                const auto [start, end] = slices[s];
                const int64_t width = end - start;
                const int64_t parent = assign_ptrs[s][kvh * n_pad + j];
                const float* cp = center_ptrs[s] + (kvh * k_used + parent) * width;
                float bound = 0.0f;
                for (int64_t x = 0; x < width; ++x) bound += qp[qh * d + start + x] * cp[x];
                bound += q_norms[s] * radius_ptrs[s][kvh * k_used + parent];
                if (bound < thp[s * h_q + qh]) { pass = false; break; }
            }
            if (!pass) continue;
            float score = 0.0f;
            const float* key = krp + (kvh * n_pad + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);
            const float w = std::exp(score - max_score);
            denom += w;
            const float* val = vrp + (kvh * n_pad + j) * d_v;
            for (int64_t x = 0; x < d_v; ++x) outp[qh * d_v + x] += w * val[x];
        }
        for (int64_t j = 0; j < n_buf; ++j) {
            float score = 0.0f;
            const float* key = bkp + (kvh * n_buf + j) * d;
            for (int64_t x = 0; x < d; ++x) score += qp[qh * d + x] * key[x];
            score *= static_cast<float>(scale);
            const float w = std::exp(score - max_score);
            denom += w;
            const float* val = bvp + (kvh * n_buf + j) * d_v;
            for (int64_t x = 0; x < d_v; ++x) outp[qh * d_v + x] += w * val[x];
        }
        const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int64_t x = 0; x < d_v; ++x) outp[qh * d_v + x] *= inv_denom;
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v1_0,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
