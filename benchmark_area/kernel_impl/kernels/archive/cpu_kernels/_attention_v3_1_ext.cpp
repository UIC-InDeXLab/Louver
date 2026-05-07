// attention v3.1 — Compact-survivors approach:
//   1. Fused q_norms + parent pass computation
//   2. Per (kvh): build compact survivor list per qh (AND filter all subspaces)
//      with parent-block s=0 early exit
//   3. Score only survivors — tight sequential loop, good cache behavior
//   4. GQA: group qh by kvh, share key reads across group
#include "_attention_v2_exact_common.h"

#include <algorithm>

torch::Tensor attend_v3_1(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {

    V2Prepared p = v2_prepare(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj, q_head_to_kv_obj);

    // GQA grouping
    std::vector<std::vector<int64_t>> qh_per_kv(static_cast<size_t>(p.h_kv));
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        qh_per_kv[static_cast<size_t>(p.q2kv[static_cast<size_t>(qh)])].push_back(qh);
    }

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

    // --- Phase 1+2 fused: compute parent pass tables ---
    const int64_t total_pass = p.h_q * p.s_count * p.k_total;
    std::unique_ptr<uint8_t[]> pass(new uint8_t[static_cast<size_t>(total_pass)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;

        const float qn = std::sqrt(std::max(
            v2_dot_dispatch(qp + qh * p.d + start, qp + qh * p.d + start,
                            static_cast<int>(width)), 0.0f));
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

    // --- Phase 3: Per-qh, build survivor list, then score ---
    #pragma omp parallel for schedule(dynamic)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const uint8_t* pass_qh = pass.get() + qh * p.s_count * p.k_total;

        // Build compact survivor list with parent-block early exit
        // Thread-local survivor buffer (worst case: all children survive)
        std::vector<int32_t> survivors;
        survivors.reserve(static_cast<size_t>(p.n_scan / 4)); // expect ~25% survival

        for (int64_t parent = 0; parent < p.k_scan; ++parent) {
            // Check s=0 (anchor) first — if fails, skip entire block of bf children
            if (pass_qh[0 * p.k_total + parent] == 0) continue;

            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                // Check remaining subspaces (s=1..S-1)
                bool passes = true;
                for (int64_t s = 1; s < p.s_count; ++s) {
                    const int64_t p_idx = static_cast<int64_t>(
                        assign_ptrs[static_cast<size_t>(s)][kvh * p.n_pad + j]);
                    if (p_idx < 0 || p_idx >= p.k_total ||
                        pass_qh[s * p.k_total + p_idx] == 0) {
                        passes = false;
                        break;
                    }
                }
                if (passes) survivors.push_back(static_cast<int32_t>(j));
            }
        }

        // Score only survivors — tight loop, good cache behavior
        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        alignas(64) float o_run[V2_D_MAX];

        for (const int32_t j : survivors) {
            const float* key = krp + (kvh * p.n_pad + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
            v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
        }

        // Buffer scoring
        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                static_cast<int>(p.d))
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v3_1,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
