// attention v3.0 — Optimized AND-gated attention with:
//   1. Fused q_norms + parent pass computation
//   2. Precomputed parent-level AND bitmask (skip entire blocks early)
//   3. Better parallelization: per-qh parallel with early parent skipping
//   4. Thread-local accumulators on stack
//   5. All cores utilized
#include "_attention_v2_exact_common.h"

torch::Tensor attend_v3_0(
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
    int64_t max_groups = 0;
    for (const auto& v : qh_per_kv) max_groups = std::max<int64_t>(max_groups, v.size());

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

    // --- Phase 1+2 fused: compute q_norms and parent pass tables ---
    // Layout: pass[qh * s_count * k_total + s * k_total + parent]
    const int64_t total_pass = p.h_q * p.s_count * p.k_total;
    std::unique_ptr<uint8_t[]> pass(new uint8_t[static_cast<size_t>(total_pass)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;

        // Fused: compute q_norm for this subspace
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

    // --- Phase 2.5: Precompute parent-level AND bitmask per (kvh, parent) ---
    // For each GQA group member qh within kvh, check if parent passes ALL subspaces.
    // parent_mask[kvh * k_total + parent] = bitmask of which GQA group members pass.
    // This avoids redundant per-child AND checks.
    std::unique_ptr<uint32_t[]> parent_mask(
        new uint32_t[static_cast<size_t>(p.h_kv * p.k_total)]);
    std::fill(parent_mask.get(), parent_mask.get() + p.h_kv * p.k_total, 0u);

    #pragma omp parallel for schedule(static)
    for (int64_t kvh = 0; kvh < p.h_kv; ++kvh) {
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        const int64_t G = static_cast<int64_t>(qh_list.size());
        for (int64_t parent = 0; parent < p.k_total; ++parent) {
            uint32_t mask = 0;
            for (int64_t g = 0; g < G; ++g) {
                const int64_t qh = qh_list[static_cast<size_t>(g)];
                bool all_pass = true;
                for (int64_t s = 0; s < p.s_count; ++s) {
                    // For s=0 (anchor), parent for child j is j/bf. Since we're
                    // iterating parents directly, the relevant parent IS `parent`.
                    int64_t p_idx = parent;
                    if (s > 0) {
                        // For non-anchor subspaces, the parent depends on the child,
                        // not the parent block. But at parent level, we need to check
                        // if this anchor-parent can possibly pass in subspace s.
                        // We can't pre-compute this without knowing the child's assignment.
                        // So we check if the anchor parent passes in s=0 only at parent level.
                        // For s>0 we need per-child check. Skip precomputation for s>0.
                    }
                    if (pass.get()[(qh * p.s_count + s) * p.k_total + p_idx] == 0) {
                        // For s=0, this is definitive. For s>0, p_idx=parent is wrong
                        // because non-anchor subspaces have different parent assignments.
                        // But for s=0 check, if it fails, the AND fails.
                        if (s == 0) { all_pass = false; break; }
                    }
                }
                // If s=0 passes, we set the bit. We'll do full AND check per child.
                if (all_pass) mask |= (1u << g);
            }
            parent_mask[kvh * p.k_total + parent] = mask;
        }
    }

    // --- Phase 3: Scan children with early parent-block skipping ---
    // Parallelize per query head for best load balance
    #pragma omp parallel for schedule(dynamic)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        // Find this qh's index in the GQA group
        int64_t g_self = 0;
        for (int64_t g = 0; g < static_cast<int64_t>(qh_list.size()); ++g) {
            if (qh_list[static_cast<size_t>(g)] == qh) { g_self = g; break; }
        }

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        alignas(64) float o_run[V2_D_MAX];

        for (int64_t parent = 0; parent < p.k_scan; ++parent) {
            // Early skip: if this qh doesn't pass s=0 for this parent, skip block
            if (!(parent_mask[kvh * p.k_total + parent] & (1u << g_self))) continue;

            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                // Full AND check across all subspaces for this child
                bool passes = true;
                for (int64_t s = 0; s < p.s_count; ++s) {
                    int64_t p_idx;
                    if (s == 0) {
                        p_idx = parent; // anchor: parent = j/bf
                    } else {
                        p_idx = static_cast<int64_t>(
                            assign_ptrs[static_cast<size_t>(s)][kvh * p.n_pad + j]);
                    }
                    if (p_idx < 0 || p_idx >= p.k_total) { passes = false; break; }
                    if (pass.get()[(qh * p.s_count + s) * p.k_total + p_idx] == 0) {
                        passes = false; break;
                    }
                }
                if (!passes) continue;

                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                    static_cast<int>(p.d))
                                    * static_cast<float>(scale);
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
            }
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
    m.def("attend", &attend_v3_0,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
