// attention v3.2 — GQA-fused compact-survivor scoring:
//   1. Fused q_norms + parent pass
//   2. Per kvh: iterate parent blocks, build per-child GQA bitmask of survivors,
//      share key/value reads across GQA group members
//   3. Online softmax per-qh accumulator
//   4. Parent-block s=0 early exit (any group member)
#include "_attention_v2_exact_common.h"

torch::Tensor attend_v3_2(
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
    TORCH_CHECK(max_groups <= V2_MAX_GROUPS, "v3.2 max GQA group size ", V2_MAX_GROUPS);

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

    // --- Phase 1+2 fused ---
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

    // --- Phase 3: GQA-fused per-kvh processing ---
    // Per-qh accumulators
    std::unique_ptr<float[]> all_m(new float[static_cast<size_t>(p.h_q)]);
    std::unique_ptr<float[]> all_l(new float[static_cast<size_t>(p.h_q)]);
    std::unique_ptr<float[]> all_o(new float[static_cast<size_t>(p.h_q * p.d_v)]);
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        all_m[qh] = -std::numeric_limits<float>::infinity();
        all_l[qh] = 0.0f;
    }

    // Parallelize over kvh. Each kvh processes all its GQA group members.
    // With only 4 KV heads but 64 threads, this doesn't use all cores.
    // So we'll parallelize over qh instead but share child pass info.
    
    // Actually let's parallelize per qh with survivor list approach
    #pragma omp parallel for schedule(dynamic)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        const uint8_t* pass_qh = pass.get() + qh * p.s_count * p.k_total;

        float m_run = -std::numeric_limits<float>::infinity();
        float l_run = 0.0f;
        alignas(64) float o_run[V2_D_MAX];

        for (int64_t parent = 0; parent < p.k_scan; ++parent) {
            // s=0 anchor check at parent level
            if (pass_qh[parent] == 0) continue;  // s=0 stored at offset 0

            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                // Check remaining subspaces
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
                if (!passes) continue;

                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                    static_cast<int>(p.d))
                                    * static_cast<float>(scale);
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                v2_online_update(score, val, m_run, l_run, o_run, p.d_v);
            }
        }

        // Buffer
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
    m.def("attend", &attend_v3_2,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
