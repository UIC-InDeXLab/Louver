// attention v3.3 — Based on v2.2 GQA-tiled approach with optimizations:
//   1. Fused q_norms + parent pass (single OMP pass)
//   2. Parent-block s=0 early exit in tiled scoring (skip entire bf block)
//   3. Larger tiles for better amortization
//   4. Explicit thread count control
#include "_attention_v2_exact_common.h"

namespace {
constexpr int64_t V3_PARENTS_PER_TILE = 64;  // larger tiles
}

torch::Tensor attend_v3_3(
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
    TORCH_CHECK(max_groups <= V2_MAX_GROUPS, "v3.3 max GQA group size ", V2_MAX_GROUPS);

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

    // --- Phase 3: GQA-tiled scoring with parent-block early exit ---
    const int64_t n_tiles = (p.k_scan + V3_PARENTS_PER_TILE - 1) / V3_PARENTS_PER_TILE;
    const int64_t total_tiles_kv = p.h_kv * n_tiles;
    const int64_t total_tiles_q = p.h_q * n_tiles;
    std::unique_ptr<float[]> tile_m(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_l(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_o(new float[static_cast<size_t>(total_tiles_q * p.d_v)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles_kv; ++task) {
        const int64_t kvh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        const int64_t G = static_cast<int64_t>(qh_list.size());
        if (G == 0) {
            // Initialize tile outputs for all qh in this group to empty
            continue;
        }

        float m_run[V2_MAX_GROUPS];
        float l_run[V2_MAX_GROUPS];
        alignas(64) float o_run[V2_MAX_GROUPS * V2_D_MAX];
        for (int64_t g = 0; g < G; ++g) {
            m_run[g] = -std::numeric_limits<float>::infinity();
            l_run[g] = 0.0f;
        }

        const int64_t parent_lo = tile * V3_PARENTS_PER_TILE;
        const int64_t parent_hi = std::min(parent_lo + V3_PARENTS_PER_TILE, p.k_scan);

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            // Early exit: check if ANY group member passes s=0 for this parent
            // If none do, skip entire block of bf children
            uint32_t s0_any = 0;
            for (int64_t g = 0; g < G; ++g) {
                const int64_t qh = qh_list[static_cast<size_t>(g)];
                if (pass.get()[(qh * p.s_count + 0) * p.k_total + parent]) {
                    s0_any |= (1u << g);
                }
            }
            if (s0_any == 0) continue;  // Skip entire parent block

            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                // For each group member, check full AND across all subspaces
                uint32_t group_pass = 0;
                for (int64_t g = 0; g < G; ++g) {
                    if (!((s0_any >> g) & 1u)) continue;  // Already failed s=0
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    bool child_passes = true;
                    // s=0 already passed, check s=1..S-1
                    for (int64_t s = 1; s < p.s_count; ++s) {
                        const int64_t p_idx = static_cast<int64_t>(
                            assign_ptrs[static_cast<size_t>(s)][kvh * p.n_pad + j]);
                        if (p_idx < 0 || p_idx >= p.k_total ||
                            pass.get()[(qh * p.s_count + s) * p.k_total + p_idx] == 0) {
                            child_passes = false;
                            break;
                        }
                    }
                    if (child_passes) group_pass |= (1u << g);
                }
                if (group_pass == 0) continue;

                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                for (int64_t g = 0; g < G; ++g) {
                    if (!((group_pass >> g) & 1u)) continue;
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                        static_cast<int>(p.d))
                                        * static_cast<float>(scale);
                    v2_online_update(score, val, m_run[g], l_run[g],
                                     o_run + g * V2_D_MAX, p.d_v);
                }
            }
        }

        for (int64_t g = 0; g < G; ++g) {
            const int64_t qh = qh_list[static_cast<size_t>(g)];
            const int64_t flat = qh * n_tiles + tile;
            tile_m[flat] = m_run[g];
            tile_l[flat] = l_run[g];
            if (l_run[g] > 0.0f) {
                std::copy(o_run + g * V2_D_MAX, o_run + g * V2_D_MAX + p.d_v,
                          tile_o.get() + flat * p.d_v);
            }
        }
    }

    // --- Phase 4: Merge tiles + buffer ---
    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        alignas(64) float o_acc[V2_D_MAX];

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t task = qh * n_tiles + tile;
            v2_merge_online(m_acc, l_acc, o_acc,
                            tile_m[task], tile_l[task],
                            tile_o.get() + task * p.d_v, p.d_v);
        }

        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score = v2_dot_dispatch(qp + qh * p.d, key,
                                                static_cast<int>(p.d))
                                * static_cast<float>(scale);
            const float* val = bvp + (kvh * p.n_buf + j) * p.d_v;
            v2_online_update(score, val, m_acc, l_acc, o_acc, p.d_v);
        }

        if (l_acc > 0.0f) {
            const float inv = 1.0f / l_acc;
            for (int64_t x = 0; x < p.d_v; ++x) outp[qh * p.d_v + x] = o_acc[x] * inv;
        } else {
            std::fill(outp + qh * p.d_v, outp + (qh + 1) * p.d_v, 0.0f);
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v3_3,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
