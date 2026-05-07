// Shared v4 full-AND GQA bitmask attention implementation.
//
// v3.3 keeps a byte pass table for every q-head/subspace/parent and rebuilds
// the GQA-group pass mask repeatedly while scanning children. v4 packs the
// query heads that share a KV head into a uint32 mask for each
// (subspace, kv-head, parent). The child AND check then becomes mask
// intersections across subspaces, while still honoring every subspace gate.
#include "_attention_v2_exact_common.h"

namespace {

#ifndef HIRA_V4_PARENTS_PER_TILE
#define HIRA_V4_PARENTS_PER_TILE 64
#endif

constexpr int64_t V4_PARENTS_PER_TILE = HIRA_V4_PARENTS_PER_TILE;

}  // namespace

#ifndef HIRA_V4_ATTEND_FN
#define HIRA_V4_ATTEND_FN attend_v4_0
#endif

torch::Tensor HIRA_V4_ATTEND_FN(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {

    V2Prepared p = v2_prepare(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj,
        q_head_to_kv_obj);

    std::vector<std::vector<int64_t>> qh_per_kv(static_cast<size_t>(p.h_kv));
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        TORCH_CHECK(kvh >= 0 && kvh < p.h_kv, "q_head_to_kv entry out of range");
        qh_per_kv[static_cast<size_t>(kvh)].push_back(qh);
    }
    int64_t max_groups = 0;
    for (const auto& v : qh_per_kv) {
        max_groups = std::max<int64_t>(max_groups, v.size());
    }
    TORCH_CHECK(max_groups <= V2_MAX_GROUPS, "v4 max GQA group size ", V2_MAX_GROUPS);

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

    std::unique_ptr<float[]> q_norms(
        new float[static_cast<size_t>(p.h_q * p.s_count)]);

    #pragma omp parallel for schedule(static)
    for (int64_t task = 0; task < p.h_q * p.s_count; ++task) {
        const int64_t qh = task / p.s_count;
        const int64_t s = task % p.s_count;
        const auto [start, end] = p.slices[static_cast<size_t>(s)];
        const int64_t width = end - start;
        q_norms[task] = std::sqrt(std::max(
            v2_dot_dispatch(qp + qh * p.d + start, qp + qh * p.d + start,
                            static_cast<int>(width)),
            0.0f));
    }

    // pass_masks[(s * h_kv + kvh) * k_total + parent] contains a bit per
    // query head in qh_per_kv[kvh] that passes this subspace/parent gate.
    std::unique_ptr<uint32_t[]> pass_masks(
        new uint32_t[static_cast<size_t>(p.s_count * p.h_kv * p.k_total)]);

#ifdef HIRA_V4_PARALLEL_PASS_BLOCKS
    const int64_t pass_parent_blocks =
        (p.k_total + V4_PARENTS_PER_TILE - 1) / V4_PARENTS_PER_TILE;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t s = 0; s < p.s_count; ++s) {
        for (int64_t kvh = 0; kvh < p.h_kv; ++kvh) {
            for (int64_t block = 0; block < pass_parent_blocks; ++block) {
                const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
                const int64_t G = static_cast<int64_t>(qh_list.size());
                const auto [start, end] = p.slices[static_cast<size_t>(s)];
                const int64_t width = end - start;
                const float* cp_base =
                    center_ptrs[static_cast<size_t>(s)] + kvh * p.k_total * width;
                const float* rp =
                    radius_ptrs[static_cast<size_t>(s)] + kvh * p.k_total;
                uint32_t* mask_base =
                    pass_masks.get() + (s * p.h_kv + kvh) * p.k_total;
                const int64_t parent_lo = block * V4_PARENTS_PER_TILE;
                const int64_t parent_hi =
                    std::min(parent_lo + V4_PARENTS_PER_TILE, p.k_total);

                for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
                    const float* center = cp_base + parent * width;
                    const float radius = rp[parent];
                    uint32_t mask = 0;
                    for (int64_t g = 0; g < G; ++g) {
                        const int64_t qh = qh_list[static_cast<size_t>(g)];
                        const float dot = v2_dot_dispatch(
                            qp + qh * p.d + start, center, static_cast<int>(width));
                        const float qn = q_norms[qh * p.s_count + s];
                        const float th = thp[s * p.h_q + qh];
                        if (dot + qn * radius >= th) {
                            mask |= (uint32_t{1} << g);
                        }
                    }
                    mask_base[parent] = mask;
                }
            }
        }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t s = 0; s < p.s_count; ++s) {
        for (int64_t kvh = 0; kvh < p.h_kv; ++kvh) {
            const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
            const int64_t G = static_cast<int64_t>(qh_list.size());
            const auto [start, end] = p.slices[static_cast<size_t>(s)];
            const int64_t width = end - start;
            const float* cp_base =
                center_ptrs[static_cast<size_t>(s)] + kvh * p.k_total * width;
            const float* rp =
                radius_ptrs[static_cast<size_t>(s)] + kvh * p.k_total;
            uint32_t* mask_base =
                pass_masks.get() + (s * p.h_kv + kvh) * p.k_total;

            for (int64_t parent = 0; parent < p.k_total; ++parent) {
                const float* center = cp_base + parent * width;
                const float radius = rp[parent];
                uint32_t mask = 0;
                for (int64_t g = 0; g < G; ++g) {
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    const float dot = v2_dot_dispatch(
                        qp + qh * p.d + start, center, static_cast<int>(width));
                    const float qn = q_norms[qh * p.s_count + s];
                    const float th = thp[s * p.h_q + qh];
                    if (dot + qn * radius >= th) {
                        mask |= (uint32_t{1} << g);
                    }
                }
                mask_base[parent] = mask;
            }
        }
    }
#endif

    const int64_t n_tiles =
        (p.k_scan + V4_PARENTS_PER_TILE - 1) / V4_PARENTS_PER_TILE;
    const int64_t total_tiles_kv = p.h_kv * n_tiles;
    const int64_t total_tiles_q = p.h_q * n_tiles;

    std::unique_ptr<float[]> tile_m(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_l(new float[static_cast<size_t>(total_tiles_q)]);
    std::unique_ptr<float[]> tile_o(
        new float[static_cast<size_t>(total_tiles_q * p.d_v)]);

    #pragma omp parallel for schedule(dynamic)
    for (int64_t task = 0; task < total_tiles_kv; ++task) {
        const int64_t kvh = task / n_tiles;
        const int64_t tile = task % n_tiles;
        const auto& qh_list = qh_per_kv[static_cast<size_t>(kvh)];
        const int64_t G = static_cast<int64_t>(qh_list.size());
        if (G == 0) continue;

        float m_run[V2_MAX_GROUPS];
        float l_run[V2_MAX_GROUPS];
        alignas(64) float o_run[V2_MAX_GROUPS * V2_D_MAX];
        for (int64_t g = 0; g < G; ++g) {
            m_run[g] = -std::numeric_limits<float>::infinity();
            l_run[g] = 0.0f;
        }

        const int64_t parent_lo = tile * V4_PARENTS_PER_TILE;
        const int64_t parent_hi =
            std::min(parent_lo + V4_PARENTS_PER_TILE, p.k_scan);
        const uint32_t* s0_masks = pass_masks.get() + kvh * p.k_total;

        for (int64_t parent = parent_lo; parent < parent_hi; ++parent) {
            const uint32_t parent_mask = s0_masks[parent];
            if (parent_mask == 0) continue;

            for (int64_t child = 0; child < p.bf; ++child) {
                const int64_t j = parent * p.bf + child;
                if (j >= p.n_scan) break;
                if (invp[kvh * p.n_pad + j]) continue;

                uint32_t child_mask = parent_mask;
                for (int64_t s = 1; s < p.s_count; ++s) {
                    const int64_t p_idx = static_cast<int64_t>(
                        assign_ptrs[static_cast<size_t>(s)][kvh * p.n_pad + j]);
                    if (p_idx < 0 || p_idx >= p.k_total) {
                        child_mask = 0;
                        break;
                    }
                    const uint32_t m =
                        pass_masks[(s * p.h_kv + kvh) * p.k_total + p_idx];
                    child_mask &= m;
                    if (child_mask == 0) break;
                }
                if (child_mask == 0) continue;

                const float* key = krp + (kvh * p.n_pad + j) * p.d;
                const float* val = vrp + (kvh * p.n_pad + j) * p.d_v;
                uint32_t mask = child_mask;
                while (mask) {
                    const int64_t g = static_cast<int64_t>(__builtin_ctz(mask));
                    mask &= (mask - 1);
                    const int64_t qh = qh_list[static_cast<size_t>(g)];
                    const float score =
                        v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
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
                std::copy(o_run + g * V2_D_MAX,
                          o_run + g * V2_D_MAX + p.d_v,
                          tile_o.get() + flat * p.d_v);
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int64_t qh = 0; qh < p.h_q; ++qh) {
        const int64_t kvh = p.q2kv[static_cast<size_t>(qh)];
        float m_acc = -std::numeric_limits<float>::infinity();
        float l_acc = 0.0f;
        alignas(64) float o_acc[V2_D_MAX];

        for (int64_t tile = 0; tile < n_tiles; ++tile) {
            const int64_t flat = qh * n_tiles + tile;
            v2_merge_online(m_acc, l_acc, o_acc,
                            tile_m[flat], tile_l[flat],
                            tile_o.get() + flat * p.d_v, p.d_v);
        }

        for (int64_t j = 0; j < p.n_buf; ++j) {
            const float* key = bkp + (kvh * p.n_buf + j) * p.d;
            const float score =
                v2_dot_dispatch(qp + qh * p.d, key, static_cast<int>(p.d))
                * static_cast<float>(scale);
            const float* val = bvp + (kvh * p.n_buf + j) * p.d_v;
            v2_online_update(score, val, m_acc, l_acc, o_acc, p.d_v);
        }

        if (l_acc > 0.0f) {
            const float inv = 1.0f / l_acc;
            for (int64_t x = 0; x < p.d_v; ++x) {
                outp[qh * p.d_v + x] = o_acc[x] * inv;
            }
        } else {
            std::fill(outp + qh * p.d_v, outp + (qh + 1) * p.d_v, 0.0f);
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &HIRA_V4_ATTEND_FN,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(),
          py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(),
          py::arg("scale") = 1.0);
}
