// Index extension: builds and incrementally updates the subspace-kcenter
// index. Owns only the structural / state-shaping code; the per-version
// attention kernels live in their own .cpp files.
#include "_cpu_common.h"

namespace py = pybind11;
using namespace hira_cpu;

namespace {

struct BuildState {
    std::vector<std::pair<int64_t, int64_t>> slices;
    std::vector<torch::Tensor> centers;
    std::vector<torch::Tensor> radii;
    std::vector<torch::Tensor> assigns_reord;
    torch::Tensor keys_reord;
    torch::Tensor invalid_mask;
    torch::Tensor reorder_perm;
    torch::Tensor values_reord;
    int64_t k = 0;
    int64_t n = 0;
    int64_t bf = 0;
    int64_t n_pad = 0;
    int64_t orig_anchor = 0;
};

std::vector<std::pair<int64_t, int64_t>> split_contiguous(int64_t d, int64_t n_subspaces) {
    TORCH_CHECK(n_subspaces > 0, "n_subspaces must be positive");
    TORCH_CHECK(d >= n_subspaces, "D must be >= n_subspaces");
    std::vector<std::pair<int64_t, int64_t>> out;
    out.reserve(static_cast<size_t>(n_subspaces));
    const int64_t base = d / n_subspaces;
    const int64_t rem = d % n_subspaces;
    int64_t off = 0;
    for (int64_t s = 0; s < n_subspaces; ++s) {
        const int64_t width = base + (s < rem ? 1 : 0);
        out.emplace_back(off, off + width);
        off += width;
    }
    return out;
}

void kcenter_subspace(
    const torch::Tensor& keys, int64_t start, int64_t end, int64_t k,
    int64_t refine_iter, torch::Tensor& assign_out,
    torch::Tensor& centers_out, torch::Tensor& radii_out) {
    const int64_t h_count = keys.size(0);
    const int64_t n = keys.size(1);
    const int64_t d = keys.size(2);
    const int64_t width = end - start;
    auto opts_f = keys.options().dtype(torch::kFloat32);
    auto opts_i = keys.options().dtype(torch::kInt32);

    assign_out = torch::empty({h_count, n}, opts_i);
    centers_out = torch::empty({h_count, k, width}, opts_f);
    radii_out = torch::zeros({h_count, k}, opts_f);

    const float* kp = keys.data_ptr<float>();
    float* cp = centers_out.data_ptr<float>();
    int32_t* ap = assign_out.data_ptr<int32_t>();
    float* rp = radii_out.data_ptr<float>();

    #pragma omp parallel for schedule(dynamic)
    for (int64_t h = 0; h < h_count; ++h) {
        std::vector<float> min_d(static_cast<size_t>(n),
                                 std::numeric_limits<float>::infinity());
        std::vector<int64_t> center_idx(static_cast<size_t>(k), 0);
        for (int64_t x = 0; x < width; ++x) {
            cp[(h * k + 0) * width + x] = kp[(h * n + 0) * d + start + x];
        }
        for (int64_t c = 1; c < k; ++c) {
            const int64_t prev = center_idx[static_cast<size_t>(c - 1)];
            for (int64_t i = 0; i < n; ++i) {
                float dist = 0.0f;
                for (int64_t x = 0; x < width; ++x) {
                    const float diff = kp[(h * n + i) * d + start + x]
                                     - kp[(h * n + prev) * d + start + x];
                    dist += diff * diff;
                }
                min_d[static_cast<size_t>(i)] =
                    std::min(min_d[static_cast<size_t>(i)], dist);
            }
            int64_t farthest = 0;
            float farthest_d = -1.0f;
            for (int64_t i = 0; i < n; ++i) {
                const float cur = min_d[static_cast<size_t>(i)];
                if (cur > farthest_d) {
                    farthest_d = cur;
                    farthest = i;
                }
            }
            center_idx[static_cast<size_t>(c)] = farthest;
            for (int64_t x = 0; x < width; ++x) {
                cp[(h * k + c) * width + x] =
                    kp[(h * n + farthest) * d + start + x];
            }
        }

        std::vector<float> sums(static_cast<size_t>(k * width), 0.0f);
        std::vector<int64_t> counts(static_cast<size_t>(k), 0);
        for (int64_t iter = 0; iter < refine_iter; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t best = 0;
                float best_d = std::numeric_limits<float>::infinity();
                for (int64_t c = 0; c < k; ++c) {
                    float dist = 0.0f;
                    for (int64_t x = 0; x < width; ++x) {
                        const float diff = kp[(h * n + i) * d + start + x]
                                         - cp[(h * k + c) * width + x];
                        dist += diff * diff;
                    }
                    if (dist < best_d) { best_d = dist; best = c; }
                }
                ++counts[static_cast<size_t>(best)];
                for (int64_t x = 0; x < width; ++x) {
                    sums[static_cast<size_t>(best * width + x)] +=
                        kp[(h * n + i) * d + start + x];
                }
            }
            for (int64_t c = 0; c < k; ++c) {
                if (counts[static_cast<size_t>(c)] == 0) continue;
                const float inv = 1.0f / static_cast<float>(counts[static_cast<size_t>(c)]);
                for (int64_t x = 0; x < width; ++x) {
                    cp[(h * k + c) * width + x] =
                        sums[static_cast<size_t>(c * width + x)] * inv;
                }
            }
        }

        for (int64_t i = 0; i < n; ++i) {
            int64_t best = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < k; ++c) {
                float dist = 0.0f;
                for (int64_t x = 0; x < width; ++x) {
                    const float diff = kp[(h * n + i) * d + start + x]
                                     - cp[(h * k + c) * width + x];
                    dist += diff * diff;
                }
                if (dist < best_d) { best_d = dist; best = c; }
            }
            ap[h * n + i] = static_cast<int32_t>(best);
            float& radius = rp[h * k + best];
            radius = std::max(radius, std::sqrt(std::max(best_d, 0.0f)));
        }
    }
}

std::vector<int64_t> balanced_assign_one_head(
    const float* keys_pad, const float* centers, int64_t h, int64_t n_pad,
    int64_t d, int64_t k, int64_t bf, int64_t start, int64_t width) {
    std::vector<std::vector<std::pair<float, int64_t>>> ranked(static_cast<size_t>(n_pad));
    std::vector<std::pair<float, int64_t>> order;
    order.reserve(static_cast<size_t>(n_pad));
    for (int64_t i = 0; i < n_pad; ++i) {
        auto& r = ranked[static_cast<size_t>(i)];
        r.reserve(static_cast<size_t>(k));
        float best = std::numeric_limits<float>::infinity();
        for (int64_t c = 0; c < k; ++c) {
            float dist = 0.0f;
            const float* kp = keys_pad + (h * n_pad + i) * d + start;
            const float* cp = centers + (h * k + c) * width;
            for (int64_t x = 0; x < width; ++x) {
                const float diff = kp[x] - cp[x];
                dist += diff * diff;
            }
            r.emplace_back(dist, c);
            best = std::min(best, dist);
        }
        std::stable_sort(r.begin(), r.end());
        order.emplace_back(best, i);
    }
    std::stable_sort(order.begin(), order.end());

    std::vector<int64_t> cap(static_cast<size_t>(k), 0);
    std::vector<int64_t> assign(static_cast<size_t>(n_pad), 0);
    for (const auto& item : order) {
        const int64_t i = item.second;
        for (const auto& cand : ranked[static_cast<size_t>(i)]) {
            const int64_t c = cand.second;
            if (cap[static_cast<size_t>(c)] < bf) {
                assign[static_cast<size_t>(i)] = c;
                ++cap[static_cast<size_t>(c)];
                break;
            }
        }
    }
    return assign;
}

py::dict state_to_dict(const BuildState& st) {
    py::dict d;
    py::list slices;
    for (const auto& p : st.slices) slices.append(py::make_tuple(p.first, p.second));
    py::list centers, radii, assigns;
    for (const auto& t : st.centers) centers.append(t);
    for (const auto& t : st.radii) radii.append(t);
    for (const auto& t : st.assigns_reord) assigns.append(t);
    d["dim_slices"] = slices;
    d["centers"] = centers;
    d["radii"] = radii;
    d["assigns_reord"] = assigns;
    d["keys_reord"] = st.keys_reord;
    d["invalid_mask"] = st.invalid_mask;
    d["reorder_perm"] = st.reorder_perm;
    if (st.values_reord.defined()) {
        d["values_reord"] = st.values_reord;
        d["D_v"] = st.values_reord.size(2);
    }
    d["K"] = st.k;
    d["N"] = st.n;
    d["bf"] = st.bf;
    d["N_pad"] = st.n_pad;
    d["anchor_subspace"] = 0;
    d["orig_anchor_subspace"] = st.orig_anchor;
    return d;
}

BuildState build_state(
    torch::Tensor keys_in, int64_t bf, int64_t n_subspaces,
    int64_t refine_iter, int64_t anchor_subspace, torch::Tensor values_in) {
    auto keys = as_cpu_float(keys_in, "keys");
    TORCH_CHECK(keys.dim() == 3, "keys must be (H, N, D)");
    TORCH_CHECK(bf > 0, "bf must be positive");
    TORCH_CHECK(refine_iter >= 0, "refine_iter must be non-negative");

    const int64_t h_count = keys.size(0);
    const int64_t n = keys.size(1);
    const int64_t d = keys.size(2);
    TORCH_CHECK(n > 0, "keys must contain at least one row");
    const int64_t k = std::max<int64_t>(1, (n + bf - 1) / bf);
    const int64_t n_pad = k * bf;
    if (anchor_subspace < 0) anchor_subspace = n_subspaces - 1;
    TORCH_CHECK(anchor_subspace >= 0 && anchor_subspace < n_subspaces,
                "anchor_subspace out of range");

    auto slices_orig = split_contiguous(d, n_subspaces);
    std::vector<torch::Tensor> centers_orig(static_cast<size_t>(n_subspaces));
    std::vector<torch::Tensor> radii_orig(static_cast<size_t>(n_subspaces));
    std::vector<torch::Tensor> assigns_orig(static_cast<size_t>(n_subspaces));
    for (int64_t s = 0; s < n_subspaces; ++s) {
        kcenter_subspace(keys, slices_orig[static_cast<size_t>(s)].first,
                         slices_orig[static_cast<size_t>(s)].second, k, refine_iter,
                         assigns_orig[static_cast<size_t>(s)],
                         centers_orig[static_cast<size_t>(s)],
                         radii_orig[static_cast<size_t>(s)]);
    }

    auto opts_f = keys.options().dtype(torch::kFloat32);
    auto opts_b = keys.options().dtype(torch::kBool);
    auto opts_l = keys.options().dtype(torch::kInt64);
    auto opts_i = keys.options().dtype(torch::kInt32);

    auto keys_pad = torch::zeros({h_count, n_pad, d}, opts_f);
    keys_pad.slice(1, 0, n).copy_(keys);

    const auto [a_start, a_end] = slices_orig[static_cast<size_t>(anchor_subspace)];
    const int64_t a_width = a_end - a_start;
    const float* keys_pad_p = keys_pad.data_ptr<float>();
    const float* centers_anchor_p =
        centers_orig[static_cast<size_t>(anchor_subspace)].data_ptr<float>();

    auto reorder_perm = torch::empty({h_count, n_pad}, opts_l);
    int64_t* perm_p = reorder_perm.data_ptr<int64_t>();
    #pragma omp parallel for schedule(dynamic)
    for (int64_t h = 0; h < h_count; ++h) {
        auto bal = balanced_assign_one_head(
            keys_pad_p, centers_anchor_p, h, n_pad, d, k, bf, a_start, a_width);
        std::vector<int64_t> order(static_cast<size_t>(n_pad));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t lhs, int64_t rhs) {
            return bal[static_cast<size_t>(lhs)] < bal[static_cast<size_t>(rhs)];
        });
        for (int64_t j = 0; j < n_pad; ++j)
            perm_p[h * n_pad + j] = order[static_cast<size_t>(j)];
    }

    auto keys_reord = torch::empty({h_count, n_pad, d}, opts_f);
    auto invalid_mask = torch::empty({h_count, n_pad}, opts_b);
    const float* kp = keys_pad.data_ptr<float>();
    float* krp = keys_reord.data_ptr<float>();
    bool* invp = invalid_mask.data_ptr<bool>();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t h = 0; h < h_count; ++h) {
        for (int64_t j = 0; j < n_pad; ++j) {
            const int64_t src = perm_p[h * n_pad + j];
            invp[h * n_pad + j] = src >= n;
            std::copy(kp + (h * n_pad + src) * d,
                      kp + (h * n_pad + src + 1) * d,
                      krp + (h * n_pad + j) * d);
        }
    }

    auto center_anchor_new = torch::zeros({h_count, k, a_width}, opts_f);
    auto radius_anchor_new = torch::zeros({h_count, k}, opts_f);
    float* canp = center_anchor_new.data_ptr<float>();
    float* ranp = radius_anchor_new.data_ptr<float>();
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t h = 0; h < h_count; ++h) {
        for (int64_t c = 0; c < k; ++c) {
            int64_t cnt = 0;
            for (int64_t child = 0; child < bf; ++child) {
                const int64_t j = c * bf + child;
                if (invp[h * n_pad + j]) continue;
                ++cnt;
                for (int64_t x = 0; x < a_width; ++x) {
                    canp[(h * k + c) * a_width + x] +=
                        krp[(h * n_pad + j) * d + a_start + x];
                }
            }
            if (cnt > 0) {
                const float inv = 1.0f / static_cast<float>(cnt);
                for (int64_t x = 0; x < a_width; ++x)
                    canp[(h * k + c) * a_width + x] *= inv;
            }
            float r = 0.0f;
            for (int64_t child = 0; child < bf; ++child) {
                const int64_t j = c * bf + child;
                if (invp[h * n_pad + j]) continue;
                float dist = 0.0f;
                for (int64_t x = 0; x < a_width; ++x) {
                    const float diff = krp[(h * n_pad + j) * d + a_start + x]
                                     - canp[(h * k + c) * a_width + x];
                    dist += diff * diff;
                }
                r = std::max(r, std::sqrt(std::max(dist, 0.0f)));
            }
            ranp[h * k + c] = r;
        }
    }
    centers_orig[static_cast<size_t>(anchor_subspace)] = center_anchor_new.contiguous();
    radii_orig[static_cast<size_t>(anchor_subspace)] = radius_anchor_new.contiguous();

    std::vector<int64_t> order;
    order.reserve(static_cast<size_t>(n_subspaces));
    order.push_back(anchor_subspace);
    for (int64_t s = 0; s < n_subspaces; ++s)
        if (s != anchor_subspace) order.push_back(s);

    BuildState st;
    st.k = k; st.n = n; st.bf = bf; st.n_pad = n_pad;
    st.orig_anchor = anchor_subspace;
    st.keys_reord = keys_reord.contiguous();
    st.invalid_mask = invalid_mask.contiguous();
    st.reorder_perm = reorder_perm.contiguous();

    const int64_t* perm_const = perm_p;
    for (int64_t idx = 0; idx < n_subspaces; ++idx) {
        const int64_t s = order[static_cast<size_t>(idx)];
        st.slices.push_back(slices_orig[static_cast<size_t>(s)]);
        st.centers.push_back(centers_orig[static_cast<size_t>(s)].contiguous());
        st.radii.push_back(radii_orig[static_cast<size_t>(s)].contiguous());
        auto a_reord = torch::zeros({h_count, n_pad}, opts_i);
        const int32_t* a_orig_p = assigns_orig[static_cast<size_t>(s)].data_ptr<int32_t>();
        int32_t* a_reord_p = a_reord.data_ptr<int32_t>();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t h = 0; h < h_count; ++h) {
            for (int64_t j = 0; j < n_pad; ++j) {
                if (idx == 0) {
                    // The first reordered subspace is the anchor. Its
                    // centers/radii were rebuilt from the contiguous
                    // bf-sized child blocks above, so the assignment must be
                    // the block parent, not the original k-center id.
                    a_reord_p[h * n_pad + j] = static_cast<int32_t>(j / bf);
                } else {
                    const int64_t src = perm_const[h * n_pad + j];
                    a_reord_p[h * n_pad + j] =
                        src < n ? a_orig_p[h * n + src] : int32_t(0);
                }
            }
        }
        st.assigns_reord.push_back(a_reord.contiguous());
    }

    if (values_in.defined()) {
        auto values = as_cpu_float(values_in, "values");
        TORCH_CHECK(values.dim() == 3, "values must be (H, N, D_v)");
        TORCH_CHECK(values.size(0) == h_count && values.size(1) == n,
                    "values must match keys H and N");
        const int64_t d_v = values.size(2);
        auto values_pad = torch::zeros({h_count, n_pad, d_v}, opts_f);
        values_pad.slice(1, 0, n).copy_(values);
        auto values_reord = torch::empty({h_count, n_pad, d_v}, opts_f);
        const float* vp = values_pad.data_ptr<float>();
        float* vrp = values_reord.data_ptr<float>();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int64_t h = 0; h < h_count; ++h) {
            for (int64_t j = 0; j < n_pad; ++j) {
                const int64_t src = perm_const[h * n_pad + j];
                std::copy(vp + (h * n_pad + src) * d_v,
                          vp + (h * n_pad + src + 1) * d_v,
                          vrp + (h * n_pad + j) * d_v);
            }
        }
        values_reord.masked_fill_(invalid_mask.unsqueeze(-1), 0.0);
        st.values_reord = values_reord.contiguous();
    }
    return st;
}

}  // namespace

py::dict build_index(
    torch::Tensor keys, int64_t bf, int64_t n_subspaces, int64_t refine_iter,
    int64_t anchor_subspace, py::object values_obj) {
    auto values = object_to_tensor_or_empty(values_obj);
    return state_to_dict(build_state(keys, bf, n_subspaces, refine_iter,
                                     anchor_subspace, values));
}

py::tuple update_index(
    py::dict state, torch::Tensor old_keys, torch::Tensor buffer_keys,
    int64_t bf, int64_t n_subspaces, int64_t refine_iter,
    py::object old_values_obj, py::object buffer_values_obj,
    int64_t anchor_subspace, bool return_merged) {
    auto buffer_values = object_to_tensor_or_empty(buffer_values_obj);
    BuildState sub = build_state(buffer_keys, bf, n_subspaces, refine_iter,
                                 anchor_subspace, buffer_values);
    py::dict sub_d = state_to_dict(sub);

    const int64_t k_old = state["K"].cast<int64_t>();
    const int64_t n_old = state["N"].cast<int64_t>();
    const int64_t n_pad_old = state["N_pad"].cast<int64_t>();
    const int64_t k_new = k_old + sub.k;
    const int64_t n_new = n_old + sub.n;
    const int64_t n_pad_new = n_pad_old + sub.n_pad;

    py::dict out;
    out["dim_slices"] = state["dim_slices"];
    out["K"] = k_new; out["N"] = n_new; out["bf"] = bf;
    out["N_pad"] = n_pad_new;
    out["anchor_subspace"] = 0;
    out["orig_anchor_subspace"] = anchor_subspace;
    out["keys_reord"] = torch::cat(
        {state["keys_reord"].cast<torch::Tensor>(), sub.keys_reord}, 1).contiguous();
    out["invalid_mask"] = torch::cat(
        {state["invalid_mask"].cast<torch::Tensor>(), sub.invalid_mask}, 1).contiguous();
    out["reorder_perm"] = torch::cat(
        {state["reorder_perm"].cast<torch::Tensor>(), sub.reorder_perm + n_old}, 1).contiguous();

    py::list centers_old = py::reinterpret_borrow<py::list>(state["centers"]);
    py::list radii_old = py::reinterpret_borrow<py::list>(state["radii"]);
    py::list assigns_old = py::reinterpret_borrow<py::list>(state["assigns_reord"]);
    py::list centers_new, radii_new, assigns_new;
    for (int64_t s = 0; s < n_subspaces; ++s) {
        centers_new.append(torch::cat(
            {centers_old[s].cast<torch::Tensor>(), sub.centers[static_cast<size_t>(s)]}, 1).contiguous());
        radii_new.append(torch::cat(
            {radii_old[s].cast<torch::Tensor>(), sub.radii[static_cast<size_t>(s)]}, 1).contiguous());
        assigns_new.append(torch::cat(
            {assigns_old[s].cast<torch::Tensor>(),
             sub.assigns_reord[static_cast<size_t>(s)] + static_cast<int32_t>(k_old)}, 1).contiguous());
    }
    out["centers"] = centers_new;
    out["radii"] = radii_new;
    out["assigns_reord"] = assigns_new;

    if (state.contains("values_reord") && sub.values_reord.defined()) {
        out["values_reord"] = torch::cat(
            {state["values_reord"].cast<torch::Tensor>(), sub.values_reord}, 1).contiguous();
        out["D_v"] = out["values_reord"].cast<torch::Tensor>().size(2);
    }

    py::object new_keys = py::none();
    py::object new_values = py::none();
    if (return_merged) {
        new_keys = py::cast(torch::cat({as_cpu_float(old_keys, "old_keys"),
                                        as_cpu_float(buffer_keys, "buffer_keys")}, 1).contiguous());
        auto old_values = object_to_tensor_or_empty(old_values_obj);
        if (old_values.defined() && buffer_values.defined()) {
            new_values = py::cast(torch::cat({as_cpu_float(old_values, "old_values"),
                                              as_cpu_float(buffer_values, "buffer_values")}, 1).contiguous());
        }
    }
    return py::make_tuple(out, new_keys, new_values, py::none());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_index", &build_index,
          py::arg("keys"), py::arg("bf"), py::arg("n_subspaces"),
          py::arg("refine_iter") = 2, py::arg("anchor_subspace") = -1,
          py::arg("values") = py::none());
    m.def("update_index", &update_index,
          py::arg("state"), py::arg("old_keys"), py::arg("buffer_keys"),
          py::arg("bf"), py::arg("n_subspaces"), py::arg("refine_iter") = 0,
          py::arg("old_values") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("anchor_subspace") = -1, py::arg("return_merged") = false);
}
