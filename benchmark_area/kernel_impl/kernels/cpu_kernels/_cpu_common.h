// Shared helpers used by both the index (build/update) extension and per-kernel
// attention extensions. Header-only so each .cpp gets its own copy and the
// per-version files compile and link independently.
#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hira_cpu {

namespace py = pybind11;

inline torch::Tensor as_cpu_float(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.defined(), name, " must be defined");
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    return t.contiguous().to(torch::kFloat32);
}

inline torch::Tensor object_to_tensor_or_empty(py::object obj) {
    if (obj.is_none()) return torch::Tensor();
    return obj.cast<torch::Tensor>();
}

inline std::vector<torch::Tensor> list_float_tensors(py::handle obj, const char* name) {
    py::list list = py::reinterpret_borrow<py::list>(obj);
    std::vector<torch::Tensor> out;
    out.reserve(static_cast<size_t>(py::len(list)));
    for (py::handle item : list) {
        out.push_back(item.cast<torch::Tensor>().contiguous().to(torch::kFloat32));
    }
    TORCH_CHECK(!out.empty(), name, " must not be empty");
    return out;
}

inline std::vector<torch::Tensor> list_int_tensors(py::handle obj, const char* name) {
    py::list list = py::reinterpret_borrow<py::list>(obj);
    std::vector<torch::Tensor> out;
    out.reserve(static_cast<size_t>(py::len(list)));
    for (py::handle item : list) {
        out.push_back(item.cast<torch::Tensor>().contiguous().to(torch::kInt32));
    }
    TORCH_CHECK(!out.empty(), name, " must not be empty");
    return out;
}

inline std::vector<std::pair<int64_t, int64_t>> slices_from_state(py::dict state) {
    py::list list = py::reinterpret_borrow<py::list>(state["dim_slices"]);
    std::vector<std::pair<int64_t, int64_t>> out;
    out.reserve(static_cast<size_t>(py::len(list)));
    for (py::handle item : list) {
        py::tuple tup = py::reinterpret_borrow<py::tuple>(item);
        out.emplace_back(tup[0].cast<int64_t>(), tup[1].cast<int64_t>());
    }
    return out;
}

inline std::vector<int64_t> resolve_q_head_to_kv(
    py::object q_head_to_kv_obj, int64_t h_q, int64_t h_kv) {
    std::vector<int64_t> q2kv(static_cast<size_t>(h_q), 0);
    if (!q_head_to_kv_obj.is_none()) {
        auto map = q_head_to_kv_obj.cast<torch::Tensor>().contiguous().to(torch::kInt64);
        TORCH_CHECK(map.dim() == 1 && map.size(0) == h_q, "q_head_to_kv shape mismatch");
        const int64_t* mp = map.data_ptr<int64_t>();
        for (int64_t h = 0; h < h_q; ++h) q2kv[static_cast<size_t>(h)] = mp[h];
    } else if (h_q == h_kv) {
        for (int64_t h = 0; h < h_q; ++h) q2kv[static_cast<size_t>(h)] = h;
    } else {
        TORCH_CHECK(h_q % h_kv == 0,
                    "H_q must be divisible by H_kv without q_head_to_kv");
        const int64_t groups = h_q / h_kv;
        for (int64_t h = 0; h < h_q; ++h) q2kv[static_cast<size_t>(h)] = h / groups;
    }
    return q2kv;
}

}  // namespace hira_cpu
