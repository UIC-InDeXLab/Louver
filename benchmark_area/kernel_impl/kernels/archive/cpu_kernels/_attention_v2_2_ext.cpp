// attention v2.2 — exact full-AND gate with GQA-aware tiled scoring.
#include "_attention_v2_exact_common.h"

torch::Tensor attend_v2_2(
    torch::Tensor q_in, torch::Tensor th_in, py::dict state,
    py::object buffer_keys_obj, py::object buffer_values_obj,
    py::object q_head_to_kv_obj, double scale) {
    return attend_v2_exact_gqa_tiled(
        q_in, th_in, state, buffer_keys_obj, buffer_values_obj,
        q_head_to_kv_obj, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attend", &attend_v2_2,
          py::arg("q"), py::arg("th_per_subspace"), py::arg("state"),
          py::arg("buffer_keys") = py::none(), py::arg("buffer_values") = py::none(),
          py::arg("q_head_to_kv") = py::none(), py::arg("scale") = 1.0);
}
