"""search_v18.2 — v18.1 graph/workspace path with full-fp16 state."""

from __future__ import annotations

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_fp16_utils import (
    _get_layout_v15_fp16,
    _next_pow2,
    buffer_dot_to_dtype,
)
from ._search_triton import triton_fused_cluster_pass_out
from .search_v15_0 import _PARENTS_PER_PROG, _fp16_fused_kernel

KERNEL_VERSION = "v18.2"


def _pack_q_into(q: torch.Tensor, layout: dict, out: torch.Tensor) -> None:
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    if d == s * max_d:
        out.copy_(q.view(h_q, s, max_d).transpose(0, 1))
        return

    out.zero_()
    for si, (s0, e0) in enumerate(layout["dim_slices"]):
        out[si, :, : e0 - s0].copy_(q[:, s0:e0])


def _make_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor) -> dict:
    s = layout["num_subspaces"]
    h_q = q.shape[0]
    max_d = layout["max_d"]
    k = layout["K"]
    n_pad = layout["N_pad"]
    device = q.device
    q_dtype = layout["centers"].dtype

    return {
        "static_q": torch.empty(h_q, q.shape[1], device=device, dtype=q_dtype),
        "static_th": torch.empty(s, h_q, device=device, dtype=q_dtype),
        "q_packed": torch.empty(s, h_q, max_d, device=device, dtype=q_dtype),
        "q_norm": torch.empty(s, h_q, device=device, dtype=q_dtype),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "out": torch.empty(h_q, n_pad, device=device, dtype=q_dtype),
        "graph": None,
        "capture_failed": False,
    }


def _prepare_inputs(work: dict, layout: dict) -> None:
    _pack_q_into(work["static_q"], layout, work["q_packed"])
    torch.linalg.vector_norm(work["q_packed"], dim=-1, out=work["q_norm"])


def _launch_kernels(work: dict, layout: dict, h_q: int, d: int) -> None:
    triton_fused_cluster_pass_out(
        q_packed=work["q_packed"],
        q_norm=work["q_norm"],
        th=work["static_th"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=layout["groups"],
        out=work["cluster_pass"],
    )

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))

    _fp16_fused_kernel[grid](
        work["static_q"],
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        work["cluster_pass"],
        layout["invalid_blocks_i8"],
        work["out"],
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=layout["num_subspaces"],
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
        num_warps=4,
    )


def _graph_cache_key(layout: dict, q: torch.Tensor, th: torch.Tensor) -> tuple:
    return (
        q.device.index,
        q.dtype,
        tuple(q.shape),
        th.dtype,
        tuple(th.shape),
        layout["keys_blocks_t_f16"].data_ptr(),
        layout["centers"].data_ptr(),
        layout["radii"].data_ptr(),
        layout["assigns_blocks"].data_ptr(),
        layout["invalid_blocks_i8"].data_ptr(),
    )


def _get_workspace(
    state: dict,
    layout: dict,
    q: torch.Tensor,
    th: torch.Tensor,
) -> dict:
    cache = state.setdefault("_search_v18_2_cache", {})
    key = _graph_cache_key(layout, q, th)
    work = cache.get("work")
    if cache.get("key") == key and work is not None:
        return work

    work = _make_workspace(layout, q, th)
    cache["key"] = key
    cache["work"] = work
    return work


def _maybe_capture_graph(state: dict, layout: dict, work: dict, h_q: int, d: int) -> None:
    if work["graph"] is not None or work["capture_failed"]:
        return
    if not bool(state.get("_search_v18_2_use_cuda_graphs", True)):
        work["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_kernels(work, layout, h_q, d)
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_kernels(work, layout, h_q, d)
        work["graph"] = graph
    except Exception:
        work["capture_failed"] = True


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v18_2 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v18_2 requires build_v2-style state")

    layout = _get_layout_v15_fp16(state, q_head_to_kv, q, "_search_v18_2_layout")
    q_dtype = layout["centers"].dtype
    q_cast = q if q.dtype == q_dtype else q.to(q_dtype)
    th_view = th_per_subspace.reshape(layout["num_subspaces"], q.shape[0])
    if th_view.dtype != q_dtype:
        th_view = th_view.to(q_dtype)
    th_view = th_view.contiguous()

    h_q = q_cast.shape[0]
    d = q_cast.shape[1]
    work = _get_workspace(state, layout, q_cast, th_view)

    work["static_q"].copy_(q_cast)
    work["static_th"].copy_(th_view)
    _prepare_inputs(work, layout)

    _maybe_capture_graph(state, layout, work, h_q, d)
    if work["graph"] is not None:
        work["graph"].replay()
    else:
        _launch_kernels(work, layout, h_q, d)

    out = work["out"]
    if bool(state.get("_search_v18_2_clone_output", False)):
        out = out.clone()

    buf_shim = {
        "mode": layout["mode"],
        "groups": layout["groups"],
        "base_heads": layout["base_heads"],
    }
    buf_dots = buffer_dot_to_dtype(q_cast, buffer_keys, q_head_to_kv, buf_shim, out.dtype)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
