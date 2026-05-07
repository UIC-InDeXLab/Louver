"""search_v18.1 — v15 pipeline with persistent workspace and CUDA Graphs.

This version keeps the same math as v15. The optimization target is launch
and allocation overhead:
  - reuse persistent buffers for q_packed / q_norm / cluster_pass / out
  - optionally capture the steady-state Triton kernel sequence in a CUDA Graph

For benchmark purposes the returned output aliases persistent workspace by
default. Set ``state["_search_v18_1_clone_output"] = True`` if the caller
needs a stable tensor after subsequent searches.

=> search_v18_1 returns a tensor backed by persistent workspace by default for speed. 
If you need a stable output tensor across later calls, set state["_search_v18_1_clone_output"] = True.

"""

from __future__ import annotations

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from .._search_triton import triton_fused_cluster_pass_out
from .._search_utils import buffer_dot
from .search_v15_0 import (
    _PARENTS_PER_PROG,
    _fp16_fused_kernel,
    _get_layout_v15,
)

KERNEL_VERSION = "v18.1"


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


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

    return {
        "static_q": torch.empty_like(q),
        "static_th": torch.empty(s, h_q, device=device, dtype=th.dtype),
        "q_packed": torch.empty(s, h_q, max_d, device=device, dtype=q.dtype),
        "q_norm": torch.empty(s, h_q, device=device, dtype=q.dtype),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "out": torch.empty(h_q, n_pad, device=device, dtype=torch.float32),
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
    cache = state.setdefault("_search_v18_1_cache", {})
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
    if not bool(state.get("_search_v18_1_use_cuda_graphs", True)):
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
        raise RuntimeError("search_v18_1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v18_1 requires build_v2-style state")

    layout = _get_layout_v15(state, q_head_to_kv, q)
    h_q = q.shape[0]
    d = q.shape[1]
    th_view = th_per_subspace.reshape(layout["num_subspaces"], h_q)
    work = _get_workspace(state, layout, q, th_view)

    work["static_q"].copy_(q)
    work["static_th"].copy_(th_view)
    _prepare_inputs(work, layout)

    _maybe_capture_graph(state, layout, work, h_q, d)
    if work["graph"] is not None:
        work["graph"].replay()
    else:
        _launch_kernels(work, layout, h_q, d)

    out = work["out"]
    if bool(state.get("_search_v18_1_clone_output", False)):
        out = out.clone()

    buf_shim = {
        "mode": layout["mode"],
        "groups": layout["groups"],
        "base_heads": layout["base_heads"],
    }
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
