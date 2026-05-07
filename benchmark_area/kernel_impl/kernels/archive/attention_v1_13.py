"""attention_v1.13 — packed gates + iterate-all-parents + CUDA graph capture.

Design: v1.9's packed-gate attention kernel (no compaction pass) wrapped in the
v1.8-style CUDA graph runtime. Removes the compaction kernel that costs v1.10
~8 μs per call relative to v1.8.
"""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_fixed_utils import get_layout_attn_rawq, next_pow2, require_fixed_bf_s
from ._attention_triton import NEG_SENT, run_attn_reduce
from ._attention_triton_v1_9 import (
    run_fused_attn_index_packed,
    triton_fused_cluster_pass_packed,
)

KERNEL_VERSION = "v1.13"
_PARENTS_PER_PROG = 8
_DEFAULT_NUM_SPLITS = 32
_GROUPS_POW_FLOOR = 4


def _empty_buffer(buffer_keys: torch.Tensor | None, buffer_values: torch.Tensor | None) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )


def _make_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    h_kv_eff = layout["base_heads"]
    k = layout["K"]
    device = q.device
    return {
        "static_q": torch.empty_like(q),
        "static_th": torch.empty_like(th),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "packed_pass": torch.empty(layout["num_subspaces"], h_kv_eff, k, device=device, dtype=torch.uint8),
        "buf_m": torch.full((h_q,), NEG_SENT, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
        "graph": None,
        "capture_failed": False,
    }


def _launch_no_buffer(
    work: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
) -> None:
    triton_fused_cluster_pass_packed(
        q=work["static_q"],
        th=work["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=work["packed_pass"],
    )

    run_fused_attn_index_packed(
        q=work["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        packed_pass=work["packed_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=layout["num_subspaces"],
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=work["m_idx"],
        out_l=work["l_idx"],
        out_o=work["o_idx"],
    )

    run_attn_reduce(
        work["m_idx"],
        work["l_idx"],
        work["o_idx"],
        work["buf_m"],
        work["buf_l"],
        work["buf_o"],
        work["out"],
    )


def _fixed_cache_key(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> tuple:
    q_map_ptr = 0 if q_head_to_kv is None else q_head_to_kv.data_ptr()
    q_map_shape = () if q_head_to_kv is None else tuple(q_head_to_kv.shape)
    return (
        q.device.index,
        q.dtype,
        tuple(q.shape),
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _maybe_capture_graph(
    state: dict,
    layout: dict,
    work: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
) -> None:
    if work["graph"] is not None or work["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_13_use_cuda_graphs", True)):
        work["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_no_buffer(
                    work, layout, h_q, k, groups, groups_pow, num_splits, anchor_s, scale,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                work, layout, h_q, k, groups, groups_pow, num_splits, anchor_s, scale,
            )
        work["graph"] = graph
    except Exception:
        work["capture_failed"] = True


def _get_fixed_empty_runtime(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> tuple[dict, dict, int, int, int]:
    cache_key = _fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_13_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return (
            fixed["layout"],
            fixed["work"],
            fixed["groups"],
            fixed["groups_pow"],
            fixed["anchor_s"],
        )

    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_13_layout",
    )
    require_fixed_bf_s(layout, bf=4, s=8, groups_max=8)

    groups = layout["groups"]
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    anchor_s = layout["anchor_subspace"]
    work = _make_workspace(layout, q, th, num_splits)

    fixed = {
        "layout": layout,
        "work": work,
        "groups": groups,
        "groups_pow": groups_pow,
        "anchor_s": anchor_s,
    }
    cache["key"] = cache_key
    cache["fixed"] = fixed
    return layout, work, groups, groups_pow, anchor_s


def attend(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    num_splits: int = _DEFAULT_NUM_SPLITS,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("attention_v1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1 requires build_v2-style state")
    if not _empty_buffer(buffer_keys, buffer_values):
        raise RuntimeError("attention_v1_13 is a fixed-shape empty-buffer variant")

    h_q = q.shape[0]
    d = q.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    q_c = q if q.is_contiguous() else q.contiguous()
    if th_per_subspace.shape == (8, h_q) and th_per_subspace.is_contiguous():
        th_view = th_per_subspace
    else:
        th_view = th_per_subspace.reshape(8, h_q).contiguous()

    layout, work, groups, groups_pow, anchor_s = _get_fixed_empty_runtime(
        state, q_c, th_view, q_head_to_kv, num_splits
    )
    k = layout["K"]
    work["static_q"].copy_(q_c)
    work["static_th"].copy_(th_view)
    _maybe_capture_graph(
        state, layout, work, h_q, k, groups, groups_pow, num_splits, anchor_s, scale,
    )
    if work["graph"] is not None:
        work["graph"].replay()
    else:
        _launch_no_buffer(
            work, layout, h_q, k, groups, groups_pow, num_splits, anchor_s, scale,
        )
    return work["out"]


KERNEL = attend
