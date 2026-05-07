"""attention_v1.15_s16 — v1.15 specialized for S=16 (BF left runtime-variable).

Same compute path as v1.15 (PARENTS_PER_PROG=16, num_stages=3 on the v1.8
non-packed index kernel + CUDA-graph capture). The underlying Triton kernel
is already generic over S via `tl.static_range(0, S)`; only the Python-side
guard and threshold reshape assume a specific S. This file hardcodes S=16.
"""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_fixed_utils import get_layout_attn_rawq, next_pow2
from ._attention_triton import NEG_SENT, run_attn_reduce
from ._attention_triton_v1_5 import triton_fused_cluster_pass_rawq
from ._attention_triton_v1_14 import run_fused_attn_index_ns

KERNEL_VERSION = "v1.15_s16"
_S_FIXED = 16
_DEFAULT_NUM_SPLITS = 32
_GROUPS_POW_FLOOR = 4
_NUM_STAGES = 3
_NUM_WARPS = 4
_GROUPS_MAX = 8
# Target ~64 cols per chunk (= PARENTS_PER_PROG * BF). v1.15 used 16 at BF=4;
# scale down proportionally so BF=16 stays within shared-memory budget with
# num_stages=3.
_TARGET_COLS_PER_CHUNK = 64


def _parents_per_prog_for_bf(bf: int) -> int:
    return max(1, _TARGET_COLS_PER_CHUNK // max(bf, 1))


def _empty_buffer(buffer_keys: torch.Tensor | None, buffer_values: torch.Tensor | None) -> bool:
    return (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )


def _require_s16(layout: dict) -> None:
    if int(layout["num_subspaces"]) != _S_FIXED:
        raise RuntimeError(
            f"attention_v1_15_s16 requires S={_S_FIXED}; got S={layout['num_subspaces']}"
        )
    if int(layout["groups"]) > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_15_s16 requires groups <= {_GROUPS_MAX}; got {layout['groups']}"
        )


def _make_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    k = layout["K"]
    device = q.device
    return {
        "static_q": torch.empty_like(q),
        "static_th": torch.empty_like(th),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
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
    d: int,
    d_v: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
) -> None:
    triton_fused_cluster_pass_rawq(
        q=work["static_q"],
        th=work["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=work["cluster_pass"],
    )

    run_fused_attn_index_ns(
        q=work["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=work["cluster_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=_S_FIXED,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=work["m_idx"],
        out_l=work["l_idx"],
        out_o=work["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
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
    d: int,
    d_v: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
) -> None:
    if work["graph"] is not None or work["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_15_s16_use_cuda_graphs", True)):
        work["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_no_buffer(
                    work, layout, h_q, d, d_v, k, groups, groups_pow, num_splits, anchor_s, scale, parents_per_prog,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                work, layout, h_q, d, d_v, k, groups, groups_pow, num_splits, anchor_s, scale, parents_per_prog,
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
    cache = state.setdefault("_attn_v1_15_s16_fixed", {})
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
        cache_name="_attn_v1_15_s16_layout",
    )
    _require_s16(layout)

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
        raise RuntimeError("attention_v1_15_s16 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_15_s16 requires build_v2-style state")
    if not _empty_buffer(buffer_keys, buffer_values):
        raise RuntimeError("attention_v1_15_s16 is a fixed-shape empty-buffer variant")

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()
    if th_per_subspace.shape == (_S_FIXED, h_q) and th_per_subspace.is_contiguous():
        th_view = th_per_subspace
    else:
        th_view = th_per_subspace.reshape(_S_FIXED, h_q).contiguous()

    layout, work, groups, groups_pow, anchor_s = _get_fixed_empty_runtime(
        state, q_c, th_view, q_head_to_kv, num_splits
    )
    d_v = layout["D_v"]
    k = layout["K"]

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    work["static_q"].copy_(q_c)
    work["static_th"].copy_(th_view)

    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]))
    _maybe_capture_graph(
        state, layout, work, h_q, d, d_v, k, groups, groups_pow, num_splits, anchor_s, scale, parents_per_prog,
    )
    if work["graph"] is not None:
        work["graph"].replay()
    else:
        _launch_no_buffer(
            work, layout, h_q, d, d_v, k, groups, groups_pow, num_splits, anchor_s, scale, parents_per_prog,
        )
    return work["out"]


KERNEL = attend
