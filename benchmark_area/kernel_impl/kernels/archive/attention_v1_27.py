"""attention_v1.27 — raw-q on-demand gating without a cluster_pass tensor.

Idea under test:
  - remove the separate cluster_pass launch
  - load the 8 fixed 16-wide q subspaces once per program
  - keep the rest of the v1.18 pipeline unchanged so the change stays focused

This experiment is intentionally fixed to the hot benchmark shape:
  bf=4, S=8, D=128, anchor_subspace=0, contiguous 16-wide subspaces.
"""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_buffer_triton import run_buffer_attn
from ._attention_fixed_utils import (
    get_layout_attn_rawq,
    next_pow2,
    require_fixed_bf_s,
)
from ._attention_triton import NEG_SENT, run_attn_reduce
from ._attention_triton_v1_27 import run_fused_attn_index_ondemand_rawq_fixed
from .attention_v1_18 import (
    _ALLOWED_S,
    _BUCKETS,
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _bucket_for,
    _copy_buffer_into_stage_incremental,
    _empty_buffer,
    _make_bucket_staging,
    _prepare_buffer_effective,
)
from .attention_v1_17 import _parents_per_prog_for_bf

KERNEL_VERSION = "v1.27"

_GROUPS_POW_FLOOR = 4
_NUM_STAGES = 3
_NUM_WARPS = 4


def _require_supported(layout: dict) -> None:
    require_fixed_bf_s(layout, bf=4, s=8, groups_max=8)
    if int(layout["anchor_subspace"]) != 0:
        raise RuntimeError(
            f"attention_v1_27 requires anchor_subspace=0; got {layout['anchor_subspace']}"
        )
    if tuple(layout["dim_slices"]) != (
        (0, 16),
        (16, 32),
        (32, 48),
        (48, 64),
        (64, 80),
        (80, 96),
        (96, 112),
        (112, 128),
    ):
        raise RuntimeError(
            "attention_v1_27 requires contiguous 16-wide dim_slices for S=8"
        )


def _make_shared_workspace(
    layout: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    num_splits: int,
) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    device = q.device
    return {
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float32),
        "static_th": torch.empty_like(th),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "buf_m": torch.full((h_q,), NEG_SENT, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _launch_no_buffer(
    shared: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    scale: float,
    parents_per_prog: int,
) -> None:
    run_fused_attn_index_ondemand_rawq_fixed(
        q=shared["static_q"],
        th=shared["static_th"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers=layout["centers"],
        radii=layout["radii"],
        assigns_blocks=layout["assigns_blocks"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    run_attn_reduce(
        shared["m_idx"],
        shared["l_idx"],
        shared["o_idx"],
        shared["buf_m"],
        shared["buf_l"],
        shared["buf_o"],
        shared["out"],
    )


def _launch_with_buffer(
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    scale: float,
    parents_per_prog: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    run_fused_attn_index_ondemand_rawq_fixed(
        q=shared["static_q"],
        th=shared["static_th"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers=layout["centers"],
        radii=layout["radii"],
        assigns_blocks=layout["assigns_blocks"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    run_buffer_attn(
        q=shared["static_q"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_kv_eff=layout["base_heads"],
        groups=groups,
        groups_pow=groups_pow,
        l_buf_max=bucket,
        buf_cols_per_prog=buf_cols,
        scale=scale,
        out_m=shared["buf_m"],
        out_l=shared["buf_l"],
        out_o=shared["buf_o"],
        num_warps=buf_warps,
        num_stages=buf_stages,
    )

    run_attn_reduce(
        shared["m_idx"],
        shared["l_idx"],
        shared["o_idx"],
        shared["buf_m"],
        shared["buf_l"],
        shared["buf_o"],
        shared["out"],
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
        tuple(q.shape),
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _capture_no_buffer_graph(
    state: dict,
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    scale: float,
    parents_per_prog: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_27_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_no_buffer(
                    shared,
                    layout,
                    h_q,
                    k,
                    groups,
                    groups_pow,
                    num_splits,
                    scale,
                    parents_per_prog,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                shared,
                layout,
                h_q,
                k,
                groups,
                groups_pow,
                num_splits,
                scale,
                parents_per_prog,
            )
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def _capture_with_buffer_graph(
    state: dict,
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    scale: float,
    parents_per_prog: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_27_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_with_buffer(
                    shared,
                    stage,
                    layout,
                    h_q,
                    k,
                    groups,
                    groups_pow,
                    num_splits,
                    scale,
                    parents_per_prog,
                    bucket,
                    buf_cols,
                    buf_warps,
                    buf_stages,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_with_buffer(
                shared,
                stage,
                layout,
                h_q,
                k,
                groups,
                groups_pow,
                num_splits,
                scale,
                parents_per_prog,
                bucket,
                buf_cols,
                buf_warps,
                buf_stages,
            )
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def _get_fixed_runtime(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> dict:
    cache_key = _fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_27_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_27_layout",
    )
    _require_supported(layout)

    groups = int(layout["groups"])
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
    shared = _make_shared_workspace(layout, q, th, num_splits)

    fixed = {
        "layout": layout,
        "shared": shared,
        "groups": groups,
        "groups_pow": groups_pow,
        "parents_per_prog": parents_per_prog,
        "buckets": {},
        "no_buffer_stage": None,
    }
    cache["key"] = cache_key
    cache["fixed"] = fixed
    return fixed


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
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError("attention_v1_27 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_27 requires build_v2-style state")

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    fixed_probe = state.get("_attn_v1_27_fixed", {}).get("fixed")
    s_hint = int(_ALLOWED_S[0]) if fixed_probe is None else 8
    if th_per_subspace.shape == (s_hint, h_q) and th_per_subspace.is_contiguous():
        th_view = th_per_subspace
    else:
        th_view = th_per_subspace.reshape(s_hint, h_q).contiguous()

    fixed = _get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th"].copy_(th_view)

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_no_buffer_graph(
            state,
            shared,
            stage,
            layout,
            h_q,
            k,
            groups,
            groups_pow,
            num_splits,
            scale,
            parents_per_prog,
        )
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(
                shared,
                layout,
                h_q,
                k,
                groups,
                groups_pow,
                num_splits,
                scale,
                parents_per_prog,
            )
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)

    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys,
        buffer_values,
        layout,
        q_head_to_kv,
    )
    _copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update((state.get("_attn_v1_27_buffer_cfg") or {}).get(bucket, {}))
    _capture_with_buffer_graph(
        state,
        shared,
        stage,
        layout,
        h_q,
        k,
        groups,
        groups_pow,
        num_splits,
        scale,
        parents_per_prog,
        bucket,
        cfg["cols"],
        cfg["num_warps"],
        cfg["num_stages"],
    )
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(
            shared,
            stage,
            layout,
            h_q,
            k,
            groups,
            groups_pow,
            num_splits,
            scale,
            parents_per_prog,
            bucket,
            cfg["cols"],
            cfg["num_warps"],
            cfg["num_stages"],
        )
    return shared["out"]


KERNEL = attend
