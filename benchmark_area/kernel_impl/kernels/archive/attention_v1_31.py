"""attention_v1.31 — Tier A fusions on top of v1.18.

Combines two of v1.30's/v1.18.1's wins into a single kernel orchestrator:
  1. Fused buffer into reduce (v1.30 approach) — single kernel for
     reduce-partials + scan-buffer, eliminating a separate buffer-attn
     launch plus the buf_m/buf_l/buf_o HBM round-trip.
  3. FP16 q once per query (v1.18.1 approach) — q/centers/radii are
     pre-cast to fp16 outside the timed path; cluster_pass and the sparse
     index kernel both consume fp16 q; no per-kernel fp32<->fp16 cast.

Note: Tier A item 2 (grid-sync cluster_pass fusion) is not implemented
here. True cooperative launch in Triton requires either a full rewrite
into a single persistent kernel or inline PTX; we leave that for a later
Tier C rewrite.

Public API matches attention_v1_18_1: q/th/buffer must be fp16,
th_per_subspace is packed (2*S, H_q) with thresholds then q-norms.
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
from ._attention_reduce_buffer_triton_v1_31 import run_attn_reduce_buffer_fp16q
from ._attention_triton_v1_31_index import run_fused_attn_index_fp16q
from ._attention_triton_v1_31_cluster import triton_fused_cluster_pass_rawq_fp16
from .attention_v1_17 import (
    _ALLOWED_S,
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _GROUPS_MAX,
    _GROUPS_POW_FLOOR,
    _NUM_STAGES,
    _NUM_WARPS,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
    _require_supported,
)

KERNEL_VERSION = "v1.31"


def _require_fp16(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and tensor.dtype != torch.float16:
        raise RuntimeError(
            f"attention_v1_31 expects {name} in fp16; got {tensor.dtype}. "
            "Convert inputs outside timing before calling this kernel."
        )


def _make_bucket_staging(layout: dict, q: torch.Tensor, bucket: int) -> dict:
    device = q.device
    d = q.shape[1]
    d_v = layout["D_v"]
    h_kv_eff = int(layout["base_heads"])
    return {
        "buf_keys_t": torch.zeros(h_kv_eff, d, bucket, device=device, dtype=torch.float16),
        "buf_values": torch.zeros(h_kv_eff, bucket, d_v, device=device, dtype=torch.float16),
        "buf_invalid": torch.ones(h_kv_eff, bucket, device=device, dtype=torch.int8),
        "valid_len": 0,
        "graph": None,
        "capture_failed": False,
    }


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
        q.dtype,
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _get_layout_fp16(
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    q: torch.Tensor,
) -> dict:
    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_31_layout",
    )
    cache = state.setdefault("_attn_v1_31_layout_fp16", {})
    cache_key = (
        layout["mode"],
        layout["groups"],
        layout["base_heads"],
        layout["num_subspaces"],
        layout["K"],
        layout["bf"],
        layout["centers"].data_ptr(),
        layout["radii"].data_ptr(),
        layout["keys_blocks_t_f16"].data_ptr(),
        layout["values_blocks_f16"].data_ptr(),
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    layout_fp16 = dict(layout)
    layout_fp16["centers"] = layout["centers"].to(torch.float16).contiguous()
    layout_fp16["radii"] = layout["radii"].to(torch.float16).contiguous()
    cache["key"] = cache_key
    cache["layout"] = layout_fp16
    return layout_fp16


def _make_shared_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    k = layout["K"]
    device = q.device
    return {
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float16),
        "static_th": torch.empty(th.shape, device=device, dtype=torch.float16),
        "static_q_norms": torch.empty((s, h_q), device=device, dtype=torch.float16),
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        # kept only for the no-buffer path that still uses run_attn_reduce
        "buf_m": torch.full((h_q,), NEG_SENT, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _copy_buffer_into_stage_incremental(
    stage: dict,
    buffer_keys_eff: torch.Tensor,
    buffer_values_eff: torch.Tensor,
) -> None:
    _require_fp16("buffer_keys", buffer_keys_eff)
    _require_fp16("buffer_values", buffer_values_eff)

    l_buf = int(buffer_keys_eff.shape[1])
    prev_len = int(stage["valid_len"])
    if l_buf < prev_len:
        stage["buf_invalid"].fill_(1)
        prev_len = 0

    if l_buf == prev_len:
        return

    keys_src = buffer_keys_eff[:, prev_len:l_buf, :].transpose(-1, -2)
    stage["buf_keys_t"][:, :, prev_len:l_buf].copy_(keys_src)

    values_src = buffer_values_eff[:, prev_len:l_buf, :]
    stage["buf_values"][:, prev_len:l_buf, :].copy_(values_src)

    stage["buf_invalid"][:, prev_len:l_buf].zero_()
    stage["valid_len"] = l_buf


def _launch_no_buffer(
    shared: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    groups_pow: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
) -> None:
    triton_fused_cluster_pass_rawq_fp16(
        q=shared["static_q"],
        q_norms=shared["static_q_norms"],
        th=shared["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=shared["cluster_pass"],
    )

    run_fused_attn_index_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=shared["cluster_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s_subspaces,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    run_attn_reduce(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"],
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
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    triton_fused_cluster_pass_rawq_fp16(
        q=shared["static_q"],
        q_norms=shared["static_q_norms"],
        th=shared["static_th"],
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=shared["cluster_pass"],
    )

    run_fused_attn_index_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=shared["cluster_pass"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s_subspaces,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    # Fused reduce + buffer scan — eliminates the separate buffer_attn
    # launch and the buf_m/buf_l/buf_o round-trip through HBM.
    run_attn_reduce_buffer_fp16q(
        q=shared["static_q"],
        m_idx=shared["m_idx"],
        l_idx=shared["l_idx"],
        o_idx=shared["o_idx"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_kv_eff=layout["base_heads"],
        groups=groups,
        l_buf_max=bucket,
        buf_cols_per_prog=buf_cols,
        scale=scale,
        out=shared["out"],
        groups_tile=4,
        num_warps=buf_warps,
        num_stages=buf_stages,
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
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_31_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_no_buffer(
                    shared, layout, h_q, k, groups, groups_pow, num_splits,
                    anchor_s, scale, parents_per_prog, s_subspaces,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
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
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    s_subspaces: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_31_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return

    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                _launch_with_buffer(
                    shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                    anchor_s, scale, parents_per_prog, s_subspaces,
                    bucket, buf_cols, buf_warps, buf_stages,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
                bucket, buf_cols, buf_warps, buf_stages,
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
    cache = state.setdefault("_attn_v1_31_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = _get_layout_fp16(state, q_head_to_kv, q)
    _require_supported(layout)

    groups = int(layout["groups"])
    if groups > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_31 requires groups <= {_GROUPS_MAX}; got {groups}"
        )
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    anchor_s = int(layout["anchor_subspace"])
    s_subspaces = int(layout["num_subspaces"])
    if s_subspaces not in _ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_31 requires S in {_ALLOWED_S}; got S={s_subspaces}"
        )
    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
    shared = _make_shared_workspace(layout, q, th, num_splits)

    fixed = {
        "layout": layout,
        "shared": shared,
        "groups": groups,
        "groups_pow": groups_pow,
        "anchor_s": anchor_s,
        "s_subspaces": s_subspaces,
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
        raise RuntimeError("attention_v1_31 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_31 requires build_v2-style state")

    _require_fp16("q", q)
    _require_fp16("th_per_subspace", th_per_subspace)
    _require_fp16("buffer_keys", buffer_keys)
    _require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_31 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_31 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]
    q_norms_view = packed[s_hint:]

    fixed = _get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    s_subspaces = fixed["s_subspaces"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th"].copy_(th_view)
    shared["static_q_norms"].copy_(q_norms_view)

    empty = _empty_buffer(buffer_keys, buffer_values)
    if empty:
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_no_buffer_graph(
            state, shared, stage, layout, h_q, k, groups, groups_pow,
            num_splits, anchor_s, scale, parents_per_prog, s_subspaces,
        )
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, s_subspaces,
            )
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)

    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys, buffer_values, layout, q_head_to_kv,
    )
    _copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update((state.get("_attn_v1_31_buffer_cfg") or {}).get(bucket, {}))
    _capture_with_buffer_graph(
        state, shared, stage, layout, h_q, k, groups, groups_pow,
        num_splits, anchor_s, scale, parents_per_prog, s_subspaces,
        bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"],
    )
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(
            shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
            anchor_s, scale, parents_per_prog, s_subspaces,
            bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"],
        )
    return shared["out"]


KERNEL = attend
