"""attention_v1.34 — anchor-only Triton path with fused reduce+buffer."""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from . import attention_v1_31 as _v31
from ._attention_reduce_buffer_triton_v1_31 import run_attn_reduce_buffer_fp16q
from ._attention_triton import run_attn_reduce
from ._attention_triton_v1_34_cluster import triton_anchor_cluster_pass_rawq_fp16
from ._attention_triton_v1_34_index import run_fused_attn_index_anchor_fp16q
from .attention_v1_17 import (
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
)

KERNEL_VERSION = "v1.34"


def _anchor_layout(layout: dict) -> dict:
    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v1_34_anchor")
    if cached is not None and cached["key"] == key:
        return cached
    cached = {
        "key": key,
        "centers": layout["centers"][anchor_s].contiguous(),
        "radii": layout["radii"][anchor_s].contiguous(),
        "dim_offset": int(layout["dim_offsets"][anchor_s].item()),
    }
    layout["_attn_v1_34_anchor"] = cached
    return cached


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
    anchor: dict,
) -> None:
    triton_anchor_cluster_pass_rawq_fp16(
        q=shared["static_q"],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        th_anchor=shared["static_th"][anchor_s],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        dim_offset=anchor["dim_offset"],
        groups=groups,
        out=shared["cluster_pass"][anchor_s],
    )

    run_fused_attn_index_anchor_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        cluster_pass_anchor=shared["cluster_pass"][anchor_s],
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
    anchor_s: int,
    scale: float,
    parents_per_prog: int,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
    anchor: dict,
) -> None:
    triton_anchor_cluster_pass_rawq_fp16(
        q=shared["static_q"],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        th_anchor=shared["static_th"][anchor_s],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        dim_offset=anchor["dim_offset"],
        groups=groups,
        out=shared["cluster_pass"][anchor_s],
    )

    run_fused_attn_index_anchor_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        cluster_pass_anchor=shared["cluster_pass"][anchor_s],
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
    anchor: dict,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_34_use_cuda_graphs", True)):
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
                    anchor_s, scale, parents_per_prog, anchor,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, anchor,
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
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
    anchor: dict,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_34_use_cuda_graphs", True)):
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
                    anchor_s, scale, parents_per_prog,
                    bucket, buf_cols, buf_warps, buf_stages, anchor,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog,
                bucket, buf_cols, buf_warps, buf_stages, anchor,
            )
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


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
        raise RuntimeError("attention_v1_34 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_34 requires build_v2-style state")

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_34 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _v31._ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_34 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_v31._ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]
    q_norms_view = packed[s_hint:]

    fixed = _v31._get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    groups = fixed["groups"]
    if groups > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_34 requires groups <= {_GROUPS_MAX}; got {groups}"
        )
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
    shared = fixed["shared"]
    anchor = _anchor_layout(layout)
    k = int(layout["K"])

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th"].copy_(th_view)
    shared["static_q_norms"].copy_(q_norms_view)

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("v1_34_no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["v1_34_no_buffer_stage"] = stage
        _capture_no_buffer_graph(
            state, shared, stage, layout, h_q, k, groups, groups_pow,
            num_splits, anchor_s, scale, parents_per_prog, anchor,
        )
        try:
            if stage["graph"] is not None:
                stage["graph"].replay()
            else:
                _launch_no_buffer(
                    shared, layout, h_q, k, groups, groups_pow, num_splits,
                    anchor_s, scale, parents_per_prog, anchor,
                )
        except Exception:
            state["_attn_v1_34_disabled"] = True
            return _v31.attend(
                q=q,
                th_per_subspace=th_per_subspace,
                state=state,
                buffer_keys=buffer_keys,
                buffer_values=buffer_values,
                keys_children=keys_children,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
                num_splits=num_splits,
            )
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)
    bucket_stages = fixed.setdefault("v1_34_buckets", {})
    stage = bucket_stages.get(bucket)
    if stage is None:
        stage = _v31._make_bucket_staging(layout, q_c, bucket)
        bucket_stages[bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys,
        buffer_values,
        layout,
        q_head_to_kv,
    )
    _v31._copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update((state.get("_attn_v1_34_buffer_cfg") or {}).get(bucket, {}))
    _capture_with_buffer_graph(
        state, shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
        anchor_s, scale, parents_per_prog,
        bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
    )
    try:
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog,
                bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
            )
    except Exception:
        state["_attn_v1_34_disabled"] = True
        return _v31.attend(
            q=q,
            th_per_subspace=th_per_subspace,
            state=state,
            buffer_keys=buffer_keys,
            buffer_values=buffer_values,
            keys_children=keys_children,
            q_head_to_kv=q_head_to_kv,
            scale=scale,
            num_splits=num_splits,
        )
    return shared["out"]


KERNEL = attend
