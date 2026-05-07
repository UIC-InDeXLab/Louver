"""attention_v1.37 — bitpacked anchor gate for the v1.34 path."""

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
from ._attention_triton_v1_37_cluster import triton_anchor_cluster_pass_rawq_fp16_bitpack
from ._attention_triton_v1_37_index import run_fused_attn_index_anchor_bits_fp16q
from .attention_v1_17 import (
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _bucket_for,
    _empty_buffer,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v1.37"

_PARENTS_PER_PROG = 32


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    if batch_nheads_mblocks >= int(0.8 * num_sms):
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1
    eff = [0.0] * max_splits
    max_eff = 0.0
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        n_waves = float(batch_nheads_mblocks * s) / float(num_sms)
        cur = n_waves / math.ceil(n_waves)
        eff[s - 1] = cur
        max_eff = max(max_eff, cur)
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        if eff[s - 1] >= 0.85 * max_eff:
            return s
    return 1


def _choose_num_splits(
    state: dict,
    q: torch.Tensor,
    layout: dict,
    requested: int,
) -> int:
    override = state.get("_attn_v1_37_num_splits")
    if override is not None:
        return int(override)
    max_splits = int(state.get("_attn_v1_37_max_splits", max(requested, 64)))
    num_sms = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
    num_n_blocks = math.ceil(int(layout["K"]) / _PARENTS_PER_PROG)
    batch_nheads_mblocks = int(layout["base_heads"])
    return _num_splits_heuristic(
        batch_nheads_mblocks=batch_nheads_mblocks,
        num_sms=num_sms,
        num_n_blocks=num_n_blocks,
        max_splits=max_splits,
    )


def _anchor_layout(layout: dict) -> dict:
    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v1_37_anchor")
    if cached is not None and cached["key"] == key:
        return cached
    cached = {
        "key": key,
        "centers": layout["centers"][anchor_s].contiguous(),
        "radii": layout["radii"][anchor_s].contiguous(),
        "dim_offset": int(layout["dim_offsets"][anchor_s].item()),
    }
    layout["_attn_v1_37_anchor"] = cached
    return cached


def _ensure_words_workspace(shared: dict, layout: dict, q: torch.Tensor) -> None:
    k_words = (int(layout["K"]) + 31) // 32
    if "cluster_pass_words" not in shared or tuple(shared["cluster_pass_words"].shape) != (q.shape[0], k_words):
        shared["cluster_pass_words"] = torch.empty(
            q.shape[0], k_words, device=q.device, dtype=torch.int32
        )


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
    anchor: dict,
) -> None:
    triton_anchor_cluster_pass_rawq_fp16_bitpack(
        q=shared["static_q"],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        th_anchor=shared["static_th"][anchor_s],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        dim_offset=anchor["dim_offset"],
        groups=groups,
        out_words=shared["cluster_pass_words"],
    )

    run_fused_attn_index_anchor_bits_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        cluster_pass_words=shared["cluster_pass_words"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
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
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
    anchor: dict,
) -> None:
    triton_anchor_cluster_pass_rawq_fp16_bitpack(
        q=shared["static_q"],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        th_anchor=shared["static_th"][anchor_s],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        dim_offset=anchor["dim_offset"],
        groups=groups,
        out_words=shared["cluster_pass_words"],
    )

    run_fused_attn_index_anchor_bits_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        cluster_pass_words=shared["cluster_pass_words"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=layout["base_heads"],
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits,
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
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
    anchor: dict,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_37_use_cuda_graphs", True)):
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
                    anchor_s, scale, anchor,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, anchor,
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
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
    anchor: dict,
) -> None:
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_37_use_cuda_graphs", True)):
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
                    anchor_s, scale,
                    bucket, buf_cols, buf_warps, buf_stages, anchor,
                )
        current.wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale,
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
        raise RuntimeError("attention_v1_37 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_37 requires build_v2-style state")

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_37 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _v31._ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_37 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_v31._ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
        packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]
    q_norms_view = packed[s_hint:]

    layout = _v31._get_layout_fp16(state, q_head_to_kv, q_c)
    anchor = _anchor_layout(layout)
    num_splits_eff = _choose_num_splits(state, q_c, layout, num_splits)
    fixed = _v31._get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits_eff)
    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    k = int(layout["K"])
    _ensure_words_workspace(shared, layout, q_c)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th"].copy_(th_view)
    shared["static_q_norms"].copy_(q_norms_view)

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("v1_37_no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["v1_37_no_buffer_stage"] = stage
        _capture_no_buffer_graph(
            state, shared, stage, layout, h_q, k, groups, groups_pow,
            num_splits_eff, anchor_s, scale, anchor,
        )
        try:
            if stage["graph"] is not None:
                stage["graph"].replay()
            else:
                _launch_no_buffer(
                    shared, layout, h_q, k, groups, groups_pow, num_splits_eff,
                    anchor_s, scale, anchor,
                )
        except Exception:
            state["_attn_v1_37_disabled"] = True
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
    bucket_stages = fixed.setdefault("v1_37_buckets", {})
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
    cfg.update((state.get("_attn_v1_37_buffer_cfg") or {}).get(bucket, {}))
    _capture_with_buffer_graph(
        state, shared, stage, layout, h_q, k, groups, groups_pow, num_splits_eff,
        anchor_s, scale,
        bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
    )
    try:
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_with_buffer(
                shared, stage, layout, h_q, k, groups, groups_pow, num_splits_eff,
                anchor_s, scale,
                bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
            )
    except Exception:
        state["_attn_v1_37_disabled"] = True
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
