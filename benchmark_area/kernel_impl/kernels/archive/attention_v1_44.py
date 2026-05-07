"""attention_v1.44 — v1.43 architecture with PARENTS_PER_PROG=16.

Smaller chunks (64 cols instead of 128) give the anchor gate finer
pruning granularity at the cost of more loop iterations.
"""

from __future__ import annotations

import math

import torch

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from . import attention_v1_31 as _v31
from .attention_v1_40 import _anchor_layout, _make_shared_workspace
from ._attention_copy_triton import fused_copy_q_th
from ._attention_reduce_buffer_triton_v1_31 import run_attn_reduce_buffer_fp16q
from ._attention_triton import run_attn_reduce
from ._attention_triton_v1_42_index import run_fused_attn_index_v1_42
from .attention_v1_17 import (
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _GROUPS_MAX,
    _bucket_for,
    _empty_buffer,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v1.44"

_PARENTS_PER_PROG = 16
_INDEX_NUM_WARPS = 4
_INDEX_NUM_STAGES = 3


def _launch_no_buffer(shared, layout, h_q, k, groups, groups_pow,
                      num_splits, anchor_s, scale, anchor):
    run_fused_attn_index_v1_42(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits, scale=scale,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=_INDEX_NUM_WARPS, num_stages=_INDEX_NUM_STAGES,
    )
    run_attn_reduce(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                        num_splits, anchor_s, scale, bucket, buf_cols,
                        buf_warps, buf_stages, anchor):
    run_fused_attn_index_v1_42(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits, scale=scale,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=_INDEX_NUM_WARPS, num_stages=_INDEX_NUM_STAGES,
    )
    run_attn_reduce_buffer_fp16q(
        q=shared["static_q"],
        m_idx=shared["m_idx"], l_idx=shared["l_idx"], o_idx=shared["o_idx"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_kv_eff=layout["base_heads"], groups=groups,
        l_buf_max=bucket, buf_cols_per_prog=buf_cols, scale=scale,
        out=shared["out"], groups_tile=4,
        num_warps=buf_warps, num_stages=buf_stages,
    )


def _capture_graph(state, stage, launch_fn, launch_args):
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_44_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch_fn(*launch_args)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_fn(*launch_args)
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def attend(
    q, th_per_subspace, state, buffer_keys, buffer_values,
    keys_children, q_head_to_kv=None, scale=None,
    num_splits: int = _DEFAULT_NUM_SPLITS,
):
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError("attention_v1_44 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_44 requires build_v2-style state")

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]

    cache_key = _v31._fixed_cache_key(state, q_c, th_view, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_44_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") != cache_key or fixed is None:
        layout = _v31._get_layout_fp16(state, q_head_to_kv, q_c)
        _v31._require_supported(layout)
        groups = int(layout["groups"])
        if groups > _GROUPS_MAX:
            raise RuntimeError(f"groups={groups} > {_GROUPS_MAX}")
        groups_pow = max(_v31.next_pow2(groups), 4)
        anchor_s = int(layout["anchor_subspace"])
        shared = _make_shared_workspace(layout, q_c, th_view, num_splits)
        fixed = {
            "layout": layout, "shared": shared,
            "groups": groups, "groups_pow": groups_pow,
            "anchor_s": anchor_s, "buckets": {},
            "no_buffer_stage": None,
        }
        cache["key"] = cache_key
        cache["fixed"] = fixed

    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    k = int(layout["K"])
    anchor = _anchor_layout(layout)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    fused_copy_q_th(q_c, packed, shared["static_q"], shared["static_th_packed"])

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_graph(state, stage, _launch_no_buffer,
                       (shared, layout, h_q, k, groups, groups_pow,
                        num_splits, anchor_s, scale, anchor))
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(shared, layout, h_q, k, groups, groups_pow,
                              num_splits, anchor_s, scale, anchor)
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)
    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _v31._make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys, buffer_values, layout, q_head_to_kv)
    _v31._copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update((state.get("_attn_v1_44_buffer_cfg") or {}).get(bucket, {}))
    _capture_graph(state, stage, _launch_with_buffer,
                   (shared, stage, layout, h_q, k, groups, groups_pow,
                    num_splits, anchor_s, scale,
                    bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor))
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                            num_splits, anchor_s, scale,
                            bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor)
    return shared["out"]


KERNEL = attend
