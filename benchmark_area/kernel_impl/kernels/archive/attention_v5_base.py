"""attention_v5_base — parameterized variant of v2.6.

Same overall structure as v2.6 (fused index+buffer kernel + reduce kernel)
but exposes (num_warps, num_stages, num_splits, parents_per_prog, index_kernel)
as keyword arguments.  Each v5.x kernel calls this with its own config and a
unique ``cache_ns`` so state caches and CUDA graphs do not collide.
"""

from __future__ import annotations

import math

import torch

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

from . import attention_v1_31 as _v31
from .attention_v1_40 import _anchor_layout, _make_shared_workspace
from ._attention_copy_triton import fused_copy_q_th
from ._attention_triton_v2_0_reduce import run_attn_reduce_v2_0
from ._attention_triton_v2_6_index import run_fused_attn_index_buf_v2_6
from ._attention_triton_v2_0_index import run_fused_attn_index_v2_0
from .attention_v1_17 import (
    _GROUPS_MAX,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
)

_LOG2E = 1.4426950408889634


def _launch_no_buffer(
    shared, layout, h_q, k, k_stride, groups, groups_pow,
    num_splits, anchor_s, scale_log2e, parents_per_prog, anchor,
    *, num_warps, num_stages, run_index_no_buf,
):
    run_index_no_buf(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k, k_stride=k_stride,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=num_warps, num_stages=num_stages,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _launch_with_buffer(
    shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
    num_splits, anchor_s, scale_log2e, parents_per_prog,
    bucket, anchor,
    *, num_warps, num_stages, run_index_with_buf,
):
    run_index_with_buf(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k, k_stride=k_stride,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        l_buf_max=bucket,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=num_warps, num_stages=num_stages,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _capture_graph(state, stage, launch_fn, launch_args, kwargs, cache_ns):
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get(f"{cache_ns}_use_cuda_graphs", True)):
        stage["capture_failed"] = True
        return
    stream = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch_fn(*launch_args, **kwargs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch_fn(*launch_args, **kwargs)
        stage["graph"] = graph
    except Exception:
        stage["capture_failed"] = True


def attend_v5(
    q, th_per_subspace, state, buffer_keys, buffer_values,
    keys_children, q_head_to_kv=None, scale=None,
    *,
    cache_ns: str,
    num_warps: int = 4,
    num_stages: int = 3,
    num_splits: int = 85,
    parents_per_prog_override: int | None = None,
    run_index_no_buf=run_fused_attn_index_v2_0,
    run_index_with_buf=run_fused_attn_index_buf_v2_6,
):
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError(f"{cache_ns} requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError(f"{cache_ns} requires build_v2-style state")

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
    cache = state.setdefault(cache_ns, {})
    fixed = cache.get("fixed")
    if cache.get("key") != cache_key or fixed is None:
        layout = _v31._get_layout_fp16(state, q_head_to_kv, q_c)
        _v31._require_supported(layout)
        groups = int(layout["groups"])
        if groups > _GROUPS_MAX:
            raise RuntimeError(f"groups={groups} > {_GROUPS_MAX}")
        groups_pow = max(_v31.next_pow2(groups), 4)
        anchor_s = int(layout["anchor_subspace"])
        if parents_per_prog_override is not None:
            parents_per_prog = int(parents_per_prog_override)
        else:
            parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
        shared = _make_shared_workspace(layout, q_c, th_view, num_splits)
        fixed = {
            "layout": layout, "shared": shared,
            "groups": groups, "groups_pow": groups_pow,
            "anchor_s": anchor_s, "parents_per_prog": parents_per_prog,
            "buckets": {}, "no_buffer_stage": None,
        }
        cache["key"] = cache_key
        cache["fixed"] = fixed

    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout.get("K_used", layout["K"]))
    k_stride = int(layout.get("K_stride", layout["K"]))
    anchor = _anchor_layout(layout)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    fused_copy_q_th(q_c, packed, shared["static_q"], shared["static_th_packed"])

    launch_kwargs_no_buf = dict(
        num_warps=num_warps, num_stages=num_stages,
        run_index_no_buf=run_index_no_buf,
    )
    launch_kwargs_buf = dict(
        num_warps=num_warps, num_stages=num_stages,
        run_index_with_buf=run_index_with_buf,
    )

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_graph(state, stage, _launch_no_buffer,
                       (shared, layout, h_q, k, k_stride, groups, groups_pow,
                        num_splits, anchor_s, scale_log2e, parents_per_prog, anchor),
                       launch_kwargs_no_buf, cache_ns)
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(shared, layout, h_q, k, k_stride, groups, groups_pow,
                              num_splits, anchor_s, scale_log2e, parents_per_prog, anchor,
                              **launch_kwargs_no_buf)
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

    _capture_graph(state, stage, _launch_with_buffer,
                   (shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
                    num_splits, anchor_s, scale_log2e, parents_per_prog,
                    bucket, anchor),
                   launch_kwargs_buf, cache_ns)
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(shared, stage, layout, h_q, k, k_stride, groups, groups_pow,
                            num_splits, anchor_s, scale_log2e, parents_per_prog,
                            bucket, anchor, **launch_kwargs_buf)
    return shared["out"]
