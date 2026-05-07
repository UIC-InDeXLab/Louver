"""attention_v2.11 — v2.6 (fused index+buffer, exp2) with full multi-subspace AND filter.

Pipeline (3 launches, all captured in a single CUDA graph):
  1. cluster_pass: per-subspace parent gating (reused from v1.31)
  2. fused_index+buffer+AND: inline anchor gate + AND across all S subspaces
     via cluster_pass + assigns_blocks, then buffer scan — single kernel
  3. reduce: merge split partials (exp2)
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
from .attention_v1_40 import _anchor_layout
from ._attention_copy_triton import fused_copy_q_th
from ._attention_triton_v1_31_cluster import triton_fused_cluster_pass_rawq_fp16
from ._attention_triton_v2_0_reduce import run_attn_reduce_v2_0
from ._attention_triton_v2_0_index import run_fused_attn_index_v2_0
from ._attention_triton_v2_11_index import run_fused_attn_index_buf_and_v2_11
from .attention_v1_17 import (
    _GROUPS_MAX,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v2.11"

_LOG2E = 1.4426950408889634
_INDEX_NUM_WARPS = 4
_INDEX_NUM_STAGES = 3
_NUM_SPLITS = 85


def _make_shared_workspace(layout: dict, q: torch.Tensor, th_view: torch.Tensor, num_splits: int) -> dict:
    """Workspace with cluster_pass buffer for the AND filter."""
    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    k = layout["K"]
    device = q.device
    static_th_packed = torch.empty((2 * s, h_q), device=device, dtype=torch.float16)
    return {
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float16),
        "static_th_packed": static_th_packed,
        "static_th": static_th_packed[:s],
        "static_q_norms": static_th_packed[s:],
        "cluster_pass": torch.empty(s, h_q, k, device=device, dtype=torch.int8),
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "buf_m": torch.full((h_q,), -1.0e30, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _launch_no_buffer(shared, layout, h_q, k, groups, groups_pow,
                      num_splits, anchor_s, scale_log2e, parents_per_prog,
                      s_subspaces, anchor):
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
    run_fused_attn_index_v2_0(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=_INDEX_NUM_WARPS, num_stages=_INDEX_NUM_STAGES,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                        num_splits, anchor_s, scale_log2e, parents_per_prog,
                        s_subspaces, bucket, anchor):
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
    run_fused_attn_index_buf_and_v2_11(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"], radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        cluster_pass=shared["cluster_pass"],
        assigns_blocks=layout["assigns_blocks"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        dim_offset=anchor["dim_offset"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        l_buf_max=bucket,
        s_subspaces=s_subspaces, anchor_s=anchor_s,
        out_m=shared["m_idx"], out_l=shared["l_idx"], out_o=shared["o_idx"],
        num_warps=_INDEX_NUM_WARPS, num_stages=_INDEX_NUM_STAGES,
    )
    run_attn_reduce_v2_0(
        shared["m_idx"], shared["l_idx"], shared["o_idx"],
        shared["buf_m"], shared["buf_l"], shared["buf_o"], shared["out"],
    )


def _capture_graph(state, stage, launch_fn, launch_args):
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v2_11_use_cuda_graphs", True)):
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
    keys_children, q_head_to_kv=None, scale=None, num_splits: int = _NUM_SPLITS,
):
    del keys_children
    if not HAS_TRITON:
        raise RuntimeError("attention_v2_11 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v2_11 requires build_v2-style state")

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
    cache = state.setdefault("_attn_v2_11_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") != cache_key or fixed is None:
        layout = _v31._get_layout_fp16(state, q_head_to_kv, q_c)
        _v31._require_supported(layout)
        groups = int(layout["groups"])
        if groups > _GROUPS_MAX:
            raise RuntimeError(f"groups={groups} > {_GROUPS_MAX}")
        groups_pow = max(_v31.next_pow2(groups), 4)
        anchor_s = int(layout["anchor_subspace"])
        s_subspaces = int(layout["num_subspaces"])
        parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
        shared = _make_shared_workspace(layout, q_c, th_view, num_splits)
        fixed = {
            "layout": layout, "shared": shared,
            "groups": groups, "groups_pow": groups_pow,
            "anchor_s": anchor_s, "s_subspaces": s_subspaces,
            "parents_per_prog": parents_per_prog,
            "buckets": {}, "no_buffer_stage": None,
        }
        cache["key"] = cache_key
        cache["fixed"] = fixed

    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    s_subspaces = fixed["s_subspaces"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])
    anchor = _anchor_layout(layout)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    fused_copy_q_th(q_c, packed, shared["static_q"], shared["static_th_packed"])

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_graph(state, stage, _launch_no_buffer,
                       (shared, layout, h_q, k, groups, groups_pow,
                        num_splits, anchor_s, scale_log2e, parents_per_prog,
                        s_subspaces, anchor))
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(shared, layout, h_q, k, groups, groups_pow,
                              num_splits, anchor_s, scale_log2e, parents_per_prog,
                              s_subspaces, anchor)
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
                   (shared, stage, layout, h_q, k, groups, groups_pow,
                    num_splits, anchor_s, scale_log2e, parents_per_prog,
                    s_subspaces, bucket, anchor))
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                            num_splits, anchor_s, scale_log2e, parents_per_prog,
                            s_subspaces, bucket, anchor)
    return shared["out"]


KERNEL = attend
