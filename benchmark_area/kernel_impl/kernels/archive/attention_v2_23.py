"""attention_v2.23 — v2.15 with tuned cluster_pass (BLOCK_K=128) and
child_survive (BLOCK_C=512, num_warps=2).

Larger BLOCK_K → better tensor core utilization, fewer blocks.
Larger BLOCK_C → fewer blocks, more work per block in child_survive.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

from . import attention_v1_31 as _v31
from .attention_v1_40 import _anchor_layout, _make_shared_workspace
from ._attention_copy_triton import fused_copy_q_th
from ._attention_triton_v2_0_reduce import run_attn_reduce_v2_0
from ._attention_triton_v2_12_index import run_fused_attn_index_buf_v2_12
from ._attention_triton_v2_15_cluster import _cluster_pass_dot_kernel
from ._attention_triton_v2_12_survive import _child_survive_kernel
from .attention_v1_17 import (
    _GROUPS_MAX,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v2.23"

_LOG2E = 1.4426950408889634
_INDEX_NUM_WARPS = 4
_INDEX_NUM_STAGES = 3
_NUM_SPLITS = 85


def _run_cluster_pass_128(q, q_norms, th, dim_offsets, dim_widths,
                          centers, radii, groups, out):
    h_q, d = q.shape
    s, h_kv, k, max_d = centers.shape
    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2
    block_k = 128
    grid = (s, h_kv, triton.cdiv(k, block_k))
    _cluster_pass_dot_kernel[grid](
        q, q_norms, dim_offsets, dim_widths, th, centers, radii, out,
        h_q, h_kv, k,
        D=d, S=s, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=4,
    )


def _run_child_survive_512(cluster_pass, assigns_blocks, invalid_blocks_i8,
                           h_q, h_kv, k, bf, groups, s_subspaces, anchor_s, out):
    total_children = k * bf
    block_c = 512
    grid = (h_q, triton.cdiv(total_children, block_c))
    _child_survive_kernel[grid](
        cluster_pass, assigns_blocks, invalid_blocks_i8, out,
        h_q, h_kv, k,
        GROUPS=groups, BF=bf,
        S=s_subspaces, ANCHOR_S=anchor_s,
        BLOCK_C=block_c,
        num_warps=2,
    )


def _make_shared(layout, q, th_view, num_splits):
    base = _make_shared_workspace(layout, q, th_view, num_splits)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    k = layout["K"]
    bf = layout["bf"]
    base["cluster_pass"] = torch.empty(s, h_q, k, device=q.device, dtype=torch.int8)
    base["child_survive"] = torch.empty(h_q, k * bf, device=q.device, dtype=torch.int8)
    return base


def _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                        num_splits, anchor_s, scale_log2e, parents_per_prog,
                        s_subspaces, bucket):
    _run_cluster_pass_128(
        q=shared["static_q"], q_norms=shared["static_q_norms"],
        th=shared["static_th"], dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"], centers=layout["centers"],
        radii=layout["radii"], groups=groups, out=shared["cluster_pass"],
    )
    _run_child_survive_512(
        cluster_pass=shared["cluster_pass"],
        assigns_blocks=layout["assigns_blocks"],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q, h_kv=layout["base_heads"], k=k,
        bf=int(layout["bf"]), groups=groups,
        s_subspaces=s_subspaces, anchor_s=anchor_s,
        out=shared["child_survive"],
    )
    run_fused_attn_index_buf_v2_12(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        child_survive=shared["child_survive"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_q=h_q, h_kv_eff=layout["base_heads"], k=k,
        groups=groups, groups_pow=groups_pow,
        parents_per_prog=parents_per_prog,
        num_splits=num_splits, scale_log2e=scale_log2e,
        l_buf_max=bucket,
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
    if not bool(state.get("_attn_v2_23_use_cuda_graphs", True)):
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
        raise RuntimeError("v2.23 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("v2.23 requires build_v2-style state")

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
    cache = state.setdefault("_attn_v2_23_fixed", {})
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
        shared = _make_shared(layout, q_c, th_view, num_splits)
        fixed = {
            "layout": layout, "shared": shared,
            "groups": groups, "groups_pow": groups_pow,
            "anchor_s": anchor_s, "s_subspaces": s_subspaces,
            "parents_per_prog": parents_per_prog,
            "buckets": {},
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

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale_log2e = float(scale) * _LOG2E

    fused_copy_q_th(q_c, packed, shared["static_q"], shared["static_th_packed"])

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf) if not _empty_buffer(buffer_keys, buffer_values) else 0

    if bucket == 0 and _empty_buffer(buffer_keys, buffer_values):
        bucket = 256

    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _v31._make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    if not _empty_buffer(buffer_keys, buffer_values):
        buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
            buffer_keys, buffer_values, layout, q_head_to_kv)
        _v31._copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    _capture_graph(state, stage, _launch_with_buffer,
                   (shared, stage, layout, h_q, k, groups, groups_pow,
                    num_splits, anchor_s, scale_log2e, parents_per_prog,
                    s_subspaces, bucket))
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(shared, stage, layout, h_q, k, groups, groups_pow,
                            num_splits, anchor_s, scale_log2e, parents_per_prog,
                            s_subspaces, bucket)
    return shared["out"]


KERNEL = attend
