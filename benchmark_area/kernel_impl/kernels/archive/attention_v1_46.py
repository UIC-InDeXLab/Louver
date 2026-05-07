"""attention_v1.46 — Native CUDA fused anchor-gate + attention.

Replaces the Triton index kernel (v1.42) and fused copy kernel with a
single native CUDA kernel featuring:
  - cp.async pipelining for K/V loads
  - exp2f + pre-multiplied scale_log2 (single HW instruction)
  - Inline anchor gate (no separate cluster_pass)
  - C++ dispatch (no Python launch overhead)
  - Native CUDA reduce kernel (replaces Triton reduce)

Falls back to v1.43 (Triton) if the CUDA extension fails to load.
"""

from __future__ import annotations

import math

import torch

from . import attention_v1_31 as _v31
from .attention_v1_40 import _anchor_layout, _make_shared_workspace
from ._attention_cuda_v1_46 import (
    is_supported as _cuda_supported,
    run_fused_attn_index_cuda_v1_46,
    run_attn_reduce_cuda_v1_46,
)
from ._attention_reduce_buffer_triton_v1_31 import run_attn_reduce_buffer_fp16q
from .attention_v1_17 import (
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _GROUPS_MAX,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v1.46"


def _keys_blocks_cuda(layout: dict) -> torch.Tensor:
    """Convert keys from (H_kv, K, D, BF) transposed to (H_kv, K, BF, D) row-major."""
    cache_key_name = "_keys_blocks_v1_46"
    src = layout["keys_blocks_t_f16"]
    key = (src.data_ptr(), tuple(src.shape))
    cached = layout.get(cache_key_name)
    if cached is not None and layout.get(cache_key_name + "_key") == key:
        return cached
    # src is (H_kv, K, D, BF) — permute to (H_kv, K, BF, D)
    result = src.permute(0, 1, 3, 2).contiguous()
    layout[cache_key_name] = result
    layout[cache_key_name + "_key"] = key
    return result


def _launch_no_buffer(
    shared: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    num_splits: int,
    scale: float,
    anchor: dict,
    cols_per_chunk: int,
) -> None:
    run_fused_attn_index_cuda_v1_46(
        q=shared["static_q"],
        keys_blocks=_keys_blocks_cuda(layout),
        values_blocks=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][int(layout["anchor_subspace"])],
        qnorm_anchor=shared["static_q_norms"][int(layout["anchor_subspace"])],
        invalid_blocks=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv=int(layout["base_heads"]),
        k=k,
        num_splits=num_splits,
        groups=groups,
        dim_offset=anchor["dim_offset"],
        anchor_width=anchor["width"],
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        cols_per_chunk=cols_per_chunk,
    )
    run_attn_reduce_cuda_v1_46(
        m_idx=shared["m_idx"],
        l_idx=shared["l_idx"],
        o_idx=shared["o_idx"],
        m_buf=shared["buf_m"],
        l_buf=shared["buf_l"],
        o_buf=shared["buf_o"],
        out=shared["out"],
        num_splits=num_splits,
    )


def _launch_with_buffer(
    shared: dict,
    stage: dict,
    layout: dict,
    h_q: int,
    k: int,
    groups: int,
    num_splits: int,
    scale: float,
    bucket: int,
    buf_cols: int,
    buf_warps: int,
    buf_stages: int,
    anchor: dict,
    cols_per_chunk: int,
) -> None:
    run_fused_attn_index_cuda_v1_46(
        q=shared["static_q"],
        keys_blocks=_keys_blocks_cuda(layout),
        values_blocks=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][int(layout["anchor_subspace"])],
        qnorm_anchor=shared["static_q_norms"][int(layout["anchor_subspace"])],
        invalid_blocks=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv=int(layout["base_heads"]),
        k=k,
        num_splits=num_splits,
        groups=groups,
        dim_offset=anchor["dim_offset"],
        anchor_width=anchor["width"],
        scale=scale,
        out_m=shared["m_idx"],
        out_l=shared["l_idx"],
        out_o=shared["o_idx"],
        cols_per_chunk=cols_per_chunk,
    )
    run_attn_reduce_buffer_fp16q(
        q=shared["static_q"],
        m_idx=shared["m_idx"],
        l_idx=shared["l_idx"],
        o_idx=shared["o_idx"],
        buf_keys_t_f16=stage["buf_keys_t"],
        buf_values_f16=stage["buf_values"],
        buf_invalid_i8=stage["buf_invalid"],
        h_kv_eff=int(layout["base_heads"]),
        groups=groups,
        l_buf_max=bucket,
        buf_cols_per_prog=buf_cols,
        scale=scale,
        out=shared["out"],
        groups_tile=4,
        num_warps=buf_warps,
        num_stages=buf_stages,
    )


def _capture_graph(state, stage, launch_fn, launch_args):
    if stage["graph"] is not None or stage["capture_failed"]:
        return
    if not bool(state.get("_attn_v1_46_use_cuda_graphs", True)):
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

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_46 requires build_v2-style state")

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_46 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _v31._ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_46 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_v31._ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]

    cache_key = _v31._fixed_cache_key(state, q_c, th_view, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_46_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") != cache_key or fixed is None:
        layout = _v31._get_layout_fp16(state, q_head_to_kv, q_c)
        _v31._require_supported(layout)
        groups = int(layout["groups"])
        if groups > _GROUPS_MAX:
            raise RuntimeError(
                f"attention_v1_46 requires groups <= {_GROUPS_MAX}; got {groups}"
            )
        anchor_s = int(layout["anchor_subspace"])
        shared = _make_shared_workspace(layout, q_c, th_view, num_splits)
        cols_per_chunk = _parents_per_prog_for_bf(int(layout["bf"]), groups) * int(layout["bf"])
        fixed = {
            "layout": layout,
            "shared": shared,
            "groups": groups,
            "anchor_s": anchor_s,
            "cols_per_chunk": cols_per_chunk,
            "buckets": {},
            "no_buffer_stage": None,
        }
        cache["key"] = cache_key
        cache["fixed"] = fixed

    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    cols_per_chunk = fixed["cols_per_chunk"]
    k = int(layout["K"])
    anchor = _anchor_layout(layout)

    # Check if the CUDA kernel supports this shape
    if not _cuda_supported(q_c, layout):
        from . import attention_v1_43 as _fallback
        return _fallback.attend(
            q=q, th_per_subspace=th_per_subspace, state=state,
            buffer_keys=buffer_keys, buffer_values=buffer_values,
            keys_children=keys_children, q_head_to_kv=q_head_to_kv,
            scale=scale, num_splits=num_splits,
        )

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    shared["static_q"].copy_(q_c)
    shared["static_th_packed"].copy_(packed)

    if _empty_buffer(buffer_keys, buffer_values):
        stage = fixed.get("no_buffer_stage")
        if stage is None:
            stage = {"graph": None, "capture_failed": False}
            fixed["no_buffer_stage"] = stage
        _capture_graph(
            state, stage, _launch_no_buffer,
            (shared, layout, h_q, k, groups, num_splits, scale, anchor,
             cols_per_chunk),
        )
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(
                shared, layout, h_q, k, groups, num_splits, scale, anchor,
                cols_per_chunk)
        return shared["out"]

    l_buf = int(buffer_keys.shape[1])
    bucket = _bucket_for(l_buf)
    stage = fixed["buckets"].get(bucket)
    if stage is None:
        stage = _v31._make_bucket_staging(layout, q_c, bucket)
        fixed["buckets"][bucket] = stage

    buf_keys_eff, buf_values_eff = _prepare_buffer_effective(
        buffer_keys, buffer_values, layout, q_head_to_kv,
    )
    _v31._copy_buffer_into_stage_incremental(stage, buf_keys_eff, buf_values_eff)

    cfg = dict(_DEFAULT_BUFFER_CFG[bucket])
    cfg.update((state.get("_attn_v1_46_buffer_cfg") or {}).get(bucket, {}))
    _capture_graph(
        state, stage, _launch_with_buffer,
        (shared, stage, layout, h_q, k, groups, num_splits, scale,
         bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
         cols_per_chunk),
    )
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(
            shared, stage, layout, h_q, k, groups, num_splits, scale,
            bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
            cols_per_chunk,
        )
    return shared["out"]


KERNEL = attend
