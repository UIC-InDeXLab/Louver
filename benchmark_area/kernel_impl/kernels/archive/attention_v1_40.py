"""attention_v1.40 — v1.34 with anchor gate fused into the index kernel.

Differences vs v1.34 / v1.39:
  - No separate cluster_pass launch. The anchor-only gate (one dot per
    parent + radius bound against threshold) is computed inline inside the
    fused index kernel, removing ~3.5 µs of launch + HBM round-trip per
    query on our benchmark.
  - Single DtoD copy of the packed ``(2*S, H_q)`` threshold/qnorms tensor
    instead of two separate copies, saving ~0.8 µs per call.
  - Keeps v1.34's reduce+buffer kernel for split merging + buffer scan.

Designed to run inside a CUDA Graph, same as v1.34.
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
from ._attention_reduce_buffer_triton_v1_31 import run_attn_reduce_buffer_fp16q
from ._attention_triton import run_attn_reduce
from ._attention_triton_v1_40_index import run_fused_attn_index_anchor_inline_fp16q
from .attention_v1_17 import (
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _GROUPS_MAX,
    _NUM_STAGES,
    _NUM_WARPS,
    _bucket_for,
    _empty_buffer,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
)

KERNEL_VERSION = "v1.40"


def _make_shared_workspace(layout: dict, q: torch.Tensor, th: torch.Tensor, num_splits: int) -> dict:
    """Workspace that packs ``th`` and ``q_norms`` into a single buffer.

    ``static_th_packed`` shape: ``(2*S, H_q) fp16`` — matches the caller's
    packed threshold layout so we can stage everything with one DtoD copy.
    """

    h_q = q.shape[0]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    k = layout["K"]
    device = q.device
    # ``th`` is passed as the S-row slice into the packed layout; we allocate
    # twice that many rows to also hold q-norms.
    static_th_packed = torch.empty((2 * s, h_q), device=device, dtype=torch.float16)
    return {
        "static_q": torch.empty(q.shape, device=device, dtype=torch.float16),
        "static_th_packed": static_th_packed,
        "static_th": static_th_packed[:s],
        "static_q_norms": static_th_packed[s:],
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "buf_m": torch.full(
            (h_q,), -1.0e30, device=device, dtype=torch.float32
        ),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _anchor_layout(layout: dict) -> dict:
    """Cached per-layout slice of anchor centers/radii and its dim offset."""

    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v1_40_anchor")
    if cached is not None and cached["key"] == key:
        return cached
    centers_anchor = layout["centers"][anchor_s].contiguous()
    # Trim any zero-padded trailing dims so the inline gate dot has tight WIDTH.
    width = int(layout["dim_widths"][anchor_s].item())
    centers_anchor = centers_anchor[..., :width].contiguous().to(torch.float16)
    radii_anchor = layout["radii"][anchor_s].contiguous().to(torch.float16)
    cached = {
        "key": key,
        "centers": centers_anchor,
        "radii": radii_anchor,
        "dim_offset": int(layout["dim_offsets"][anchor_s].item()),
        "width": width,
    }
    layout["_attn_v1_40_anchor"] = cached
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
    run_fused_attn_index_anchor_inline_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
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
    run_fused_attn_index_anchor_inline_fp16q(
        q=shared["static_q"],
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=shared["static_th"][anchor_s],
        q_norm_anchor=shared["static_q_norms"][anchor_s],
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        dim_offset=anchor["dim_offset"],
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
    if not bool(state.get("_attn_v1_40_use_cuda_graphs", True)):
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
    if not bool(state.get("_attn_v1_40_use_cuda_graphs", True)):
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


def _get_fixed_runtime_v140(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> dict:
    """Like v1.31's fixed runtime, but with v1.40's workspace (packed th)."""

    cache_key = _v31._fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_40_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = _v31._get_layout_fp16(state, q_head_to_kv, q)
    _v31._require_supported(layout)

    groups = int(layout["groups"])
    if groups > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_40 requires groups <= {_GROUPS_MAX}; got {groups}"
        )
    groups_pow = max(_v31.next_pow2(groups), 4)
    anchor_s = int(layout["anchor_subspace"])
    s_subspaces = int(layout["num_subspaces"])
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
        raise RuntimeError("attention_v1_40 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_40 requires build_v2-style state")

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_40 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _v31._ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_40 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_v31._ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]

    fixed = _get_fixed_runtime_v140(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    shared = fixed["shared"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])
    anchor = _anchor_layout(layout)

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
        _capture_no_buffer_graph(
            state, shared, stage, layout, h_q, k, groups, groups_pow,
            num_splits, anchor_s, scale, parents_per_prog, anchor,
        )
        if stage["graph"] is not None:
            stage["graph"].replay()
        else:
            _launch_no_buffer(
                shared, layout, h_q, k, groups, groups_pow, num_splits,
                anchor_s, scale, parents_per_prog, anchor,
            )
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
    cfg.update((state.get("_attn_v1_40_buffer_cfg") or {}).get(bucket, {}))
    _capture_with_buffer_graph(
        state, shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
        anchor_s, scale, parents_per_prog,
        bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
    )
    if stage["graph"] is not None:
        stage["graph"].replay()
    else:
        _launch_with_buffer(
            shared, stage, layout, h_q, k, groups, groups_pow, num_splits,
            anchor_s, scale, parents_per_prog,
            bucket, cfg["cols"], cfg["num_warps"], cfg["num_stages"], anchor,
        )
    return shared["out"]


KERNEL = attend
