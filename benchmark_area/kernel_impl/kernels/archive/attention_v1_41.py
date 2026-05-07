"""attention_v1.41 — graph-free v1.40 with zero staging copies.

The CUDA-graph path in v1.40 requires two DtoD staging copies (q + packed_th)
per query, costing ~1.6 µs of GPU time.  Since our total GPU work (~28 µs)
always exceeds the CPU launch overhead of two Triton kernels (~10 µs), the GPU
never stalls and graphs provide no benefit — the DtoD copies are pure waste.

This version:
  - Passes the caller's tensors directly into the Triton kernels.
  - Never captures or replays CUDA graphs.
  - Pre-allocates only the output/partial buffers (m, l, o, out) once.
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

KERNEL_VERSION = "v1.41"


def _anchor_layout(layout: dict) -> dict:
    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v1_41_anchor")
    if cached is not None and cached["key"] == key:
        return cached
    width = int(layout["dim_widths"][anchor_s].item())
    cached = {
        "key": key,
        "centers": layout["centers"][anchor_s][..., :width].contiguous().to(torch.float16),
        "radii": layout["radii"][anchor_s].contiguous().to(torch.float16),
        "dim_offset": int(layout["dim_offsets"][anchor_s].item()),
        "width": width,
    }
    layout["_attn_v1_41_anchor"] = cached
    return cached


def _make_workspace(layout: dict, h_q: int, num_splits: int) -> dict:
    d_v = layout["D_v"]
    device = layout["keys_blocks_t_f16"].device
    return {
        "m_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "l_idx": torch.empty(h_q, num_splits, device=device, dtype=torch.float32),
        "o_idx": torch.empty(h_q, num_splits, d_v, device=device, dtype=torch.float32),
        "out": torch.empty(h_q, d_v, device=device, dtype=torch.float32),
        "buf_m": torch.full((h_q,), -1.0e30, device=device, dtype=torch.float32),
        "buf_l": torch.zeros((h_q,), device=device, dtype=torch.float32),
        "buf_o": torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
    }


def _get_fixed_runtime(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> dict:
    cache_key = _v31._fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_41_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = _v31._get_layout_fp16(state, q_head_to_kv, q)
    _v31._require_supported(layout)

    groups = int(layout["groups"])
    if groups > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_41 requires groups <= {_GROUPS_MAX}; got {groups}"
        )
    groups_pow = max(_v31.next_pow2(groups), 4)
    anchor_s = int(layout["anchor_subspace"])
    parents_per_prog = _parents_per_prog_for_bf(int(layout["bf"]), groups)
    h_q = q.shape[0]
    ws = _make_workspace(layout, h_q, num_splits)

    fixed = {
        "layout": layout,
        "ws": ws,
        "groups": groups,
        "groups_pow": groups_pow,
        "anchor_s": anchor_s,
        "parents_per_prog": parents_per_prog,
        "buckets": {},
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
        raise RuntimeError("attention_v1_41 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_41 requires build_v2-style state")

    _v31._require_fp16("q", q)
    _v31._require_fp16("th_per_subspace", th_per_subspace)
    _v31._require_fp16("buffer_keys", buffer_keys)
    _v31._require_fp16("buffer_values", buffer_values)

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    if th_per_subspace.dim() != 2 or th_per_subspace.shape[1] != h_q:
        raise RuntimeError(
            "attention_v1_41 expects packed fp16 thresholds with shape (2*S, H_q)"
        )
    rows = int(th_per_subspace.shape[0])
    s_hint = rows // 2
    if rows % 2 != 0 or s_hint not in _v31._ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_41 expects packed fp16 thresholds with 2*S rows "
            f"for S in {_v31._ALLOWED_S}; got shape={tuple(th_per_subspace.shape)}"
        )
    packed = th_per_subspace if th_per_subspace.is_contiguous() else th_per_subspace.contiguous()
    th_view = packed[:s_hint]
    q_norms_view = packed[s_hint:]

    fixed = _get_fixed_runtime(state, q_c, th_view, q_head_to_kv, num_splits)
    layout = fixed["layout"]
    ws = fixed["ws"]
    groups = fixed["groups"]
    groups_pow = fixed["groups_pow"]
    anchor_s = fixed["anchor_s"]
    parents_per_prog = fixed["parents_per_prog"]
    k = int(layout["K"])
    anchor = _anchor_layout(layout)

    if scale is None:
        scale = 1.0 / math.sqrt(d)
    scale = float(scale)

    run_fused_attn_index_anchor_inline_fp16q(
        q=q_c,
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        centers_anchor=anchor["centers"],
        radii_anchor=anchor["radii"],
        th_anchor=th_view[anchor_s],
        q_norm_anchor=q_norms_view[anchor_s],
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
        out_m=ws["m_idx"],
        out_l=ws["l_idx"],
        out_o=ws["o_idx"],
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    if _empty_buffer(buffer_keys, buffer_values):
        run_attn_reduce(
            ws["m_idx"],
            ws["l_idx"],
            ws["o_idx"],
            ws["buf_m"],
            ws["buf_l"],
            ws["buf_o"],
            ws["out"],
        )
    else:
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
        cfg.update((state.get("_attn_v1_41_buffer_cfg") or {}).get(bucket, {}))

        run_attn_reduce_buffer_fp16q(
            q=q_c,
            m_idx=ws["m_idx"],
            l_idx=ws["l_idx"],
            o_idx=ws["o_idx"],
            buf_keys_t_f16=stage["buf_keys_t"],
            buf_values_f16=stage["buf_values"],
            buf_invalid_i8=stage["buf_invalid"],
            h_kv_eff=layout["base_heads"],
            groups=groups,
            l_buf_max=bucket,
            buf_cols_per_prog=cfg["cols"],
            scale=scale,
            out=ws["out"],
            groups_tile=4,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return ws["out"]


KERNEL = attend
