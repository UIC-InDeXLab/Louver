"""attention_v1.18 — v1.17 plus incremental append-only buffer staging.

The decode buffer in this codebase grows monotonically until flush. v1.16/v1.17
restage the entire buffer into the fixed bucket tensors on every query. This
variant only copies the newly appended suffix within a bucket and reuses the
previously staged prefix.
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
from .attention_v1_17 import (
    _ALLOWED_S,
    _BUCKETS,
    _DEFAULT_BUFFER_CFG,
    _DEFAULT_NUM_SPLITS,
    _GROUPS_MAX,
    _GROUPS_POW_FLOOR,
    _buffer_cfg,
    _bucket_for,
    _capture_no_buffer_graph,
    _capture_with_buffer_graph,
    _empty_buffer,
    _launch_no_buffer,
    _launch_with_buffer,
    _make_shared_workspace,
    _parents_per_prog_for_bf,
    _prepare_buffer_effective,
    _require_supported,
)

KERNEL_VERSION = "v1.18"


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
        th.dtype,
        tuple(th.shape),
        num_splits,
        q_map_ptr,
        q_map_shape,
        state["keys_reord"].data_ptr(),
        state["values_blocks_f16"].data_ptr(),
    )


def _copy_buffer_into_stage_incremental(
    stage: dict,
    buffer_keys_eff: torch.Tensor,
    buffer_values_eff: torch.Tensor,
) -> None:
    l_buf = int(buffer_keys_eff.shape[1])
    prev_len = int(stage["valid_len"])
    if l_buf < prev_len:
        stage["buf_invalid"].fill_(1)
        prev_len = 0

    if l_buf == prev_len:
        return

    keys_src = buffer_keys_eff[:, prev_len:l_buf, :].transpose(-1, -2)
    if keys_src.dtype != torch.float16:
        keys_src = keys_src.to(torch.float16)
    stage["buf_keys_t"][:, :, prev_len:l_buf].copy_(keys_src)

    values_src = buffer_values_eff[:, prev_len:l_buf, :]
    if values_src.dtype != torch.float16:
        values_src = values_src.to(torch.float16)
    stage["buf_values"][:, prev_len:l_buf, :].copy_(values_src)

    stage["buf_invalid"][:, prev_len:l_buf].zero_()
    stage["valid_len"] = l_buf


def _get_fixed_runtime(
    state: dict,
    q: torch.Tensor,
    th: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    num_splits: int,
) -> dict:
    cache_key = _fixed_cache_key(state, q, th, q_head_to_kv, num_splits)
    cache = state.setdefault("_attn_v1_18_fixed", {})
    fixed = cache.get("fixed")
    if cache.get("key") == cache_key and fixed is not None:
        return fixed

    layout = get_layout_attn_rawq(
        state,
        q_head_to_kv,
        q,
        cache_name="_attn_v1_18_layout",
    )
    _require_supported(layout)

    groups = int(layout["groups"])
    if groups > _GROUPS_MAX:
        raise RuntimeError(
            f"attention_v1_18 requires groups <= {_GROUPS_MAX}; got {groups}"
        )
    groups_pow = max(next_pow2(groups), _GROUPS_POW_FLOOR)
    anchor_s = int(layout["anchor_subspace"])
    s_subspaces = int(layout["num_subspaces"])
    if s_subspaces not in _ALLOWED_S:
        raise RuntimeError(
            f"attention_v1_18 requires S in {_ALLOWED_S}; got S={s_subspaces}"
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
        raise RuntimeError("attention_v1_18 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1_18 requires build_v2-style state")

    h_q = q.shape[0]
    d = q.shape[1]
    q_c = q if q.is_contiguous() else q.contiguous()

    fixed_probe = state.get("_attn_v1_18_fixed", {}).get("fixed")
    s_hint = int(fixed_probe["s_subspaces"]) if fixed_probe is not None else None
    if s_hint is None:
        if th_per_subspace.dim() == 2 and th_per_subspace.shape[0] in _ALLOWED_S:
            s_hint = int(th_per_subspace.shape[0])
        else:
            s_hint = int(th_per_subspace.numel() // h_q)
        if s_hint not in _ALLOWED_S:
            raise RuntimeError(
                f"attention_v1_18 requires S in {_ALLOWED_S}; inferred S={s_hint}"
            )

    if th_per_subspace.shape == (s_hint, h_q) and th_per_subspace.is_contiguous():
        th_view = th_per_subspace
    else:
        th_view = th_per_subspace.reshape(s_hint, h_q).contiguous()

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
    cfg.update((state.get("_attn_v1_18_buffer_cfg") or {}).get(bucket, {}))
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
