"""Shared helpers for full-fp16 search variants."""

from __future__ import annotations

import torch

from ._search_utils import _mapping_mode, buffer_dot


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _require_fp16_state(
    state: dict,
    *required_keys: str,
) -> None:
    missing = [key for key in required_keys if key not in state]
    if missing:
        req = ", ".join(missing)
        raise RuntimeError(
            f"fp16 search kernel requires fp16 build state with keys: {req}"
        )


def _get_layout_v15_fp16(
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    q: torch.Tensor,
    cache_name: str,
) -> dict:
    _require_fp16_state(state, "keys_blocks_t_f16", "assigns_blocks", "invalid_blocks_i8")

    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_f16 = state["keys_blocks_t_f16"]
    assigns_blocks = state["assigns_blocks"]
    invalid_blocks = state["invalid_blocks_i8"]

    cache = state.setdefault(cache_name, {})
    cache_key = (
        mode,
        groups,
        mapping_sig,
        keys_reord.data_ptr(),
        tuple(keys_reord.shape),
        keys_f16.data_ptr(),
        tuple(keys_f16.shape),
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    max_d = max(widths)
    s = len(dim_slices)
    k = state["K"]
    bf = state["bf"]
    n_pad = state["N_pad"]
    centers_src = state["centers"]
    radii_src = state["radii"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        keys_f16 = keys_f16.index_select(0, q_head_to_kv).contiguous()
        assigns_blocks = assigns_blocks.index_select(1, q_head_to_kv).contiguous()
        invalid_blocks = invalid_blocks.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        h_kv_eff = h_kv

    center_dtype = centers_src[0].dtype
    centers = torch.zeros(s, h_kv_eff, k, max_d, device=q.device, dtype=center_dtype)
    for idx, c in enumerate(centers_src):
        centers[idx, :, :, : c.shape[-1]] = c

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": h_kv_eff,
        "num_subspaces": s,
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
        "centers": centers.contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "keys_blocks_t_f16": keys_f16,
        "assigns_blocks": assigns_blocks,
        "invalid_blocks_i8": invalid_blocks,
        "K": k,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def _get_layout_v12_fp16(
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    q: torch.Tensor,
    cache_name: str,
) -> dict:
    _require_fp16_state(state, "keys_blocks_t_f16", "assigns_parent_major", "invalid_blocks_i8")

    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_f16 = state["keys_blocks_t_f16"]
    assigns_parent = state["assigns_parent_major"]
    invalid_blocks = state["invalid_blocks_i8"]

    cache = state.setdefault(cache_name, {})
    cache_key = (
        mode,
        groups,
        mapping_sig,
        keys_reord.data_ptr(),
        tuple(keys_reord.shape),
        keys_f16.data_ptr(),
        tuple(keys_f16.shape),
        assigns_parent.data_ptr(),
        tuple(assigns_parent.shape),
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    max_d = max(widths)
    s = len(dim_slices)
    k = state["K"]
    bf = state["bf"]
    n_pad = state["N_pad"]
    centers_src = state["centers"]
    radii_src = state["radii"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        keys_f16 = keys_f16.index_select(0, q_head_to_kv).contiguous()
        assigns_parent = assigns_parent.index_select(0, q_head_to_kv).contiguous()
        invalid_blocks = invalid_blocks.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        h_kv_eff = h_kv

    center_dtype = centers_src[0].dtype
    centers = torch.zeros(s, h_kv_eff, k, max_d, device=q.device, dtype=center_dtype)
    for idx, center_s in enumerate(centers_src):
        centers[idx, :, :, : center_s.shape[-1]] = center_s

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": h_kv_eff,
        "num_subspaces": s,
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
        "centers": centers.contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "keys_blocks_t_f16": keys_f16,
        "assigns_parent_major": assigns_parent,
        "invalid_blocks_i8": invalid_blocks,
        "K": k,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def pack_query_for_fp16_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_dtype = layout["centers"].dtype
    q_cast = q if q.dtype == q_dtype else q.to(q_dtype)
    h_q = q_cast.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q_cast.shape[1]

    if d == s * max_d:
        q_packed = q_cast.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        q_packed = torch.zeros(s, h_q, max_d, device=q_cast.device, dtype=q_dtype)
        for si, (s0, e0) in enumerate(layout["dim_slices"]):
            q_packed[si, :, : e0 - s0] = q_cast[:, s0:e0]
        q_packed = q_packed.contiguous()

    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q)
    if th_packed.dtype != q_dtype:
        th_packed = th_packed.to(q_dtype)
    th_packed = th_packed.contiguous()
    return q_cast, q_packed, q_norm, th_packed


def buffer_dot_to_dtype(
    q: torch.Tensor,
    buffer_keys: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    layout: dict,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if buffer_keys is None or buffer_keys.shape[1] == 0:
        return None

    q_for_buffer = q if q.dtype == buffer_keys.dtype else q.to(buffer_keys.dtype)
    out = buffer_dot(q_for_buffer, buffer_keys, q_head_to_kv, layout)
    if out is not None and out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out
