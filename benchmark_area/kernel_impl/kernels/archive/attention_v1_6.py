"""attention_v1.6 — v1.5 + extra-workspace reuse + adaptive NUM_SPLITS.

Combines the two validated wins (workspace reuse from v1.2, fused q-pack
from v1.5) and adds two micro-tweaks that are neutral-to-slight-win in
isolation:
  - Reuse buffer-sentinel tensors across calls (saves 3 torch.{full,zeros}
    allocations per empty-buffer call).
  - Adaptive NUM_SPLITS = ceil(K / 32) clamped to [1, 16] (v1.3 idea).
"""

from __future__ import annotations

import math

import torch

try:
    import triton

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._attention_triton import NEG_SENT, run_attn_reduce, run_fused_attn_index
from ._attention_triton_v1_5 import triton_fused_cluster_pass_rawq
from ._search_utils import _mapping_mode

KERNEL_VERSION = "v1.6"
_PARENTS_PER_PROG = 8
_PARENTS_PER_SPLIT_TGT = 32


def _auto_num_splits(k: int) -> int:
    return max(1, min(16, (k + _PARENTS_PER_SPLIT_TGT - 1) // _PARENTS_PER_SPLIT_TGT))


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _get_layout_attn(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    if "values_blocks_f16" not in state:
        raise RuntimeError(
            "attention_v1 requires build_v2.4 state "
            "(missing `values_blocks_f16` — pass values to build)."
        )

    cache_src = state.setdefault("_attn_v1_0_key_pack", {})
    keys_reord_ptr = keys_reord.data_ptr()
    key_cache_key = (
        keys_reord_ptr,
        tuple(keys_reord.shape),
        state["K"],
        state["bf"],
        len(state["assigns_reord"]),
    )
    if cache_src.get("key") == key_cache_key:
        keys_f16 = cache_src["keys_f16"]
        assigns_blocks = cache_src["assigns_blocks"]
        invalid_blocks = cache_src["invalid_blocks"]
    else:
        h_kv_, _, d = keys_reord.shape
        k = state["K"]
        bf = state["bf"]
        s = len(state["assigns_reord"])
        keys_f16 = (
            keys_reord.view(h_kv_, k, bf, d)
            .permute(0, 1, 3, 2)
            .to(torch.float16)
            .contiguous()
        )
        assigns_blocks = (
            torch.stack(state["assigns_reord"], dim=0)
            .to(torch.int16 if k < 32768 else torch.int32)
            .view(s, h_kv_, k, bf)
            .contiguous()
        )
        invalid_blocks = (
            state["invalid_mask"].view(h_kv_, k, bf).to(torch.int8).contiguous()
        )
        cache_src["key"] = key_cache_key
        cache_src["keys_f16"] = keys_f16
        cache_src["assigns_blocks"] = assigns_blocks
        cache_src["invalid_blocks"] = invalid_blocks

    values_f16 = state["values_blocks_f16"]

    cache = state.setdefault("_attn_v1_6_layout", {})
    cache_key = (
        mode,
        groups,
        mapping_sig,
        keys_f16.data_ptr(),
        values_f16.data_ptr(),
        tuple(keys_f16.shape),
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    offsets = [start for start, end in dim_slices]
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
        keys_f16_eff = keys_f16.index_select(0, q_head_to_kv).contiguous()
        values_f16_eff = values_f16.index_select(0, q_head_to_kv).contiguous()
        assigns_blocks_eff = assigns_blocks.index_select(1, q_head_to_kv).contiguous()
        invalid_blocks_eff = invalid_blocks.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        keys_f16_eff = keys_f16
        values_f16_eff = values_f16
        assigns_blocks_eff = assigns_blocks
        invalid_blocks_eff = invalid_blocks
        h_kv_eff = h_kv

    centers = torch.zeros(s, h_kv_eff, k, max_d, device=q.device, dtype=torch.float32)
    for idx, c in enumerate(centers_src):
        centers[idx, :, :, : c.shape[-1]] = c

    dim_offsets_t = torch.tensor(offsets, device=q.device, dtype=torch.int32)
    dim_widths_t = torch.tensor(widths, device=q.device, dtype=torch.int32)

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": h_kv_eff,
        "num_subspaces": s,
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
        "dim_offsets": dim_offsets_t,
        "dim_widths": dim_widths_t,
        "centers": centers.contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "keys_blocks_t_f16": keys_f16_eff,
        "values_blocks_f16": values_f16_eff,
        "assigns_blocks": assigns_blocks_eff,
        "invalid_blocks_i8": invalid_blocks_eff,
        "K": k,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
        "D_v": values_f16.shape[-1],
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def _buffer_partial(
    q: torch.Tensor,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    layout: dict,
    scale: float,
    d_v: int,
    sentinels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )
    if empty:
        return sentinels  # (m, l, o) filled with (-inf, 0, 0) once.

    if layout["mode"] == "expanded":
        assert q_head_to_kv is not None
        k_buf = buffer_keys.index_select(0, q_head_to_kv)
        v_buf = buffer_values.index_select(0, q_head_to_kv)
    elif layout["mode"] == "grouped":
        groups = layout["groups"]
        k_buf = buffer_keys.repeat_interleave(groups, dim=0)
        v_buf = buffer_values.repeat_interleave(groups, dim=0)
    else:
        k_buf = buffer_keys
        v_buf = buffer_values

    k_buf = k_buf.to(torch.float32)
    v_buf = v_buf.to(torch.float32)

    scores = torch.bmm(q.unsqueeze(1), k_buf.transpose(-1, -2)).squeeze(1) * scale
    m = scores.max(dim=-1).values
    p = torch.exp(scores - m.unsqueeze(-1))
    l_ = p.sum(dim=-1)
    o = torch.bmm(p.unsqueeze(1), v_buf).squeeze(1)
    return m.contiguous(), l_.contiguous(), o.contiguous()


def attend(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("attention_v1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("attention_v1 requires build_v2-style state")

    layout = _get_layout_attn(state, q_head_to_kv, q)
    h_q = q.shape[0]
    d = q.shape[1]
    d_v = layout["D_v"]
    s = layout["num_subspaces"]
    h_kv_eff = layout["base_heads"]
    k = layout["K"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    if num_splits is None:
        num_splits = _auto_num_splits(k)

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    q_c = q if q.is_contiguous() else q.contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    ws = state.setdefault("_attn_v1_6_ws", {})
    ws_key = (h_q, num_splits, d_v, k, s, q.device.index)
    if ws.get("key") != ws_key:
        ws["m_idx"] = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
        ws["l_idx"] = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
        ws["o_idx"] = torch.empty(h_q, num_splits, d_v, device=q.device, dtype=torch.float32)
        ws["out"] = torch.empty(h_q, d_v, device=q.device, dtype=torch.float32)
        ws["cluster_pass"] = torch.empty(s, h_q, k, device=q.device, dtype=torch.int8)
        ws["buf_m"] = torch.full((h_q,), NEG_SENT, device=q.device, dtype=torch.float32)
        ws["buf_l"] = torch.zeros((h_q,), device=q.device, dtype=torch.float32)
        ws["buf_o"] = torch.zeros((h_q, d_v), device=q.device, dtype=torch.float32)
        ws["key"] = ws_key

    m_idx = ws["m_idx"]
    l_idx = ws["l_idx"]
    o_idx = ws["o_idx"]
    out = ws["out"]
    cluster_pass = ws["cluster_pass"]
    sentinels = (ws["buf_m"], ws["buf_l"], ws["buf_o"])

    triton_fused_cluster_pass_rawq(
        q=q_c,
        th=th_packed,
        dim_offsets=layout["dim_offsets"],
        dim_widths=layout["dim_widths"],
        centers=layout["centers"],
        radii=layout["radii"],
        groups=groups,
        out=cluster_pass,
    )

    run_fused_attn_index(
        q=q_c,
        keys_blocks_t_f16=layout["keys_blocks_t_f16"],
        values_blocks_f16=layout["values_blocks_f16"],
        assigns_blocks=layout["assigns_blocks"],
        cluster_pass=cluster_pass,
        invalid_blocks_i8=layout["invalid_blocks_i8"],
        h_q=h_q,
        h_kv_eff=h_kv_eff,
        k=k,
        groups=groups,
        groups_pow=groups_pow,
        s_subspaces=s,
        parents_per_prog=_PARENTS_PER_PROG,
        num_splits=num_splits,
        anchor_s=anchor_s,
        scale=float(scale),
        out_m=m_idx,
        out_l=l_idx,
        out_o=o_idx,
    )

    m_buf, l_buf, o_buf = _buffer_partial(
        q, buffer_keys, buffer_values, q_head_to_kv, layout, scale, d_v, sentinels
    )

    run_attn_reduce(m_idx, l_idx, o_idx, m_buf, l_buf, o_buf, out)
    return out


KERNEL = attend
