"""attention_v1.3 — v1.0 with adaptive NUM_SPLITS.

Derives num_splits from K so that each split owns roughly 32 parents.
For K=279, BF=16 → num_splits ≈ 9 (was 16 hardcoded). Reduces partial
(H_q, NUM_SPLITS, D_v) HBM write volume + reduce-kernel work, at the
cost of slightly less intra-kvh parallelism.
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
from ._search_triton import triton_fused_cluster_pass
from ._search_utils import _mapping_mode

KERNEL_VERSION = "v1.3"
_PARENTS_PER_PROG = 8
_DEFAULT_NUM_SPLITS = None     # adaptive; see _auto_num_splits
_PARENTS_PER_SPLIT_TGT = 32


def _auto_num_splits(k: int) -> int:
    return max(1, min(16, (k + _PARENTS_PER_SPLIT_TGT - 1) // _PARENTS_PER_SPLIT_TGT))


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _pack_q(q: torch.Tensor, layout: dict) -> tuple[torch.Tensor, torch.Tensor]:
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]
    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        q_packed = q.new_zeros(s, h_q, max_d)
        for si, (s0, e0) in enumerate(layout["dim_slices"]):
            q_packed[si, :, : e0 - s0] = q[:, s0:e0]
        q_packed = q_packed.contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    return q_packed, q_norm


def _get_layout_attn(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    """Attention-specific layout cache. Extends search_v15 layout with V."""
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    if "values_blocks_f16" not in state:
        raise RuntimeError(
            "attention_v1 requires build_v2.4 state "
            "(missing `values_blocks_f16` — pass values to build)."
        )

    # fp16 key pack (same as search_v15).
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

    cache = state.setdefault("_attn_v1_0_layout", {})
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

    layout = {
        "mode": mode,
        "groups": groups,
        "base_heads": h_kv_eff,
        "num_subspaces": s,
        "max_d": max_d,
        "dim_slices": tuple(dim_slices),
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
    q: torch.Tensor,                  # (H_q, D)
    buffer_keys: torch.Tensor | None, # (H_kv, N_buf, D) or None
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    layout: dict,
    scale: float,
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce (m, l, o) partial for the buffer. Returns sentinels if empty."""
    h_q = q.shape[0]
    device = q.device

    empty = (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )
    if empty:
        m = torch.full((h_q,), NEG_SENT, device=device, dtype=torch.float32)
        l_ = torch.zeros((h_q,), device=device, dtype=torch.float32)
        o = torch.zeros((h_q, d_v), device=device, dtype=torch.float32)
        return m, l_, o

    # GQA expansion.
    if layout["mode"] == "expanded":
        assert q_head_to_kv is not None
        k_buf = buffer_keys.index_select(0, q_head_to_kv)
        v_buf = buffer_values.index_select(0, q_head_to_kv)
    elif layout["mode"] == "grouped":
        groups = layout["groups"]
        # Expand (H_kv, N_buf, D) to (H_q=H_kv*groups, N_buf, D) by repeat_interleave.
        k_buf = buffer_keys.repeat_interleave(groups, dim=0)
        v_buf = buffer_values.repeat_interleave(groups, dim=0)
    else:
        k_buf = buffer_keys
        v_buf = buffer_values

    k_buf = k_buf.to(torch.float32)
    v_buf = v_buf.to(torch.float32)

    scores = torch.bmm(q.unsqueeze(1), k_buf.transpose(-1, -2)).squeeze(1) * scale  # (H_q, N_buf)
    m = scores.max(dim=-1).values
    p = torch.exp(scores - m.unsqueeze(-1))
    l_ = p.sum(dim=-1)
    o = torch.bmm(p.unsqueeze(1), v_buf).squeeze(1)  # (H_q, D_v)
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
    num_splits: int | None = _DEFAULT_NUM_SPLITS,
) -> torch.Tensor:
    """Fused sparse-index attention.

    Returns:
        (H_q, D_v) attention output in f32.
    """
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

    # cluster_pass — same as search_v15.
    q_packed, q_norm = _pack_q(q, layout)
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()
    cluster_pass = triton_fused_cluster_pass(
        q_packed, q_norm, th_packed,
        layout["centers"], layout["radii"], groups,
    )

    # Index partials (H_q, NUM_SPLITS[, D_v]).
    m_idx = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
    l_idx = torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32)
    o_idx = torch.empty(h_q, num_splits, d_v, device=q.device, dtype=torch.float32)

    run_fused_attn_index(
        q=q.contiguous(),
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

    # Buffer partial.
    m_buf, l_buf, o_buf = _buffer_partial(
        q, buffer_keys, buffer_values, q_head_to_kv, layout, scale, d_v
    )

    # Reduce.
    out = torch.empty(h_q, d_v, device=q.device, dtype=torch.float32)
    run_attn_reduce(m_idx, l_idx, o_idx, m_buf, l_buf, o_buf, out)
    return out


KERNEL = attend
