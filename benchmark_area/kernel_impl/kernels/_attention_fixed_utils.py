"""Shared helpers for fixed-shape attention_v1.x experiments."""

from __future__ import annotations

import torch


def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _mapping_mode(
    q_head_to_kv: torch.Tensor | None,
    h_q: int,
    h_kv: int,
) -> tuple[str, int, tuple[int, ...] | None]:
    if q_head_to_kv is None:
        return "identity", 1, None

    if h_q % h_kv == 0:
        groups = h_q // h_kv
        expected = (
            torch.arange(h_q, device=q_head_to_kv.device, dtype=q_head_to_kv.dtype)
            // groups
        )
        if torch.equal(q_head_to_kv, expected):
            return "grouped", groups, None

    return "expanded", 1, tuple(int(x) for x in q_head_to_kv.tolist())


def get_layout_attn_rawq(
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    q: torch.Tensor,
    *,
    cache_name: str,
) -> dict:
    """Prepare the raw-q attention layout used by v1.5+ kernels."""
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    if "values_blocks_f16" not in state:
        raise RuntimeError(
            "attention_v1 requires build_v2.4 state "
            "(missing `values_blocks_f16` — pass values to build)."
        )

    cache_src = state.setdefault("_attn_v1_key_pack", {})
    keys_reord_ptr = keys_reord.data_ptr()
    k = int(state["K"])
    k_used = int(state.get("K_used", k))
    k_stride = int(state.get("K_cap", k))
    key_cache_key = (
        keys_reord_ptr,
        tuple(keys_reord.shape),
        k,
        state["bf"],
        len(state["assigns_reord"]),
    )
    if cache_src.get("key") == key_cache_key:
        keys_f16 = cache_src["keys_f16"]
        assigns_blocks = cache_src["assigns_blocks"]
        invalid_blocks = cache_src["invalid_blocks"]
    else:
        h_kv_, _, d = keys_reord.shape
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

    cache = state.setdefault(cache_name, {})
    cache_key = (
        mode,
        groups,
        mapping_sig,
        keys_f16.data_ptr(),
        values_f16.data_ptr(),
        tuple(keys_f16.shape),
        k_used,
        k_stride,
    )
    if cache.get("key") == cache_key:
        return cache["layout"]

    dim_slices = state["dim_slices"]
    widths = [end - start for start, end in dim_slices]
    offsets = [start for start, end in dim_slices]
    max_d = max(widths)
    s = len(dim_slices)
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
        "K_used": k_used,
        "K_stride": k_stride,
        "bf": bf,
        "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
        "D_v": values_f16.shape[-1],
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def require_fixed_bf_s(layout: dict, *, bf: int = 4, s: int = 8, groups_max: int = 4) -> None:
    if int(layout["bf"]) != bf or int(layout["num_subspaces"]) != s:
        raise RuntimeError(
            f"fixed-shape kernel requires bf={bf}, S={s}; "
            f"got bf={layout['bf']}, S={layout['num_subspaces']}"
        )
    if int(layout["groups"]) > groups_max:
        raise RuntimeError(
            f"fixed-shape kernel requires groups <= {groups_max}; got {layout['groups']}"
        )


def buffer_partial(
    q: torch.Tensor,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    layout: dict,
    scale: float,
    d_v: int,
    sentinels: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = (
        buffer_keys is None
        or buffer_values is None
        or buffer_keys.shape[1] == 0
    )
    if empty:
        if sentinels is not None:
            return sentinels
        h_q = q.shape[0]
        device = q.device
        return (
            torch.full((h_q,), -1.0e30, device=device, dtype=torch.float32),
            torch.zeros((h_q,), device=device, dtype=torch.float32),
            torch.zeros((h_q, d_v), device=device, dtype=torch.float32),
        )

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
