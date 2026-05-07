"""Shared helpers for experimental update_v3 kernels."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._build_update_active import build_v2_0_seeded_state, build_v2_4_state

K_CAP_CHUNK = 1024


def assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _round_up(x: int, chunk: int) -> int:
    return ((int(x) + chunk - 1) // chunk) * chunk


def _arena_k_cap_for(k_needed: int) -> int:
    return _round_up(int(k_needed) + K_CAP_CHUNK, K_CAP_CHUNK)


def maybe_merged(
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    old_values: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    return_merged: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not return_merged:
        return None, None
    new_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
    new_values = None
    if old_values is not None and buffer_values is not None:
        new_values = torch.cat([old_values, buffer_values], dim=1).contiguous()
    return new_keys, new_values


def drop_redundant_value_reord(state: dict) -> dict:
    state.pop("values_reord", None)
    return state


def ensure_block_tensors(state: dict) -> dict:
    """Materialize derived block tensors when a build kernel left them lazy.

    build_v2_7 keeps the compact reordered tensors and lets attention helpers
    derive block layouts on demand. The v3/v4 arena merge path needs those
    block tensors explicitly so it can copy appended ranges into capacity
    storage.
    """
    keys_reord = state["keys_reord"]
    invalid_mask = state["invalid_mask"]
    assigns_reord = state["assigns_reord"]
    h_kv, _, d = keys_reord.shape
    k = int(state["K"])
    bf = int(state["bf"])
    s = len(assigns_reord)

    if "keys_blocks_t" not in state:
        state["keys_blocks_t"] = (
            keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
        )
    if "assigns_blocks" not in state:
        state["assigns_blocks"] = (
            torch.stack(assigns_reord, dim=0)
            .to(assign_dtype(k))
            .view(s, h_kv, k, bf)
            .contiguous()
        )
    if "invalid_blocks_i8" not in state:
        state["invalid_blocks_i8"] = invalid_mask.view(h_kv, k, bf).to(torch.int8).contiguous()
    return state


def build_sub_cpu(
    buffer_keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    anchor_subspace: int,
    buffer_values: torch.Tensor | None,
    *,
    with_values: bool = True,
) -> dict:
    state = build_v2_4_state(
        keys=buffer_keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=0,
        anchor_subspace=anchor_subspace,
        values=buffer_values if with_values else None,
    )
    return drop_redundant_value_reord(state)


def build_sub_gpu_rounds(
    buffer_keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    anchor_subspace: int,
    buffer_values: torch.Tensor | None,
    *,
    with_values: bool = True,
) -> dict:
    state = build_v2_0_seeded_state(
        keys=buffer_keys,
        bf=bf,
        n_subspaces=n_subspaces,
        refine_iter=0,
        anchor_subspace=anchor_subspace,
        balance_mode="gpu_rounds",
    )

    keys_reord = state["keys_reord"]
    invalid_mask = state["invalid_mask"]
    assigns_reord = state["assigns_reord"]
    h_kv, _, d = keys_reord.shape
    k = int(state["K"])
    s = len(assigns_reord)

    state["keys_blocks_t"] = (
        keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
    )
    state["assigns_blocks"] = (
        torch.stack(assigns_reord, dim=0)
        .to(assign_dtype(k))
        .view(s, h_kv, k, bf)
        .contiguous()
    )
    state["invalid_blocks_i8"] = invalid_mask.view(h_kv, k, bf).to(torch.int8).contiguous()

    if with_values and buffer_values is not None:
        pack_values_torch(state, buffer_values)
    return state


def pack_values_torch(state: dict, values: torch.Tensor) -> None:
    reorder_perm: torch.Tensor = state["reorder_perm"]
    invalid_mask: torch.Tensor = state["invalid_mask"]
    h_kv, n_pad_state = reorder_perm.shape
    h_kv_v, n_raw, d_v = values.shape
    if h_kv != h_kv_v:
        raise ValueError(f"head mismatch: reorder={h_kv} vs values={h_kv_v}")

    pad = n_pad_state - n_raw
    if pad > 0:
        pad_zeros = torch.zeros(h_kv, pad, d_v, device=values.device, dtype=values.dtype)
        values_padded = torch.cat([values, pad_zeros], dim=1)
    elif pad == 0:
        values_padded = values
    else:
        raise ValueError(f"values has more rows ({n_raw}) than N_pad ({n_pad_state})")

    values_reord = values_padded.gather(
        1, reorder_perm[..., None].expand(-1, -1, d_v)
    ).masked_fill(invalid_mask[..., None], 0.0)
    state["values_blocks_f16"] = (
        values_reord.view(h_kv, int(state["K"]), int(state["bf"]), d_v)
        .to(torch.float16)
        .contiguous()
    )
    state["D_v"] = d_v


if HAS_TRITON:

    @triton.jit
    def _pack_values_blocks_kernel(
        Values_ptr,
        Reorder_ptr,
        Invalid_ptr,
        Out_ptr,
        H,
        N_RAW,
        K,
        BF: tl.constexpr,
        D_V: tl.constexpr,
        BLOCK_DV: tl.constexpr,
    ):
        h = tl.program_id(0)
        k = tl.program_id(1)
        child = tl.program_id(2)
        dv = tl.arange(0, BLOCK_DV)
        mask_dv = dv < D_V

        phys = k * BF + child
        src = tl.load(Reorder_ptr + h * (K * BF) + phys)
        invalid = tl.load(Invalid_ptr + h * (K * BF) + phys) != 0
        valid = (src < N_RAW) & (~invalid)
        vals = tl.load(
            Values_ptr + (h * N_RAW + src) * D_V + dv,
            mask=mask_dv & valid,
            other=0.0,
        ).to(tl.float16)
        tl.store(
            Out_ptr + ((h * K + k) * BF + child) * D_V + dv,
            vals,
            mask=mask_dv,
        )


def pack_values_direct(state: dict, values: torch.Tensor) -> None:
    if not HAS_TRITON:
        pack_values_torch(state, values)
        return
    if values.shape[-1] > 256:
        pack_values_torch(state, values)
        return

    h_kv, n_raw, d_v = values.shape
    k = int(state["K"])
    bf = int(state["bf"])
    out = torch.empty(h_kv, k, bf, d_v, device=values.device, dtype=torch.float16)
    block_dv = next_pow2(d_v)
    _pack_values_blocks_kernel[(h_kv, k, bf)](
        values,
        state["reorder_perm"],
        state["invalid_mask"],
        out,
        h_kv,
        n_raw,
        k,
        BF=bf,
        D_V=d_v,
        BLOCK_DV=block_dv,
        num_warps=4,
    )
    state["values_blocks_f16"] = out
    state["D_v"] = d_v


def merge_cat(
    state: dict,
    sub: dict,
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    old_values: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    bf: int,
    anchor_subspace: int,
    return_merged: bool,
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    ensure_block_tensors(state)
    ensure_block_tensors(sub)

    k_old = int(state.get("K_used", state["K"]))
    k_sub = int(sub["K"])
    k_new = k_old + k_sub
    n_old = int(state.get("N_used", state["N"]))
    n_sub = int(sub["N"])
    n_pad_new = k_new * bf

    keys_reord_new = torch.cat(
        [state["keys_reord"][:, : k_old * bf], sub["keys_reord"]], dim=1
    ).contiguous()
    invalid_mask_new = torch.cat(
        [state["invalid_mask"][:, : k_old * bf], sub["invalid_mask"]], dim=1
    ).contiguous()
    reorder_perm_new = torch.cat(
        [state["reorder_perm"][:, : k_old * bf], sub["reorder_perm"] + n_old], dim=1
    ).contiguous()

    assigns_reord_new: list[torch.Tensor] = []
    for s_old, s_sub in zip(state["assigns_reord"], sub["assigns_reord"]):
        assigns_reord_new.append(
            torch.cat([s_old[:, : k_old * bf], s_sub + k_old], dim=1).contiguous()
        )

    centers_new = [
        torch.cat([c_old[:, :k_old], c_sub], dim=1).contiguous()
        for c_old, c_sub in zip(state["centers"], sub["centers"])
    ]
    radii_new = [
        torch.cat([r_old[:, :k_old], r_sub], dim=1).contiguous()
        for r_old, r_sub in zip(state["radii"], sub["radii"])
    ]

    keys_blocks_t_new = torch.cat(
        [state["keys_blocks_t"][:, :k_old], sub["keys_blocks_t"]], dim=1
    ).contiguous()
    invalid_blocks_i8_new = torch.cat(
        [state["invalid_blocks_i8"][:, :k_old], sub["invalid_blocks_i8"]], dim=1
    ).contiguous()

    dtype = assign_dtype(k_new)
    old_blocks = state["assigns_blocks"][:, :, :k_old]
    if old_blocks.dtype != dtype:
        old_blocks = old_blocks.to(dtype)
    sub_blocks = sub["assigns_blocks"].to(dtype) + k_old
    assigns_blocks_new = torch.cat([old_blocks, sub_blocks], dim=2).contiguous()

    new_state = {
        "dim_slices": state["dim_slices"],
        "centers": centers_new,
        "radii": radii_new,
        "assigns_reord": assigns_reord_new,
        "keys_reord": keys_reord_new,
        "invalid_mask": invalid_mask_new,
        "reorder_perm": reorder_perm_new,
        "K": k_new,
        "K_used": k_new,
        "K_cap": k_new,
        "N": n_old + n_sub,
        "N_used": n_old + n_sub,
        "bf": bf,
        "N_pad": n_pad_new,
        "anchor_subspace": state.get("anchor_subspace", anchor_subspace),
        "keys_blocks_t": keys_blocks_t_new,
        "assigns_blocks": assigns_blocks_new,
        "invalid_blocks_i8": invalid_blocks_i8_new,
    }

    if "values_blocks_f16" in state and "values_blocks_f16" in sub:
        new_state["values_blocks_f16"] = torch.cat(
            [state["values_blocks_f16"][:, :k_old], sub["values_blocks_f16"]], dim=1
        ).contiguous()
        new_state["D_v"] = state["D_v"]

    new_keys, new_values = maybe_merged(
        old_keys, buffer_keys, old_values, buffer_values, return_merged
    )
    return new_state, new_keys, new_values


def _copy_state_to_arena(src: dict, k_cap: int, bf: int, assign_blocks_dtype: torch.dtype) -> dict:
    ensure_block_tensors(src)

    k_used = int(src.get("K_used", src["K"]))
    n_used = int(src.get("N_used", src["N"]))
    n_pad_used = k_used * bf
    n_pad_cap = k_cap * bf

    keys_reord = src["keys_reord"]
    h_kv, _, d = keys_reord.shape
    device = keys_reord.device

    keys_reord_cap = torch.zeros(h_kv, n_pad_cap, d, device=device, dtype=keys_reord.dtype)
    keys_reord_cap[:, :n_pad_used].copy_(keys_reord[:, :n_pad_used])

    invalid_mask_cap = torch.ones(h_kv, n_pad_cap, device=device, dtype=torch.bool)
    invalid_mask_cap[:, :n_pad_used].copy_(src["invalid_mask"][:, :n_pad_used])

    reorder_perm_cap = torch.zeros(h_kv, n_pad_cap, device=device, dtype=src["reorder_perm"].dtype)
    reorder_perm_cap[:, :n_pad_used].copy_(src["reorder_perm"][:, :n_pad_used])

    assigns_reord_cap = []
    for a in src["assigns_reord"]:
        a_cap = torch.zeros(h_kv, n_pad_cap, device=device, dtype=a.dtype)
        a_cap[:, :n_pad_used].copy_(a[:, :n_pad_used])
        assigns_reord_cap.append(a_cap)

    centers_cap = []
    radii_cap = []
    for c, r in zip(src["centers"], src["radii"]):
        c_cap = torch.zeros(h_kv, k_cap, c.shape[-1], device=device, dtype=c.dtype)
        r_cap = torch.zeros(h_kv, k_cap, device=device, dtype=r.dtype)
        c_cap[:, :k_used].copy_(c[:, :k_used])
        r_cap[:, :k_used].copy_(r[:, :k_used])
        centers_cap.append(c_cap)
        radii_cap.append(r_cap)

    keys_blocks_t = src["keys_blocks_t"]
    keys_blocks_t_cap = torch.zeros(
        h_kv, k_cap, keys_blocks_t.shape[2], bf, device=device, dtype=keys_blocks_t.dtype
    )
    keys_blocks_t_cap[:, :k_used].copy_(keys_blocks_t[:, :k_used])

    invalid_blocks_cap = torch.ones(h_kv, k_cap, bf, device=device, dtype=torch.int8)
    invalid_blocks_cap[:, :k_used].copy_(src["invalid_blocks_i8"][:, :k_used])

    s_dim = len(src["assigns_reord"])
    assigns_blocks_cap = torch.zeros(
        s_dim, h_kv, k_cap, bf, device=device, dtype=assign_blocks_dtype
    )
    assigns_blocks_cap[:, :, :k_used].copy_(
        src["assigns_blocks"][:, :, :k_used].to(assign_blocks_dtype)
    )

    arena = {
        "dim_slices": src["dim_slices"],
        "centers": centers_cap,
        "radii": radii_cap,
        "assigns_reord": assigns_reord_cap,
        "keys_reord": keys_reord_cap,
        "invalid_mask": invalid_mask_cap,
        "reorder_perm": reorder_perm_cap,
        "K": k_cap,
        "K_used": k_used,
        "K_cap": k_cap,
        "N": n_used,
        "N_used": n_used,
        "bf": bf,
        "N_pad": n_pad_cap,
        "anchor_subspace": src.get("anchor_subspace", 0),
        "keys_blocks_t": keys_blocks_t_cap,
        "assigns_blocks": assigns_blocks_cap,
        "invalid_blocks_i8": invalid_blocks_cap,
    }

    if "values_blocks_f16" in src:
        values = src["values_blocks_f16"]
        values_cap = torch.zeros(
            h_kv, k_cap, bf, values.shape[-1], device=device, dtype=values.dtype
        )
        values_cap[:, :k_used].copy_(values[:, :k_used])
        arena["values_blocks_f16"] = values_cap
        arena["D_v"] = src["D_v"]
    return arena


def _arena_from_base(state: dict, k_needed: int, bf: int) -> tuple[dict, int, int]:
    base_k = int(state["K"])
    base_n = int(state["N"])
    k_cap = _arena_k_cap_for(k_needed)
    dtype = assign_dtype(k_cap)
    cache = state.setdefault("_update_v3_arena_cache", {})
    arena = cache.get("arena")
    if (
        arena is None
        or int(arena["K_cap"]) < k_cap
        or arena["assigns_blocks"].dtype != dtype
    ):
        arena = _copy_state_to_arena(state, k_cap, bf, dtype)
        cache["arena"] = arena
    arena["K_used"] = base_k
    arena["N_used"] = base_n
    arena["N"] = base_n
    return arena, base_k, base_n


def _grow_arena(state: dict, k_needed: int, bf: int) -> dict:
    k_new_cap = _arena_k_cap_for(k_needed)
    return _copy_state_to_arena(state, k_new_cap, bf, assign_dtype(k_new_cap))


_ATTN_LAYOUT_CACHE_KEYS = (
    "_attn_v1_key_pack",
    "_attn_v1_31_layout",
    "_attn_v1_31_layout_fp16",
    "_attn_v1_5_layout",
    "_attn_v1_14_layout",
    "_attn_v1_15_layout",
    "_attn_v1_15_layout_fp16",
    "_attn_v1_16_fixed",
    "_attn_v1_17_fixed",
    "_attn_v1_18_fixed",
    "_attn_v1_20_fixed",
    "_attn_v1_22_fixed",
    "_attn_v1_23_fixed",
    "_attn_v1_24_fixed",
    "_attn_v2_6_fixed",
    "_attn_v2_15_fixed",
)


def invalidate_attention_layout_caches(state: dict) -> None:
    """Drop cached attention layouts so the next attend() rebuilds them.

    Required after any in-place mutation of the arena state because the layout
    cache keys on data_ptr only — and in-place writes to keys_reord / centers /
    radii do not change the data_ptr.
    """
    for k in _ATTN_LAYOUT_CACHE_KEYS:
        state.pop(k, None)


def merge_arena(
    state: dict,
    sub: dict,
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    old_values: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    bf: int,
    anchor_subspace: int,
    return_merged: bool,
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
    k_sub = int(sub["K"])
    if "K_cap" in state:
        arena = state if int(state["K_cap"]) >= int(state["K_used"]) + k_sub else _grow_arena(
            state, int(state["K_used"]) + k_sub, bf
        )
        k_old = int(arena["K_used"])
        n_old = int(arena["N_used"])
    else:
        k_old_base = int(state["K"])
        arena, k_old, n_old = _arena_from_base(state, k_old_base + k_sub, bf)

    k_new = k_old + k_sub
    n_sub = int(sub["N"])
    n_new = n_old + n_sub
    n_pad_old = k_old * bf
    n_pad_new = k_new * bf

    arena["keys_reord"][:, n_pad_old:n_pad_new].copy_(sub["keys_reord"])
    arena["invalid_mask"][:, n_pad_old:n_pad_new].copy_(sub["invalid_mask"])
    arena["reorder_perm"][:, n_pad_old:n_pad_new].copy_(sub["reorder_perm"] + n_old)
    for dst, src in zip(arena["assigns_reord"], sub["assigns_reord"]):
        dst[:, n_pad_old:n_pad_new].copy_(src + k_old)

    for c_dst, c_src in zip(arena["centers"], sub["centers"]):
        c_dst[:, k_old:k_new].copy_(c_src)
    for r_dst, r_src in zip(arena["radii"], sub["radii"]):
        r_dst[:, k_old:k_new].copy_(r_src)

    arena["keys_blocks_t"][:, k_old:k_new].copy_(sub["keys_blocks_t"])
    arena["invalid_blocks_i8"][:, k_old:k_new].copy_(sub["invalid_blocks_i8"])
    arena["assigns_blocks"][:, :, k_old:k_new].copy_(
        sub["assigns_blocks"].to(arena["assigns_blocks"].dtype) + k_old
    )
    if "values_blocks_f16" in arena and "values_blocks_f16" in sub:
        arena["values_blocks_f16"][:, k_old:k_new].copy_(sub["values_blocks_f16"])

    if n_pad_new < int(arena["N_pad"]):
        arena["invalid_mask"][:, n_pad_new:].fill_(True)
        arena["invalid_blocks_i8"][:, k_new:].fill_(1)

    arena["K_used"] = k_new
    arena["N_used"] = n_new
    arena["N"] = n_new
    arena["K"] = int(arena["K_cap"])
    arena["anchor_subspace"] = state.get("anchor_subspace", anchor_subspace)
    invalidate_attention_layout_caches(arena)

    new_keys, new_values = maybe_merged(
        old_keys, buffer_keys, old_values, buffer_values, return_merged
    )
    return arena, new_keys, new_values


# ─────────────────────────────────────────────────────────────────────────
# Async (split) merge for parallel updates.
#
# merge_arena_async writes data into the arena's [k_old:k_new] slice but
# leaves the invalid flags untouched (they were initialized to "invalid").
# That way, attention running concurrently on a different stream observes
# the new range as still-invalid and skips it (output remains correct).
#
# After update_done_event has fired and attn_stream has waited on it,
# apply_pending_publish() runs on attn_stream to flip the invalid flags
# (= publish), and to invalidate the layout cache so the next attend()
# rebuilds the cached fp16 / fp32 copies that mirror the arena.
# ─────────────────────────────────────────────────────────────────────────


def _ensure_arena(state: dict, k_sub: int, bf: int) -> tuple[dict, int, int]:
    if "K_cap" in state:
        arena = state if int(state["K_cap"]) >= int(state["K_used"]) + k_sub else _grow_arena(
            state, int(state["K_used"]) + k_sub, bf
        )
        return arena, int(arena["K_used"]), int(arena["N_used"])
    k_old_base = int(state["K"])
    arena, k_old, n_old = _arena_from_base(state, k_old_base + k_sub, bf)
    return arena, k_old, n_old


def merge_arena_async(
    state: dict,
    sub: dict,
    old_keys: torch.Tensor,
    buffer_keys: torch.Tensor,
    old_values: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    bf: int,
    anchor_subspace: int,
    return_merged: bool,
) -> tuple[dict, torch.Tensor | None, torch.Tensor | None, dict]:
    """Write the merged data into the arena WITHOUT publishing the new range.

    Returns ``(arena, new_keys, new_values, pending_publish)``. The
    ``pending_publish`` dict carries everything needed by
    ``apply_pending_publish`` to (a) flip the invalid flags and (b) update
    the bookkeeping fields. Caller must invoke that AFTER waiting on the
    update_done_event from a stream the new state will be read on.
    """
    k_sub = int(sub["K"])
    arena, k_old, n_old = _ensure_arena(state, k_sub, bf)

    k_new = k_old + k_sub
    n_sub = int(sub["N"])
    n_new = n_old + n_sub
    n_pad_old = k_old * bf
    n_pad_new = k_new * bf

    # Data writes into the unused tail of the arena. The matching slots in
    # invalid_blocks_i8 / invalid_mask are already 1 (filled by arena init or
    # by the previous publish), so attention treats this range as nonexistent.
    arena["keys_reord"][:, n_pad_old:n_pad_new].copy_(sub["keys_reord"])
    arena["reorder_perm"][:, n_pad_old:n_pad_new].copy_(sub["reorder_perm"] + n_old)
    for dst, src in zip(arena["assigns_reord"], sub["assigns_reord"]):
        dst[:, n_pad_old:n_pad_new].copy_(src + k_old)

    for c_dst, c_src in zip(arena["centers"], sub["centers"]):
        c_dst[:, k_old:k_new].copy_(c_src)
    for r_dst, r_src in zip(arena["radii"], sub["radii"]):
        r_dst[:, k_old:k_new].copy_(r_src)

    arena["keys_blocks_t"][:, k_old:k_new].copy_(sub["keys_blocks_t"])
    arena["assigns_blocks"][:, :, k_old:k_new].copy_(
        sub["assigns_blocks"].to(arena["assigns_blocks"].dtype) + k_old
    )
    if "values_blocks_f16" in arena and "values_blocks_f16" in sub:
        arena["values_blocks_f16"][:, k_old:k_new].copy_(sub["values_blocks_f16"])

    new_keys, new_values = maybe_merged(
        old_keys, buffer_keys, old_values, buffer_values, return_merged
    )

    pending = {
        "arena": arena,
        "k_old": k_old,
        "k_new": k_new,
        "n_old": n_old,
        "n_new": n_new,
        "n_pad_old": n_pad_old,
        "n_pad_new": n_pad_new,
        "anchor_subspace": state.get("anchor_subspace", anchor_subspace),
        "sub_invalid_mask": sub["invalid_mask"],          # (h_kv, n_pad_sub) bool
        "sub_invalid_blocks_i8": sub["invalid_blocks_i8"],  # (h_kv, k_sub, bf) int8
    }
    return arena, new_keys, new_values, pending


def apply_pending_publish(pending: dict) -> dict:
    """Flip invalid flags and bump K_used/N_used to make the new range visible.

    Run this on the attention stream AFTER it has waited on the update_done
    event for ``pending``. Mutates the arena in place and invalidates cached
    attention layouts so the next attend() rebuilds them.
    """
    arena = pending["arena"]
    k_old = int(pending["k_old"])
    k_new = int(pending["k_new"])
    n_pad_old = int(pending["n_pad_old"])
    n_pad_new = int(pending["n_pad_new"])

    arena["invalid_mask"][:, n_pad_old:n_pad_new].copy_(pending["sub_invalid_mask"])
    arena["invalid_blocks_i8"][:, k_old:k_new].copy_(pending["sub_invalid_blocks_i8"])

    if n_pad_new < int(arena["N_pad"]):
        arena["invalid_mask"][:, n_pad_new:].fill_(True)
        arena["invalid_blocks_i8"][:, k_new:].fill_(1)

    arena["K_used"] = k_new
    arena["N_used"] = int(pending["n_new"])
    arena["N"] = int(pending["n_new"])
    arena["K"] = int(arena["K_cap"])
    arena["anchor_subspace"] = pending["anchor_subspace"]
    invalidate_attention_layout_caches(arena)
    return arena
