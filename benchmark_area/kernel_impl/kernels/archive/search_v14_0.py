"""search_v14.0 — two-level anchor hierarchy with bitmask gating.

Search pipeline:
  1. Compute super-anchor pass as one int32 bitmask per (kv head, super-parent).
  2. In the Triton kernel, skip whole super-parent groups whose mask is zero.
  3. For surviving super-parents, compute parent-level anchor bounds on demand.
  4. Evaluate non-anchor upper bounds on demand and score surviving children.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_utils import _mapping_mode, buffer_dot, pack_query_subspaces

KERNEL_VERSION = "v14.0"


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _anchor_pass_bitmask(
    q_sub: torch.Tensor,
    th_sub: torch.Tensor,
    centers_sub: torch.Tensor,
    radii_sub: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    h_kv = centers_sub.shape[0]
    q_grouped = q_sub.view(h_kv, groups, q_sub.shape[-1])
    q_norm = q_grouped.norm(dim=-1)
    center_dots = torch.bmm(q_grouped, centers_sub.transpose(1, 2))
    ub = center_dots + radii_sub[:, None, :] * q_norm[:, :, None]
    th_grouped = th_sub.view(h_kv, groups, 1)
    passed = ub >= th_grouped

    if groups > 31:
        raise RuntimeError(f"search_v14 only supports up to 31 query groups, got {groups}")
    weights = (1 << torch.arange(groups, device=q_sub.device, dtype=torch.int32)).view(
        1, groups, 1
    )
    return (passed.to(torch.int32) * weights).sum(dim=1).contiguous()


def _get_or_make_block_pack(state: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        "keys_blocks_t" in state
        and "assigns_blocks" in state
        and "invalid_blocks_i8" in state
    ):
        return (
            state["keys_blocks_t"],
            state["assigns_blocks"],
            state["invalid_blocks_i8"],
        )

    keys_reord = state["keys_reord"]
    cache = state.setdefault("_search_v14_0_block_pack", {})
    cache_key = (
        keys_reord.data_ptr(),
        tuple(keys_reord.shape),
        state["K"],
        state["bf"],
        len(state["assigns_reord"]),
    )
    if cache.get("key") == cache_key:
        pack = cache["pack"]
        return pack["keys_blocks_t"], pack["assigns_blocks"], pack["invalid_blocks_i8"]

    h_kv, _, d = keys_reord.shape
    k = state["K"]
    bf = state["bf"]
    s = len(state["assigns_reord"])

    pack = {
        "keys_blocks_t": (
            keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).contiguous()
        ),
        "assigns_blocks": (
            torch.stack(state["assigns_reord"], dim=0)
            .to(_assign_dtype(k))
            .view(s, h_kv, k, bf)
            .contiguous()
        ),
        "invalid_blocks_i8": (
            state["invalid_mask"].view(h_kv, k, bf).to(torch.int8).contiguous()
        ),
    }
    cache["key"] = cache_key
    cache["pack"] = pack
    return pack["keys_blocks_t"], pack["assigns_blocks"], pack["invalid_blocks_i8"]


if HAS_TRITON:

    @triton.jit
    def _two_level_anchor_kernel(
        Q_ptr,                  # (H_q, D)
        QPacked_ptr,            # (S, H_q, MAX_D)
        QNorm_ptr,              # (S, H_q)
        Th_ptr,                 # (S, H_q)
        KeysBlocksT_ptr,        # (H_kv, K, D, BF)
        AssignsBlocks_ptr,      # (S, H_kv, K, BF)
        Centers_ptr,            # (S, H_kv, K, MAX_D)
        Radii_ptr,              # (S, H_kv, K)
        SuperMask_ptr,          # (H_kv, K2) int32
        SuperParentIds_ptr,     # (H_kv, K2, SUPER_BF) int32
        SuperParentInv_ptr,     # (H_kv, K2, SUPER_BF) int8
        InvalidBlocks_ptr,      # (H_kv, K, BF) int8
        Out_ptr,                # (H_q, N_pad)
        H_Q,
        H_KV,
        K,
        K2,
        N_PAD,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        MAX_D: tl.constexpr,
        SUPER_BF: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        super_idx = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        bit_weights = (1 << g_range).to(tl.int32)
        hq_vec = kvh * GROUPS + g_range

        cols = tl.arange(0, SUPER_BF * BF)
        slot_rel = cols // BF
        child_rel = cols % BF
        parent_idx = tl.load(
            SuperParentIds_ptr + ((kvh * K2 + super_idx) * SUPER_BF + slot_rel)
        ).to(tl.int32)
        parent_inv_col = tl.load(
            SuperParentInv_ptr + ((kvh * K2 + super_idx) * SUPER_BF + slot_rel)
        )
        col_valid = parent_inv_col == 0
        child_idx = parent_idx * BF + child_rel

        out_offs = hq_vec[:, None] * N_PAD + child_idx[None, :]
        out_mask = g_valid[:, None] & col_valid[None, :]

        super_mask = tl.load(SuperMask_ptr + kvh * K2 + super_idx).to(tl.int32)
        if super_mask == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        d_sub = tl.arange(0, MAX_D)
        q_anchor = tl.load(
            QPacked_ptr + (ANCHOR_S * H_Q + hq_vec[:, None]) * MAX_D + d_sub[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        qn_anchor = tl.load(
            QNorm_ptr + ANCHOR_S * H_Q + hq_vec,
            mask=g_valid,
            other=0.0,
        )
        th_anchor = tl.load(
            Th_ptr + ANCHOR_S * H_Q + hq_vec,
            mask=g_valid,
            other=float("inf"),
        )
        center_anchor = tl.load(
            Centers_ptr
            + ((ANCHOR_S * H_KV + kvh) * K + parent_idx[:, None]) * MAX_D
            + d_sub[None, :],
            mask=col_valid[:, None],
            other=0.0,
        )
        cdot_anchor = tl.sum(q_anchor[:, None, :] * center_anchor[None, :, :], axis=2)
        radii_anchor = tl.load(
            Radii_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx,
            mask=col_valid,
            other=0.0,
        )
        ub_anchor = cdot_anchor + qn_anchor[:, None] * radii_anchor[None, :]
        passed_anchor = (ub_anchor >= th_anchor[:, None]).to(tl.int32) * bit_weights[:, None]
        child_mask = tl.sum(passed_anchor, axis=0) & super_mask
        child_mask = tl.where(col_valid, child_mask, 0)

        inv_child = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx) * BF + child_rel),
            mask=col_valid,
            other=1,
        )
        child_mask = tl.where((inv_child == 0) & col_valid, child_mask, 0)

        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                q_sub = tl.load(
                    QPacked_ptr + (s * H_Q + hq_vec[:, None]) * MAX_D + d_sub[None, :],
                    mask=g_valid[:, None],
                    other=0.0,
                )
                qn_sub = tl.load(QNorm_ptr + s * H_Q + hq_vec, mask=g_valid, other=0.0)
                th_sub = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx) * BF
                    + child_rel,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)
                center_tile = tl.load(
                    Centers_ptr
                    + ((s * H_KV + kvh) * K + assign[:, None]) * MAX_D
                    + d_sub[None, :],
                    mask=col_valid[:, None],
                    other=0.0,
                )
                cdot = tl.sum(q_sub[:, None, :] * center_tile[None, :, :], axis=2)
                radii = tl.load(
                    Radii_ptr + (s * H_KV + kvh) * K + assign,
                    mask=col_valid,
                    other=0.0,
                )
                ub = cdot + qn_sub[:, None] * radii[None, :]
                passed = (ub >= th_sub[:, None]).to(tl.int32) * bit_weights[:, None]
                pass_mask = tl.sum(passed, axis=0)
                child_mask = child_mask & pass_mask

        live_cols = child_mask != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        d_full = tl.arange(0, D)
        q_full = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_full[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx[None, :]) * D + d_full[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :],
            other=0.0,
        )
        acc = tl.dot(q_full, keys_tile, allow_tf32=True)

        survive2d = out_mask & ((((child_mask[None, :] >> g_range[:, None]) & 1) != 0))
        out = tl.where(survive2d, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def _get_layout_v14(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_blocks_src, assigns_blocks_src, invalid_blocks_src = _get_or_make_block_pack(state)
    super_centers_src = state["super_centers_anchor"]
    super_radii_src = state["super_radii_anchor"]
    super_parent_ids_src = state["super_parent_ids"]
    super_parent_invalid_src = state["super_parent_invalid_i8"]

    cache = state.setdefault("_search_v14_0_cache", {})
    cache_key = (
        mode,
        groups,
        mapping_sig,
        keys_reord.data_ptr(),
        tuple(keys_reord.shape),
        keys_blocks_src.data_ptr(),
        tuple(keys_blocks_src.shape),
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
    k2 = state["anchor_super_k"]
    super_bf = state["anchor_super_bf"]

    centers_src = state["centers"]
    radii_src = state["radii"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        keys_blocks_base = keys_blocks_src.index_select(0, q_head_to_kv).contiguous()
        assigns_blocks_base = assigns_blocks_src.index_select(1, q_head_to_kv).contiguous()
        invalid_blocks_base = invalid_blocks_src.index_select(0, q_head_to_kv).contiguous()
        super_centers_base = super_centers_src.index_select(0, q_head_to_kv).contiguous()
        super_radii_base = super_radii_src.index_select(0, q_head_to_kv).contiguous()
        super_parent_ids_base = super_parent_ids_src.index_select(0, q_head_to_kv).contiguous()
        super_parent_invalid_base = super_parent_invalid_src.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        keys_blocks_base = keys_blocks_src
        assigns_blocks_base = assigns_blocks_src
        invalid_blocks_base = invalid_blocks_src
        super_centers_base = super_centers_src
        super_radii_base = super_radii_src
        super_parent_ids_base = super_parent_ids_src
        super_parent_invalid_base = super_parent_invalid_src
        h_kv_eff = h_kv

    centers = torch.zeros(
        s,
        h_kv_eff,
        k,
        max_d,
        device=keys_blocks_base.device,
        dtype=centers_src[0].dtype,
    )
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
        "keys_blocks_t": keys_blocks_base,
        "assigns_blocks": assigns_blocks_base,
        "invalid_blocks_i8": invalid_blocks_base,
        "super_centers_anchor": super_centers_base,
        "super_radii_anchor": super_radii_base,
        "super_parent_ids": super_parent_ids_base,
        "super_parent_invalid_i8": super_parent_invalid_base,
        "K": k,
        "K2": k2,
        "bf": bf,
        "super_bf": super_bf,
        "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }
    cache["key"] = cache_key
    cache["layout"] = layout
    return layout


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,  # ignored
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v14 requires Triton")
    if "super_centers_anchor" not in state:
        raise RuntimeError("search_v14 requires build_v2_3 state")

    layout = _get_layout_v14(state, q_head_to_kv, q)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]
    anchor_s = layout["anchor_subspace"]

    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        _, q_packed, _ = pack_query_subspaces(q, layout)
        q_packed = q_packed.reshape(s, h_q, max_d).contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    s0, e0 = layout["dim_slices"][anchor_s]
    super_mask = _anchor_pass_bitmask(
        q_sub=q[:, s0:e0].contiguous(),
        th_sub=th_packed[anchor_s].contiguous(),
        centers_sub=layout["super_centers_anchor"].contiguous(),
        radii_sub=layout["super_radii_anchor"].contiguous(),
        groups=layout["groups"],
    )

    h_kv = layout["base_heads"]
    k = layout["K"]
    k2 = layout["K2"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    super_bf = layout["super_bf"]

    out = torch.empty(h_q, n_pad, device=q.device, dtype=torch.float32)
    grid = (h_kv, k2)
    _two_level_anchor_kernel[grid](
        q.contiguous(),
        q_packed,
        q_norm,
        th_packed,
        layout["keys_blocks_t"],
        layout["assigns_blocks"],
        layout["centers"],
        layout["radii"],
        super_mask,
        layout["super_parent_ids"],
        layout["super_parent_invalid_i8"],
        layout["invalid_blocks_i8"],
        out,
        h_q,
        h_kv,
        k,
        k2,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=s,
        MAX_D=max_d,
        SUPER_BF=super_bf,
        num_warps=4,
    )

    buf_layout_shim = {
        "mode": layout["mode"],
        "groups": groups,
        "base_heads": h_kv,
    }
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_layout_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
