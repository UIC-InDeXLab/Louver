"""search_v11.0 — batched-parent anchor-block search.

Improvements over search_v10.0:
  1. Gate on all subspaces before loading keys / doing tl.dot.
  2. Consume block-packed build state: keys_blocks_t (H, K, D, BF) and
     assigns_blocks (S, H, K, BF).
  3. Batch several parents per Triton program so one query block load is
     reused across multiple bf blocks.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_triton import triton_fused_cluster_pass
from ._search_utils import _mapping_mode, buffer_dot, pack_query_subspaces

KERNEL_VERSION = "v11.0"
_PARENTS_PER_PROG = 8


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


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
    cache = state.setdefault("_search_v11_0_block_pack", {})
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
    def _fused_anchor_parent_batch_kernel(
        Q_ptr,              # (H_q, D)
        KeysBlocksT_ptr,    # (H_kv, K, D, BF)
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        ClusterPass_ptr,    # (S, H_q, K) int8
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        Out_ptr,            # (H_q, N_pad) f32
        H_Q,
        H_KV,
        K,
        N_PAD,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        parent_block = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        parent_idx = parent_block * PARENTS_PER_PROG + parent_rel
        col_valid = parent_idx < K
        parent_idx_safe = tl.where(col_valid, parent_idx, 0)
        child_idx = parent_idx_safe * BF + child_rel

        out_offs = hq_vec[:, None] * N_PAD + child_idx[None, :]
        out_mask = g_valid[:, None] & col_valid[None, :]

        anchor_pass = tl.load(
            ClusterPass_ptr + (ANCHOR_S * H_Q + hq_vec[:, None]) * K + parent_idx_safe[None, :],
            mask=out_mask,
            other=0,
        )
        survive = (anchor_pass != 0) & out_mask

        inv = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
            mask=col_valid,
            other=1,
        )
        survive = survive & (inv[None, :] == 0)

        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx_safe) * BF
                    + child_rel,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq_vec[:, None]) * K + assign[None, :],
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

        live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        d_range = tl.arange(0, D)
        q_full = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )

        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :],
            other=0.0,
        )
        acc = tl.dot(q_full, keys_tile, allow_tf32=True)

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def _get_layout_v11(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_blocks_src, assigns_blocks_src, invalid_blocks_src = _get_or_make_block_pack(state)

    cache = state.setdefault("_search_v11_0_cache", {})
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

    centers_src = state["centers"]
    radii_src = state["radii"]

    if mode == "expanded":
        assert q_head_to_kv is not None
        centers_src = [t.index_select(0, q_head_to_kv).contiguous() for t in centers_src]
        radii_src = [t.index_select(0, q_head_to_kv).contiguous() for t in radii_src]
        keys_blocks_base = keys_blocks_src.index_select(0, q_head_to_kv).contiguous()
        assigns_blocks_base = assigns_blocks_src.index_select(1, q_head_to_kv).contiguous()
        invalid_blocks_base = invalid_blocks_src.index_select(0, q_head_to_kv).contiguous()
        h_kv_eff = h_q
    else:
        keys_blocks_base = keys_blocks_src
        assigns_blocks_base = assigns_blocks_src
        invalid_blocks_base = invalid_blocks_src
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
        "K": k,
        "bf": bf,
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
    keys_children: torch.Tensor,  # ignored — keys come from build state
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v11 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v11 requires build_v2-style state")

    layout = _get_layout_v11(state, q_head_to_kv, q)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        _, q_packed, _ = pack_query_subspaces(q, layout)
        q_packed = q_packed.reshape(s, h_q, max_d).contiguous()

    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()
    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed,
        q_norm,
        th_packed,
        layout["centers"],
        layout["radii"],
        layout["groups"],
    )

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    out = torch.empty(h_q, n_pad, device=q.device, dtype=torch.float32)
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))
    _fused_anchor_parent_batch_kernel[grid](
        q.contiguous(),
        layout["keys_blocks_t"],
        layout["assigns_blocks"],
        cluster_pass_flat,
        layout["invalid_blocks_i8"],
        out,
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=s,
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
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
