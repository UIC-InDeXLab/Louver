"""search_v15.0 — v11 + FP16 keys + fused q-pack.

Improvements over v11:
  1. Keys stored in float16 → 2x less HBM bandwidth for key loads.
     tl.dot with fp16 inputs uses higher-throughput tensor core path.
     Accumulator stays f32 — no precision loss in output.
  2. q packing done inline (view+transpose) instead of calling
     pack_query_subspaces (eliminates 3 torch launches).
  3. Cluster-pass kernel takes q directly (reads subspace slices from q)
     instead of requiring a pre-packed q_packed buffer.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from .._search_triton import triton_fused_cluster_pass
from .._search_utils import _mapping_mode, buffer_dot

KERNEL_VERSION = "v15.0"
_PARENTS_PER_PROG = 8


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _get_or_make_fp16_block_pack(
    state: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fp16 keys_blocks_t and other block-packed tensors."""
    cache = state.setdefault("_search_v15_0_block_pack", {})
    keys_reord = state["keys_reord"]
    cache_key = (
        keys_reord.data_ptr(),
        tuple(keys_reord.shape),
        state["K"],
        state["bf"],
        len(state["assigns_reord"]),
    )
    if cache.get("key") == cache_key:
        p = cache["pack"]
        return p["keys_blocks_t_f16"], p["assigns_blocks"], p["invalid_blocks_i8"]

    h_kv, _, d = keys_reord.shape
    k = state["K"]
    bf = state["bf"]
    s = len(state["assigns_reord"])

    pack = {
        "keys_blocks_t_f16": (
            keys_reord.view(h_kv, k, bf, d)
            .permute(0, 1, 3, 2)
            .to(torch.float16)
            .contiguous()
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
    return pack["keys_blocks_t_f16"], pack["assigns_blocks"], pack["invalid_blocks_i8"]


if HAS_TRITON:

    @triton.jit
    def _fp16_fused_kernel(
        Q_ptr,              # (H_q, D) f32
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
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

        # --- Gate: anchor cluster pass
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

        # --- Gate: non-anchor subspaces
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

        # --- Dot product: load q as f16, load keys (already f16), tl.dot
        d_range = tl.arange(0, D)
        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_full = q_full_f32.to(tl.float16)

        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :],
            other=0.0,
        )
        # FP16 tl.dot with f32 accumulator
        acc = tl.dot(q_full, keys_tile)  # (GROUPS_POW, PARENTS_PER_PROG*BF) f32

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def _get_layout_v15(state: dict, q_head_to_kv: torch.Tensor | None, q: torch.Tensor):
    keys_reord: torch.Tensor = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, mapping_sig = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_f16, assigns_blocks, invalid_blocks = _get_or_make_fp16_block_pack(state)

    cache = state.setdefault("_search_v15_0_cache", {})
    cache_key = (
        mode, groups, mapping_sig,
        keys_reord.data_ptr(), tuple(keys_reord.shape),
        keys_f16.data_ptr(),
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


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v15 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v15 requires build_v2-style state")

    layout = _get_layout_v15(state, q_head_to_kv, q)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    # --- Idea 2: fused q-pack. Single view+transpose, no pack_query_subspaces.
    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        q_packed = q.new_zeros(s, h_q, max_d)
        for si, (s0, e0) in enumerate(layout["dim_slices"]):
            q_packed[si, :, : e0 - s0] = q[:, s0:e0]
        q_packed = q_packed.contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed, q_norm, th_packed,
        layout["centers"], layout["radii"], layout["groups"],
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
    _fp16_fused_kernel[grid](
        q.contiguous(),
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        cluster_pass_flat,
        layout["invalid_blocks_i8"],
        out,
        h_q, h_kv, k, n_pad,
        ANCHOR_S=anchor_s,
        D=d, BF=bf,
        GROUPS=groups, GROUPS_POW=groups_pow,
        S=s, PARENTS_PER_PROG=_PARENTS_PER_PROG,
        num_warps=4,
    )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
