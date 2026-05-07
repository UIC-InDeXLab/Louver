"""search_v17.0 — v15 + bitmask cluster_pass (idea 4 only, no compaction).

Idea 4: Pack all GROUPS pass bits into one int32 per (s, h_kv, k).
  cluster_pass shrinks from (S, H_q, K) int8 = 350KB to (S, H_kv, K)
  int32 = 50KB. Non-anchor gate does ONE load per child (not per
  group per child), then extracts bits. Fewer L2 fetches, possibly
  fits in L1 entirely.

Keeps v15's batched tl.dot structure and full grid (no compaction).
v17.1 adds compaction on top.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_utils import _mapping_mode, buffer_dot

KERNEL_VERSION = "v17.0"
_PARENTS_PER_PROG = 8


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _assign_dtype(k: int) -> torch.dtype:
    return torch.int16 if k < 32768 else torch.int32


def _get_or_make_fp16_block_pack(state: dict):
    cache = state.setdefault("_search_v17_0_block_pack", {})
    keys_reord = state["keys_reord"]
    cache_key = (keys_reord.data_ptr(), tuple(keys_reord.shape), state["K"], state["bf"], len(state["assigns_reord"]))
    if cache.get("key") == cache_key:
        p = cache["pack"]
        return p["keys_blocks_t_f16"], p["assigns_blocks"], p["invalid_blocks_i8"]

    h_kv, _, d = keys_reord.shape
    k = state["K"]
    bf = state["bf"]
    s = len(state["assigns_reord"])
    pack = {
        "keys_blocks_t_f16": keys_reord.view(h_kv, k, bf, d).permute(0, 1, 3, 2).to(torch.float16).contiguous(),
        "assigns_blocks": torch.stack(state["assigns_reord"], dim=0).to(_assign_dtype(k)).view(s, h_kv, k, bf).contiguous(),
        "invalid_blocks_i8": state["invalid_mask"].view(h_kv, k, bf).to(torch.int8).contiguous(),
    }
    cache["key"] = cache_key
    cache["pack"] = pack
    return pack["keys_blocks_t_f16"], pack["assigns_blocks"], pack["invalid_blocks_i8"]


if HAS_TRITON:

    @triton.jit
    def _fused_cluster_pass_bitmask_kernel(
        QPacked_ptr,    # (S, H_q, MAX_D)        f32
        QNorm_ptr,      # (S, H_q)               f32
        Th_ptr,         # (S, H_q)               f32
        Centers_ptr,    # (S, H_kv, K, MAX_D)    f32
        Radii_ptr,      # (S, H_kv, K)           f32
        Out_ptr,        # (S, H_kv, K)           i32 bitmask
        H_Q,
        H_KV,
        K,
        S: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        MAX_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        s = tl.program_id(0)
        kvh = tl.program_id(1)
        k0 = tl.program_id(2) * BLOCK_K

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        d_range = tl.arange(0, MAX_D)
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        hq_vec = kvh * GROUPS + g_range

        qp = tl.load(
            QPacked_ptr + (s * H_Q + hq_vec[:, None]) * MAX_D + d_range[None, :],
            mask=g_valid[:, None], other=0.0,
        )
        qn = tl.load(QNorm_ptr + s * H_Q + hq_vec, mask=g_valid, other=0.0)
        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)
        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]) & g_valid[:, None]

        bit_weights = (1 << g_range).to(tl.int32)
        bitmask = tl.sum(passed.to(tl.int32) * bit_weights[:, None], axis=0)

        tl.store(Out_ptr + (s * H_KV + kvh) * K + k_range, bitmask, mask=k_mask)


    @triton.jit
    def _bitmask_batched_kernel(
        Q_ptr,              # (H_q, D) f32
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        BitmaskPass_ptr,    # (S, H_kv, K) i32
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        CompactIds_ptr,     # (H_kv, max_survive) int32 — or None if no compaction
        Counts_ptr,         # (H_kv,) int32 — or None
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
        USE_COMPACT: tl.constexpr,
        MAX_SURVIVE: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        parent_block = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        # Determine parent indices
        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        if USE_COMPACT:
            count = tl.load(Counts_ptr + kvh)
            block_start = parent_block * PARENTS_PER_PROG
            # Early-exit if entire block is past compact end
            if block_start >= count:
                return
            compact_base = block_start + parent_rel
            col_valid = compact_base < count
            compact_safe = tl.where(col_valid, compact_base, 0)
            parent_idx = tl.load(
                CompactIds_ptr + kvh * MAX_SURVIVE + compact_safe,
                mask=col_valid, other=0,
            ).to(tl.int32)
        else:
            parent_idx = parent_block * PARENTS_PER_PROG + parent_rel
            col_valid = parent_idx < K

        parent_idx_safe = tl.where(col_valid, parent_idx, 0)
        child_idx = parent_idx_safe * BF + child_rel

        out_offs = hq_vec[:, None] * N_PAD + child_idx[None, :]
        out_mask = g_valid[:, None] & col_valid[None, :]

        # --- Gate: anchor bitmask ---
        anchor_bm = tl.load(
            BitmaskPass_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx_safe,
            mask=col_valid, other=0,
        ).to(tl.int32)
        # Expand per parent to per (group, child): does this group pass anchor for this parent?
        anchor_group_pass = ((anchor_bm[None, :] >> g_range[:, None]) & 1) != 0
        survive = anchor_group_pass & out_mask

        # Invalid mask
        inv = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
            mask=col_valid, other=1,
        )
        survive = survive & (inv[None, :] == 0)

        # --- Gate: non-anchor subspaces (bitmask lookup) ---
        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx_safe) * BF
                    + child_rel,
                    mask=col_valid, other=0,
                ).to(tl.int32)
                # ONE load per child (not per group per child!)
                bm = tl.load(
                    BitmaskPass_ptr + (s * H_KV + kvh) * K + assign,
                    mask=col_valid, other=0,
                ).to(tl.int32)
                # Extract per-group bits
                child_pass = ((bm[None, :] >> g_range[:, None]) & 1) != 0
                survive = survive & child_pass

        live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        # --- Dot product: FP16 tl.dot ---
        d_range = tl.arange(0, D)
        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None], other=0.0,
        )
        q_full = q_full_f32.to(tl.float16)

        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :], other=0.0,
        )

        acc = tl.dot(q_full, keys_tile)

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def _fused_cluster_pass_bitmask(
    q_packed: torch.Tensor,
    q_norm: torch.Tensor,
    th: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Returns int32 bitmask cluster_pass (S, H_kv, K)."""
    S, H_q, max_d = q_packed.shape
    H_kv = centers.shape[1]
    K = centers.shape[2]
    out = torch.empty(S, H_kv, K, device=q_packed.device, dtype=torch.int32)

    groups_pow = max(_next_pow2(groups), 8)
    block_k = 64

    grid = (S, H_kv, triton.cdiv(K, block_k))
    _fused_cluster_pass_bitmask_kernel[grid](
        q_packed, q_norm, th, centers, radii, out,
        H_q, H_kv, K,
        S=S, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out


def _compact_anchor_parents(
    bitmask: torch.Tensor,
    invalid_blocks: torch.Tensor,
    anchor_s: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compact parents that pass anchor for at least one group.

    Returns (compact_ids[H_kv, K], counts[H_kv], K).
    Skips the .item() sync — kernel uses Counts_ptr to know when to stop.
    Worst-case width K means grid is same as no-compaction; the win is
    that surviving parents are densely packed at the front.
    """
    anchor_any = bitmask[anchor_s] != 0
    all_invalid = (invalid_blocks != 0).all(dim=-1)
    survive = anchor_any & ~all_invalid

    counts = survive.sum(dim=1).to(torch.int32)
    sort_key = survive.to(torch.int32)
    compact_ids = sort_key.argsort(dim=1, descending=True, stable=True).to(torch.int32).contiguous()
    return compact_ids, counts, compact_ids.shape[1]


def _get_layout_v17(state, q_head_to_kv, q):
    keys_reord = state["keys_reord"]
    h_kv = keys_reord.shape[0]
    h_q = int(q.shape[0]) if q_head_to_kv is None else int(q_head_to_kv.shape[0])
    mode, groups, _ = _mapping_mode(q_head_to_kv, h_q, h_kv)

    keys_f16, assigns_blocks, invalid_blocks = _get_or_make_fp16_block_pack(state)

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

    return {
        "mode": mode, "groups": groups, "base_heads": h_kv_eff,
        "num_subspaces": s, "max_d": max_d, "dim_slices": tuple(dim_slices),
        "centers": centers.contiguous(),
        "radii": torch.stack(radii_src, dim=0).contiguous(),
        "keys_blocks_t_f16": keys_f16,
        "assigns_blocks": assigns_blocks,
        "invalid_blocks_i8": invalid_blocks,
        "K": k, "bf": bf, "N_pad": n_pad,
        "anchor_subspace": state.get("anchor_subspace", 0),
    }


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v17 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v17 requires build_v2-style state")

    layout = _get_layout_v17(state, q_head_to_kv, q)
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
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    # Idea 4: bitmask cluster_pass → (S, H_kv, K) int32
    bitmask_pass = _fused_cluster_pass_bitmask(
        q_packed, q_norm, th_packed,
        layout["centers"], layout["radii"], groups,
    )

    out = torch.empty(h_q, n_pad, device=q.device, dtype=torch.float32)
    # Dummy tensors for non-compaction path (kernel ignores via constexpr)
    dummy_compact = torch.zeros(1, device=q.device, dtype=torch.int32)
    dummy_counts = torch.zeros(1, device=q.device, dtype=torch.int32)

    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))
    _bitmask_batched_kernel[grid](
        q.contiguous(),
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        bitmask_pass,
        layout["invalid_blocks_i8"],
        dummy_compact,
        dummy_counts,
        out,
        h_q, h_kv, k, n_pad,
        ANCHOR_S=anchor_s,
        D=d, BF=bf,
        GROUPS=groups, GROUPS_POW=groups_pow,
        S=s, PARENTS_PER_PROG=_PARENTS_PER_PROG,
        USE_COMPACT=False,
        MAX_SURVIVE=1,
        num_warps=4,
    )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
