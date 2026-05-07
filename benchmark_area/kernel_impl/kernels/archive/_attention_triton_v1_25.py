"""v1.25 attention kernel — v1.20 with the anchor-subspace gate fused inline.

Design change vs v1.20:
  * The cluster_pass kernel no longer computes the anchor subspace. Its output
    for the anchor slot is unused (it's still allocated as a single contiguous
    (S, H_q, K) tensor for pointer-math simplicity, just not written for
    anchor).
  * The main attention kernel derives the anchor gate on-the-fly from
    `centers[ANCHOR_S, kvh, parent, :]`, `radii[ANCHOR_S, kvh, parent]`,
    the anchor slice of q, and `th[ANCHOR_S, hq]` — saves one HBM round-trip
    per parent per call.

The non-anchor gates still use the cluster_pass table because their lookups
go through `assigns[s, kvh, parent, child]`, a scatter across parent indices
that's cheaper to service from the compact (S, H_q, K) table than to recompute.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_cluster_pass_rawq_skip_anchor_kernel(
        Q_ptr,             # (H_q, D)             f32
        DimOffsets_ptr,    # (S,)                 i32
        DimWidths_ptr,     # (S,)                 i32
        Th_ptr,            # (S, H_q)             f32
        Centers_ptr,       # (S, H_kv, K, MAX_D)  f32
        Radii_ptr,         # (S, H_kv, K)         f32
        Out_ptr,           # (S, H_q, K)          i8 — anchor row untouched
        H_Q,
        H_KV,
        K,
        D: tl.constexpr,
        S: tl.constexpr,
        ANCHOR_S: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        MAX_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        s = tl.program_id(0)
        # Skip the anchor subspace entirely — the main attn kernel computes it
        # inline. Using an early return avoids both the compute and the write.
        if s == ANCHOR_S:
            return

        kvh = tl.program_id(1)
        k0 = tl.program_id(2) * BLOCK_K

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        d_range = tl.arange(0, MAX_D)
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        hq_vec = kvh * GROUPS + g_range

        dim_off = tl.load(DimOffsets_ptr + s)
        width = tl.load(DimWidths_ptr + s)
        d_valid = d_range < width

        q_load_mask = g_valid[:, None] & d_valid[None, :]
        qp = tl.load(
            Q_ptr + hq_vec[:, None] * D + (dim_off + d_range)[None, :],
            mask=q_load_mask,
            other=0.0,
        )
        qn = tl.sqrt(tl.sum(qp * qp, axis=1))

        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)

        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int8)

        out_offs = (s * H_Q + hq_vec[:, None]) * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)


def triton_fused_cluster_pass_rawq_skip_anchor(
    q: torch.Tensor,
    th: torch.Tensor,
    dim_offsets: torch.Tensor,
    dim_widths: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    groups: int,
    anchor_s: int,
    out: torch.Tensor,
) -> torch.Tensor:
    h_q, d = q.shape
    s, h_kv, k, max_d = centers.shape

    groups_pow = 1
    while groups_pow < max(groups, 8):
        groups_pow *= 2
    block_k = 64

    grid = (s, h_kv, triton.cdiv(k, block_k))
    _fused_cluster_pass_rawq_skip_anchor_kernel[grid](
        q, dim_offsets, dim_widths, th, centers, radii, out,
        h_q, h_kv, k,
        D=d, S=s, ANCHOR_S=anchor_s,
        GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out


if HAS_TRITON:

    @triton.jit
    def _fused_attn_index_anchor_inline_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        AssignsBlocks_ptr,
        ClusterPass_ptr,
        InvalidBlocks_ptr,
        Centers_ptr,             # (S, H_kv, K, MAX_D)  f32
        Radii_ptr,               # (S, H_kv, K)         f32
        Th_ptr,                  # (S, H_q)             f32
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q, H_KV, K,
        ANCHOR_S: tl.constexpr,
        ANCHOR_OFF: tl.constexpr,
        ANCHOR_WIDTH: tl.constexpr,
        MAX_D: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        split = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)
        max_d_range = tl.arange(0, MAX_D)
        max_d_anchor_mask = max_d_range < ANCHOR_WIDTH

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        # ── Anchor-slice q + ℓ2 norm (computed once per program) ──
        q_anchor = tl.load(
            Q_ptr + hq_vec[:, None] * D + (ANCHOR_OFF + max_d_range)[None, :],
            mask=g_valid[:, None] & max_d_anchor_mask[None, :],
            other=0.0,
        )
        q_anchor_norm = tl.sqrt(tl.sum(q_anchor * q_anchor, axis=1))

        th_anchor = tl.load(
            Th_ptr + ANCHOR_S * H_Q + hq_vec,
            mask=g_valid,
            other=float("inf"),
        )

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            parent_idx = p_chunk_start + parent_rel
            col_valid = parent_idx < p_end
            parent_idx_safe = tl.where(col_valid, parent_idx, 0)

            out_mask = g_valid[:, None] & col_valid[None, :]

            # ── Inline anchor gate ──
            # centers_anchor: (PARENTS_PER_PROG*BF, MAX_D) — redundant across
            # the BF children of the same parent; centers beyond ANCHOR_WIDTH
            # are zero-padded at build time, so the dot is naturally correct.
            centers_anchor = tl.load(
                Centers_ptr
                + ((ANCHOR_S * H_KV + kvh) * K + parent_idx_safe[:, None]) * MAX_D
                + max_d_range[None, :],
                mask=col_valid[:, None],
                other=0.0,
            )
            radii_anchor = tl.load(
                Radii_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx_safe,
                mask=col_valid,
                other=0.0,
            )
            cdot = tl.sum(
                q_anchor[:, None, :] * centers_anchor[None, :, :],
                axis=2,
            )
            ub = cdot + radii_anchor[None, :] * q_anchor_norm[:, None]
            anchor_pass = ub >= th_anchor[:, None]
            survive = anchor_pass & out_mask

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
                        ClusterPass_ptr
                        + (s * H_Q + hq_vec[:, None]) * K
                        + assign[None, :],
                        mask=survive,
                        other=0,
                    )
                    survive = survive & (passed != 0)

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
            if tl.max(live_cols.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysBlocksT_ptr
                    + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
                    + child_rel[None, :],
                    mask=live_cols[None, :],
                    other=0.0,
                )
                scores = tl.dot(q_f16, keys_tile)
                scores = tl.where(survive, scores, -1.0e30)

                chunk_max = tl.max(scores, axis=1)
                m_new = tl.maximum(m, chunk_max)
                alpha = tl.exp(m - m_new)
                p = tl.exp(scores - m_new[:, None])
                p = tl.where(survive, p, 0.0)
                l_acc = alpha * l_acc + tl.sum(p, axis=1)

                v_tile = tl.load(
                    ValuesBlocks_ptr
                    + ((kvh * K + parent_idx_safe[:, None]) * BF + child_rel[:, None]) * D_V
                    + dv_range[None, :],
                    mask=live_cols[:, None],
                    other=0.0,
                )
                p_f16 = p.to(tl.float16)
                o_acc = alpha[:, None] * o_acc + tl.dot(p_f16, v_tile)

                m = m_new

        tl.store(
            M_out_ptr + hq_vec * NUM_SPLITS + split,
            m,
            mask=g_valid,
        )
        tl.store(
            L_out_ptr + hq_vec * NUM_SPLITS + split,
            l_acc,
            mask=g_valid,
        )
        tl.store(
            O_out_ptr
            + (hq_vec[:, None] * NUM_SPLITS + split) * D_V
            + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def run_fused_attn_index_anchor_inline(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    assigns_blocks: torch.Tensor,
    cluster_pass: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    th_per_subspace: torch.Tensor,
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    s_subspaces: int,
    parents_per_prog: int,
    num_splits: int,
    anchor_s: int,
    anchor_off: int,
    anchor_width: int,
    max_d: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 3,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_anchor_inline_kernel[grid](
        q, keys_blocks_t_f16, values_blocks_f16,
        assigns_blocks, cluster_pass, invalid_blocks_i8,
        centers, radii, th_per_subspace,
        out_m, out_l, out_o,
        h_q, h_kv_eff, k,
        ANCHOR_S=anchor_s,
        ANCHOR_OFF=anchor_off,
        ANCHOR_WIDTH=anchor_width,
        MAX_D=max_d,
        D=d, D_V=d_v, BF=values_blocks_f16.shape[2],
        GROUPS=groups, GROUPS_POW=groups_pow,
        S=s_subspaces, PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
