"""v1.40 attention index kernel — anchor gate fused inline (no cluster_pass).

The v1.34 kernel reads a pre-computed ``cluster_pass[H_q, K]`` i8 gate that is
produced by a separate anchor-cluster kernel. That design adds a full kernel
launch (~3.5 µs in our bench) + a round-trip of the gate tensor through HBM,
even though the anchor math is trivial (one dot per parent).

This version folds the anchor gate into the splitkv body:

    cdot          = q_anchor · c_anchor
    parent_pass   = (cdot + r_anchor * ||q_anchor||) >= th_anchor

computed directly on the chunk of parents the program is about to consume. We
still keep the per-block ``invalid_blocks`` mask so buffer/padding blocks are
excluded.
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
    def _fused_attn_index_anchor_inline_v1_40_kernel(
        Q_ptr,                  # (H_q, D)           f16
        KeysBlocksT_ptr,        # (H_kv, K, D, BF)   f16
        ValuesBlocks_ptr,       # (H_kv, K, BF, D_v) f16
        CentersAnchor_ptr,      # (H_kv, K, WIDTH)   f16
        RadiiAnchor_ptr,        # (H_kv, K)          f16
        ThAnchor_ptr,           # (H_q,)             f16   threshold for anchor subspace
        QNormAnchor_ptr,        # (H_q,)             f16   ||q_anchor|| per q-head
        InvalidBlocks_ptr,      # (H_kv, K, BF)      i8
        M_out_ptr,              # (H_q, NUM_SPLITS)       f32
        L_out_ptr,              # (H_q, NUM_SPLITS)       f32
        O_out_ptr,              # (H_q, NUM_SPLITS, D_v)  f32
        H_Q,
        K,
        DIM_OFFSET: tl.constexpr,
        WIDTH: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
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
        width_range = tl.arange(0, WIDTH)

        q_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_anchor_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + (DIM_OFFSET + width_range)[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        qn_anchor = tl.load(
            QNormAnchor_ptr + hq_vec, mask=g_valid, other=0.0,
        ).to(tl.float32)
        th_anchor = tl.load(
            ThAnchor_ptr + hq_vec, mask=g_valid, other=float("inf"),
        ).to(tl.float32)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        parent_rel_p = tl.arange(0, PARENTS_PER_PROG)

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            # Per-parent metadata (anchor gate)
            parent_idx_p = p_chunk_start + parent_rel_p
            col_valid_p = parent_idx_p < p_end
            parent_idx_p_safe = tl.where(col_valid_p, parent_idx_p, 0)

            centers = tl.load(
                CentersAnchor_ptr
                + (kvh * K + parent_idx_p_safe[:, None]) * WIDTH
                + width_range[None, :],
                mask=col_valid_p[:, None],
                other=0.0,
            )  # [PARENTS_PER_PROG, WIDTH] fp16
            radii = tl.load(
                RadiiAnchor_ptr + kvh * K + parent_idx_p_safe,
                mask=col_valid_p,
                other=0.0,
            ).to(tl.float32)

            # cdot[g, p] = q_anchor[g, :] · centers[p, :]
            # Using tl.dot with tl.trans for tensor-core path.
            cdot = tl.dot(q_anchor_f16, tl.trans(centers))  # fp32 [GROUPS_POW, PARENTS_PER_PROG]
            ub = cdot + radii[None, :] * qn_anchor[:, None]
            parent_pass = ub >= th_anchor[:, None]  # [GROUPS_POW, PARENTS_PER_PROG] bool
            parent_pass = parent_pass & g_valid[:, None] & col_valid_p[None, :]

            # Per-parent: is anyone in this group passing this parent?
            any_hq_lives_parent = tl.max(parent_pass.to(tl.int32), axis=0) != 0

            # Per-column shape (PARENTS_PER_PROG*BF) metadata
            parent_idx = p_chunk_start + parent_rel
            col_valid = parent_idx < p_end
            parent_idx_safe = tl.where(col_valid, parent_idx, 0)

            # Expand parent_pass to cols dimension via broadcast + reshape.
            pp_exp = tl.broadcast_to(
                parent_pass[:, :, None], [GROUPS_POW, PARENTS_PER_PROG, BF]
            )
            anchor_pass_cols = tl.reshape(pp_exp, [GROUPS_POW, PARENTS_PER_PROG * BF])

            inv = tl.load(
                InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
                mask=col_valid,
                other=1,
            )
            col_live = col_valid & (inv == 0)
            survive = anchor_pass_cols & col_live[None, :]

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
            if tl.max(live_cols.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysBlocksT_ptr
                    + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
                    + child_rel[None, :],
                    mask=live_cols[None, :],
                    other=0.0,
                )
                scores = tl.dot(q_f16, keys_tile) * SCALE
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
                o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
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


def run_fused_attn_index_anchor_inline_fp16q(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    centers_anchor: torch.Tensor,
    radii_anchor: torch.Tensor,
    th_anchor: torch.Tensor,
    q_norm_anchor: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    dim_offset: int,
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    parents_per_prog: int,
    num_splits: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    width = centers_anchor.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_anchor_inline_v1_40_kernel[grid](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        centers_anchor,
        radii_anchor,
        th_anchor,
        q_norm_anchor,
        invalid_blocks_i8,
        out_m,
        out_l,
        out_o,
        h_q,
        k,
        DIM_OFFSET=int(dim_offset),
        WIDTH=int(width),
        D=d,
        D_V=d_v,
        BF=values_blocks_f16.shape[2],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
