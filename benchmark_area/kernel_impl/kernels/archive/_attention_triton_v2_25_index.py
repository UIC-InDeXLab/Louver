"""v2.25 fused index+buffer kernel — 3-launch pipeline.

Reads cluster_pass directly (no child_survive pre-computation).
Uses the anchor gate to skip dead chunks first, then performs the
full AND filter only for surviving chunks by reading assigns_blocks
and cluster_pass.

Pipeline: cluster_pass(tl.dot) → THIS kernel → reduce = 3 launches.
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
    def _fused_attn_index_buf_v2_25_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        ClusterPass_ptr,
        AssignsBlocks_ptr,
        InvalidBlocks_ptr,
        CentersAnchor_ptr,
        RadiiAnchor_ptr,
        ThAnchor_ptr,
        QNormAnchor_ptr,
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
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
        SCALE_LOG2E: tl.constexpr,
        S: tl.constexpr,
        ANCHOR_S: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_SPLIT: tl.constexpr,
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
            parent_idx_p = p_chunk_start + parent_rel_p
            col_valid_p = parent_idx_p < p_end
            parent_idx_p_safe = tl.where(col_valid_p, parent_idx_p, 0)

            # --- Anchor gate (parent-level) ---
            centers = tl.load(
                CentersAnchor_ptr
                + (kvh * K + parent_idx_p_safe[:, None]) * WIDTH
                + width_range[None, :],
                mask=col_valid_p[:, None],
                other=0.0,
            )
            radii = tl.load(
                RadiiAnchor_ptr + kvh * K + parent_idx_p_safe,
                mask=col_valid_p,
                other=0.0,
            ).to(tl.float32)

            cdot = tl.dot(q_anchor_f16, tl.trans(centers))
            ub = cdot + radii[None, :] * qn_anchor[:, None]
            parent_pass = (ub >= th_anchor[:, None]) & g_valid[:, None] & col_valid_p[None, :]

            # Broadcast parent pass to children
            parent_idx = p_chunk_start + parent_rel
            col_valid = parent_idx < p_end
            parent_idx_safe = tl.where(col_valid, parent_idx, 0)

            pp_exp = tl.broadcast_to(
                parent_pass[:, :, None], [GROUPS_POW, PARENTS_PER_PROG, BF]
            )
            anchor_pass_cols = tl.reshape(pp_exp, [GROUPS_POW, PARENTS_PER_PROG * BF])

            inv = tl.load(
                InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
                mask=col_valid,
                other=1,
            )
            survive = anchor_pass_cols & (col_valid & (inv == 0))[None, :]

            # --- AND filter from cluster_pass (guarded by anchor gate) ---
            any_survive_vec = tl.max(survive.to(tl.int32), axis=0)
            if tl.max(any_survive_vec, axis=0) != 0:
                for s_idx in tl.static_range(0, S):
                    if s_idx != ANCHOR_S:
                        assign = tl.load(
                            AssignsBlocks_ptr
                            + ((s_idx * H_KV + kvh) * K + parent_idx_safe) * BF
                            + child_rel,
                            mask=any_survive_vec != 0,
                            other=0,
                        ).to(tl.int32)
                        passed = tl.load(
                            ClusterPass_ptr
                            + (s_idx * H_Q + hq_vec[:, None]) * K
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
                    scores = tl.dot(q_f16, keys_tile) * SCALE_LOG2E
                    scores = tl.where(survive, scores, -1.0e30)

                    chunk_max = tl.max(scores, axis=1)
                    m_new = tl.maximum(m, chunk_max)
                    alpha = tl.exp2(m - m_new)
                    p = tl.exp2(scores - m_new[:, None])
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

        # ── Phase 2: buffer keys ──
        buf_start = split * BUF_COLS_PER_SPLIT
        buf_end = tl.minimum(buf_start + BUF_COLS_PER_SPLIT, L_BUF_MAX)
        buf_cols_inner = tl.arange(0, BUF_COLS_PER_SPLIT)

        buf_col_idx = buf_start + buf_cols_inner
        buf_col_valid = buf_col_idx < buf_end
        buf_inv = tl.load(
            BufInvalid_ptr + kvh * L_BUF_MAX + buf_col_idx,
            mask=buf_col_valid,
            other=1,
        )
        buf_live = buf_col_valid & (buf_inv == 0)
        buf_survive = g_valid[:, None] & buf_live[None, :]

        any_buf_live = tl.max(buf_live.to(tl.int32), axis=0) != 0
        if any_buf_live:
            buf_keys_tile = tl.load(
                BufKeysT_ptr
                + (kvh * D + d_range[:, None]) * L_BUF_MAX
                + buf_col_idx[None, :],
                mask=buf_live[None, :],
                other=0.0,
            )
            buf_scores = tl.dot(q_f16, buf_keys_tile) * SCALE_LOG2E
            buf_scores = tl.where(buf_survive, buf_scores, -1.0e30)

            buf_chunk_max = tl.max(buf_scores, axis=1)
            m_new = tl.maximum(m, buf_chunk_max)
            alpha = tl.exp2(m - m_new)
            bp = tl.exp2(buf_scores - m_new[:, None])
            bp = tl.where(buf_survive, bp, 0.0)
            l_acc = alpha * l_acc + tl.sum(bp, axis=1)

            buf_v_tile = tl.load(
                BufValues_ptr
                + (kvh * L_BUF_MAX + buf_col_idx[:, None]) * D_V
                + dv_range[None, :],
                mask=buf_live[:, None],
                other=0.0,
            )
            o_acc = alpha[:, None] * o_acc + tl.dot(bp.to(tl.float16), buf_v_tile)
            m = m_new

        tl.store(M_out_ptr + hq_vec * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + (hq_vec[:, None] * NUM_SPLITS + split) * D_V + dv_range[None, :],
            o_acc, mask=g_valid[:, None],
        )


def _next_pow2_min16(x: int) -> int:
    p = 16
    while p < x:
        p *= 2
    return p


def run_fused_attn_index_buf_v2_25(
    q, keys_blocks_t_f16, values_blocks_f16,
    cluster_pass, assigns_blocks, invalid_blocks_i8,
    centers_anchor, radii_anchor, th_anchor, q_norm_anchor,
    buf_keys_t_f16, buf_values_f16, buf_invalid_i8,
    h_q, h_kv_eff, k, groups, groups_pow,
    parents_per_prog, num_splits, scale_log2e,
    dim_offset, width, s_subspaces, anchor_s,
    l_buf_max,
    out_m, out_l, out_o,
    num_warps=4, num_stages=3,
):
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    buf_cols_per_split = _next_pow2_min16(
        max(1, (l_buf_max + num_splits - 1) // num_splits)
    )
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_buf_v2_25_kernel[grid](
        q, keys_blocks_t_f16, values_blocks_f16,
        cluster_pass, assigns_blocks, invalid_blocks_i8,
        centers_anchor, radii_anchor, th_anchor, q_norm_anchor,
        buf_keys_t_f16, buf_values_f16, buf_invalid_i8,
        out_m, out_l, out_o,
        h_q, h_kv_eff, k,
        DIM_OFFSET=dim_offset,
        WIDTH=width,
        D=d, D_V=d_v,
        BF=int(values_blocks_f16.shape[2]),
        GROUPS=groups, GROUPS_POW=groups_pow,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE_LOG2E=float(scale_log2e),
        S=s_subspaces,
        ANCHOR_S=anchor_s,
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_SPLIT=buf_cols_per_split,
        num_warps=num_warps,
        num_stages=num_stages,
    )
