"""v1.24 attention index kernel — on-chip reduction across split groups.

This keeps v1.20's logical 32-way split partitioning for work distribution,
but each program processes multiple consecutive splits and merges them in
registers before writing one partial. That reduces the global partial traffic
and the size of the follow-up reduce pass without collapsing to a single
program per kv head.
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
    def _fused_attn_index_grouped_splits_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        AssignsBlocks_ptr,
        ClusterPass_ptr,
        InvalidBlocks_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
        K,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLITS_PER_PROG: tl.constexpr,
        OUTER_SPLITS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        outer_split = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        m_run = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_run = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_run = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for inner_split in tl.static_range(0, SPLITS_PER_PROG):
            split = outer_split * SPLITS_PER_PROG + inner_split
            if split < NUM_SPLITS:
                p_start = split * parents_per_split
                p_end = tl.minimum(p_start + parents_per_split, K)

                m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
                l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
                o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

                for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
                    parent_idx = p_chunk_start + parent_rel
                    col_valid = parent_idx < p_end
                    parent_idx_safe = tl.where(col_valid, parent_idx, 0)

                    out_mask = g_valid[:, None] & col_valid[None, :]
                    anchor_pass = tl.load(
                        ClusterPass_ptr
                        + (ANCHOR_S * H_Q + hq_vec[:, None]) * K
                        + parent_idx_safe[None, :],
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
                        o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
                        m = m_new

                m_merge = tl.maximum(m_run, m)
                alpha_run = tl.exp(m_run - m_merge)
                alpha_split = tl.exp(m - m_merge)
                l_run = alpha_run * l_run + alpha_split * l_acc
                o_run = alpha_run[:, None] * o_run + alpha_split[:, None] * o_acc
                m_run = m_merge

        tl.store(
            M_out_ptr + hq_vec * OUTER_SPLITS + outer_split,
            m_run,
            mask=g_valid,
        )
        tl.store(
            L_out_ptr + hq_vec * OUTER_SPLITS + outer_split,
            l_run,
            mask=g_valid,
        )
        tl.store(
            O_out_ptr
            + (hq_vec[:, None] * OUTER_SPLITS + outer_split) * D_V
            + dv_range[None, :],
            o_run,
            mask=g_valid[:, None],
        )


def run_fused_attn_index_grouped_splits(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    assigns_blocks: torch.Tensor,
    cluster_pass: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    s_subspaces: int,
    parents_per_prog: int,
    num_splits: int,
    splits_per_prog: int,
    anchor_s: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    outer_splits = (num_splits + splits_per_prog - 1) // splits_per_prog
    _fused_attn_index_grouped_splits_kernel[(h_kv_eff, outer_splits)](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        assigns_blocks,
        cluster_pass,
        invalid_blocks_i8,
        out_m,
        out_l,
        out_o,
        h_q,
        h_kv_eff,
        k,
        ANCHOR_S=anchor_s,
        D=d,
        D_V=d_v,
        BF=values_blocks_f16.shape[2],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=s_subspaces,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SPLITS_PER_PROG=splits_per_prog,
        OUTER_SPLITS=outer_splits,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
