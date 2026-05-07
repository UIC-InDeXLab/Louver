"""v1.23 attention kernels — no split-partial tensors or reduce pass.

This variant keeps the precomputed cluster-pass path from v1.20, but the
index kernel now scans the whole parent range for one kv-head group and emits
one final online-softmax partial per query head. That removes the large
``(H_q, NUM_SPLITS, D_v)`` writeback and the follow-up reduce kernel.
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
    def _fused_attn_index_final_kernel(
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
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)

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

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(0, K, PARENTS_PER_PROG):
            parent_idx = p_chunk_start + parent_rel
            col_valid = parent_idx < K
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

        tl.store(M_out_ptr + hq_vec, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + hq_vec[:, None] * D_V + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )

    @triton.jit
    def _attn_finalize_single_kernel(
        L_idx_ptr,
        O_idx_ptr,
        Out_ptr,
        D_V: tl.constexpr,
    ):
        hq = tl.program_id(0)
        dv = tl.arange(0, D_V)

        l_idx = tl.load(L_idx_ptr + hq)
        o_idx = tl.load(O_idx_ptr + hq * D_V + dv)
        l_safe = tl.where(l_idx > 0.0, l_idx, 1.0)
        out = o_idx / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)

    @triton.jit
    def _attn_merge_two_kernel(
        M_idx_ptr,
        L_idx_ptr,
        O_idx_ptr,
        M_buf_ptr,
        L_buf_ptr,
        O_buf_ptr,
        Out_ptr,
        D_V: tl.constexpr,
    ):
        hq = tl.program_id(0)
        dv = tl.arange(0, D_V)

        m_idx = tl.load(M_idx_ptr + hq)
        l_idx = tl.load(L_idx_ptr + hq)
        m_buf = tl.load(M_buf_ptr + hq)
        l_buf = tl.load(L_buf_ptr + hq)

        m_global = tl.maximum(m_idx, m_buf)
        alpha_idx = tl.exp(m_idx - m_global)
        alpha_buf = tl.exp(m_buf - m_global)
        l_sum = alpha_idx * l_idx + alpha_buf * l_buf

        o_idx = tl.load(O_idx_ptr + hq * D_V + dv)
        o_buf = tl.load(O_buf_ptr + hq * D_V + dv)
        o_sum = alpha_idx * o_idx + alpha_buf * o_buf

        l_safe = tl.where(l_sum > 0.0, l_sum, 1.0)
        out = o_sum / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)


def run_fused_attn_index_final(
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
    _fused_attn_index_final_kernel[(h_kv_eff,)](
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
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def run_attn_finalize_single(
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    h_q = l_idx.shape[0]
    _attn_finalize_single_kernel[(h_q,)](
        l_idx,
        o_idx,
        out,
        D_V=o_idx.shape[-1],
    )


def run_attn_merge_two(
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    m_buf: torch.Tensor,
    l_buf: torch.Tensor,
    o_buf: torch.Tensor,
    out: torch.Tensor,
) -> None:
    h_q = m_idx.shape[0]
    _attn_merge_two_kernel[(h_q,)](
        m_idx,
        l_idx,
        o_idx,
        m_buf,
        l_buf,
        o_buf,
        out,
        D_V=o_idx.shape[-1],
    )
