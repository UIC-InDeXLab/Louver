"""v1.37 attention index kernel — bitpacked anchor-only gate."""

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
    def _fused_attn_index_anchor_bits_fp16q_v1_37_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        ClusterPassWords_ptr,
        InvalidBlocks_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        K,
        K_WORDS,
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

        q_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        parents = tl.arange(0, PARENTS_PER_PROG)
        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            parent_idx = p_chunk_start + parents
            parent_valid = parent_idx < p_end
            col_valid = parent_valid[parent_rel]

            base_word_idx = p_chunk_start // 32
            base_bit = p_chunk_start % 32
            next_word_idx = tl.minimum(base_word_idx + 1, K_WORDS - 1)

            word0 = tl.load(
                ClusterPassWords_ptr + hq_vec * K_WORDS + base_word_idx,
                mask=g_valid & (base_word_idx < K_WORDS),
                other=0,
            ).to(tl.uint32)
            word1 = tl.load(
                ClusterPassWords_ptr + hq_vec * K_WORDS + next_word_idx,
                mask=g_valid & (next_word_idx < K_WORDS),
                other=0,
            ).to(tl.uint32)
            if base_bit == 0:
                gate_word = word0
            else:
                gate_word = (word0 >> base_bit) | (word1 << (32 - base_bit))

            parent_bits = ((gate_word[:, None] >> parents[None, :]) & 1) != 0
            survive = parent_bits[:, parent_rel] & (g_valid[:, None] & col_valid[None, :])

            inv = tl.load(
                InvalidBlocks_ptr + ((kvh * K + parent_idx[parent_rel]) * BF + child_rel),
                mask=col_valid,
                other=1,
            )
            survive = survive & (inv[None, :] == 0)

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
            if tl.max(live_cols.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysBlocksT_ptr
                    + ((kvh * K + parent_idx[parent_rel][None, :]) * D + d_range[:, None]) * BF
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
                    + ((kvh * K + parent_idx[parent_rel][:, None]) * BF + child_rel[:, None]) * D_V
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


def run_fused_attn_index_anchor_bits_fp16q(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    cluster_pass_words: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
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
    k_words = (k + 31) // 32
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_anchor_bits_fp16q_v1_37_kernel[grid](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        cluster_pass_words,
        invalid_blocks_i8,
        out_m,
        out_l,
        out_o,
        h_q,
        k,
        k_words,
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
