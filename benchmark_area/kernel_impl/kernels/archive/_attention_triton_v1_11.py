"""v1.11 main attention kernel — dead-chunk early exit on top of v1.10.

Identical to `_fused_attn_index_parentlist_kernel` except we wrap the heavy
keys/values load + tensor-core dots in `if tl.max(child_masks, axis=0) != 0`
so fully-pruned parent chunks are skipped.
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
    def _fused_attn_index_parentlist_skip_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        ParentIds_ptr,
        ChildMasks_ptr,
        Counts_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
        K,
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

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        count = tl.load(Counts_ptr + kvh)
        parents_per_split = (count + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, count)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            list_idx = p_chunk_start + parent_rel
            col_valid = list_idx < p_end
            list_idx_safe = tl.where(col_valid, list_idx, 0)

            child_masks = tl.load(
                ChildMasks_ptr + kvh * K + list_idx_safe,
                mask=col_valid,
                other=0,
            ).to(tl.int32)

            # Early exit: skip the entire chunk if every (group, child) bit is dead.
            any_live = tl.max(child_masks, axis=0)
            if any_live != 0:
                parent_idx = tl.load(
                    ParentIds_ptr + kvh * K + list_idx_safe,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)

                shifts = g_range[:, None] * BF + child_rel[None, :]
                survive = (((child_masks[None, :] >> shifts) & 1) != 0) & (
                    g_valid[:, None] & col_valid[None, :]
                )

                live_cols = tl.max(survive.to(tl.int32), axis=0) != 0

                keys_tile = tl.load(
                    KeysBlocksT_ptr
                    + ((kvh * K + parent_idx[None, :]) * D + d_range[:, None]) * BF
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
                    + ((kvh * K + parent_idx[:, None]) * BF + child_rel[:, None]) * D_V
                    + dv_range[None, :],
                    mask=live_cols[:, None],
                    other=0.0,
                )
                o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
                m = m_new

        tl.store(M_out_ptr + hq_vec * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + (hq_vec[:, None] * NUM_SPLITS + split) * D_V + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def run_fused_attn_index_parentlist_skip(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    parent_ids: torch.Tensor,
    child_masks: torch.Tensor,
    counts: torch.Tensor,
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
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_parentlist_skip_kernel[grid](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        parent_ids,
        child_masks,
        counts,
        out_m,
        out_l,
        out_o,
        h_q,
        h_kv_eff,
        k,
        D=d,
        D_V=d_v,
        BF=values_blocks_f16.shape[2],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
    )
