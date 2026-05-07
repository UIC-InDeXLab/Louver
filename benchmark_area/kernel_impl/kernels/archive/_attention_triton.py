"""Triton kernels for fused sparse-index attention.

Pipeline (see attention_v1_0):
  1. cluster_pass kernel (reused from _search_triton) → (S, H_q, K) gate.
  2. fused attention index kernel → per-split partials (m, l, o).
  3. buffer attention (torch) → one partial per hq.
  4. reduce kernel → final (H_q, D_v) attention output.

Design notes:
  - No causal mask inside the kernel: every indexed / buffered key-value
    pair is a valid past attention target. The caller decides what to
    put in the index and the buffer.
  - Values are stored fp16 in (H_kv, K, BF, D_v) layout (see build_v2_4)
    so the V-tile load is contiguous per (parent, child).
  - Online softmax uses a large negative sentinel (-1e30) for the initial
    running max so that -inf-only chunks don't trigger NaN in `m - m_new`.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


NEG_SENT = -1.0e30  # safe "minus-infinity" sentinel that still subtracts cleanly


if HAS_TRITON:

    @triton.jit
    def _fused_attn_index_kernel(
        Q_ptr,                  # (H_q, D)           f32
        KeysBlocksT_ptr,        # (H_kv, K, D, BF)   f16
        ValuesBlocks_ptr,       # (H_kv, K, BF, D_v) f16
        AssignsBlocks_ptr,      # (S, H_kv, K, BF)
        ClusterPass_ptr,        # (S, H_q, K)        i8
        InvalidBlocks_ptr,      # (H_kv, K, BF)      i8
        M_out_ptr,              # (H_q, NUM_SPLITS)       f32
        L_out_ptr,              # (H_q, NUM_SPLITS)       f32
        O_out_ptr,              # (H_q, NUM_SPLITS, D_v)  f32
        H_Q, H_KV, K,
        ANCHOR_S: tl.constexpr,
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

        # Load q once and scale it; fp16 for tensor-core dot.
        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_scaled = q_full_f32 * SCALE
        q_f16 = q_scaled.to(tl.float16)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        # Parent range owned by this split.
        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        # Iterate parent chunks inside this split.
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

            # Per-column liveness for masked loads of keys/values.
            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0

            # Load keys tile and compute scaled scores. dot output is f32.
            keys_tile = tl.load(
                KeysBlocksT_ptr
                + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
                + child_rel[None, :],
                mask=live_cols[None, :],
                other=0.0,
            )
            scores = tl.dot(q_f16, keys_tile)  # (GROUPS_POW, P*BF)
            scores = tl.where(survive, scores, -1.0e30)

            # Online softmax update.
            chunk_max = tl.max(scores, axis=1)                # (GROUPS_POW,)
            m_new = tl.maximum(m, chunk_max)                  # (GROUPS_POW,)
            alpha = tl.exp(m - m_new)                         # (GROUPS_POW,)
            p = tl.exp(scores - m_new[:, None])               # (GROUPS_POW, P*BF)
            p = tl.where(survive, p, 0.0)
            l_acc = alpha * l_acc + tl.sum(p, axis=1)

            # Load V tile and accumulate p @ V.
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

        # Write split partials.
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

    @triton.jit
    def _attn_reduce_kernel(
        M_idx_ptr,              # (H_q, NUM_SPLITS)       f32
        L_idx_ptr,              # (H_q, NUM_SPLITS)       f32
        O_idx_ptr,              # (H_q, NUM_SPLITS, D_v)  f32
        M_buf_ptr,              # (H_q,)                  f32   (NEG_SENT if no buffer)
        L_buf_ptr,              # (H_q,)                  f32
        O_buf_ptr,              # (H_q, D_v)              f32
        Out_ptr,                # (H_q, D_v)              f32
        NUM_SPLITS: tl.constexpr,
        D_V: tl.constexpr,
        SPLITS_POW: tl.constexpr,
    ):
        hq = tl.program_id(0)
        s_range = tl.arange(0, SPLITS_POW)
        s_valid = s_range < NUM_SPLITS
        dv = tl.arange(0, D_V)

        m_idx = tl.load(M_idx_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=-1.0e30)
        l_idx = tl.load(L_idx_ptr + hq * NUM_SPLITS + s_range, mask=s_valid, other=0.0)
        m_buf = tl.load(M_buf_ptr + hq)
        l_buf = tl.load(L_buf_ptr + hq)

        # Global max across all partials.
        m_global = tl.maximum(tl.max(m_idx, axis=0), m_buf)

        # Rescale per-split partials.
        alpha_idx = tl.exp(m_idx - m_global)  # (SPLITS_POW,)
        l_sum = tl.sum(alpha_idx * l_idx, axis=0) + tl.exp(m_buf - m_global) * l_buf

        # Weighted o from index partials.
        o_idx = tl.load(
            O_idx_ptr + (hq * NUM_SPLITS + s_range[:, None]) * D_V + dv[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        # Sum over splits: (SPLITS_POW, D_v) -> (D_v,) with per-split alpha.
        o_sum = tl.sum(alpha_idx[:, None] * o_idx, axis=0)  # (D_v,)

        o_buf = tl.load(O_buf_ptr + hq * D_V + dv)
        o_sum = o_sum + tl.exp(m_buf - m_global) * o_buf

        # Guard against l_sum == 0 (no survivors anywhere).
        l_safe = tl.where(l_sum > 0.0, l_sum, 1.0)
        out = o_sum / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)


def run_fused_attn_index(
    q: torch.Tensor,                  # (H_q, D) f32
    keys_blocks_t_f16: torch.Tensor,  # (H_kv, K, D, BF) f16
    values_blocks_f16: torch.Tensor,  # (H_kv, K, BF, D_v) f16
    assigns_blocks: torch.Tensor,     # (S, H_kv, K, BF) i16/i32
    cluster_pass: torch.Tensor,       # (S, H_q, K) i8
    invalid_blocks_i8: torch.Tensor,  # (H_kv, K, BF) i8
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    s_subspaces: int,
    parents_per_prog: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    out_m: torch.Tensor,               # (H_q, NUM_SPLITS) f32
    out_l: torch.Tensor,               # (H_q, NUM_SPLITS) f32
    out_o: torch.Tensor,               # (H_q, NUM_SPLITS, D_v) f32
    num_warps: int = 4,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_kernel[grid](
        q, keys_blocks_t_f16, values_blocks_f16,
        assigns_blocks, cluster_pass, invalid_blocks_i8,
        out_m, out_l, out_o,
        h_q, h_kv_eff, k,
        ANCHOR_S=anchor_s,
        D=d, D_V=d_v, BF=values_blocks_f16.shape[2],
        GROUPS=groups, GROUPS_POW=groups_pow,
        S=s_subspaces, PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
    )


def run_attn_reduce(
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    m_buf: torch.Tensor,
    l_buf: torch.Tensor,
    o_buf: torch.Tensor,
    out: torch.Tensor,
) -> None:
    h_q, num_splits = m_idx.shape
    d_v = o_idx.shape[-1]
    splits_pow = 1
    while splits_pow < max(num_splits, 1):
        splits_pow *= 2
    _attn_reduce_kernel[(h_q,)](
        m_idx, l_idx, o_idx,
        m_buf, l_buf, o_buf,
        out,
        NUM_SPLITS=num_splits,
        D_V=d_v,
        SPLITS_POW=splits_pow,
    )
