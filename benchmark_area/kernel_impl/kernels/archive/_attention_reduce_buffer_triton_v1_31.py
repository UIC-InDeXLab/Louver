"""v1.31 fused reduce + buffer kernel with fp16 q input.

Combines v1.30's fused "reduce-partials + scan-buffer" kernel with
v1.18.1's fp16 q convention (q is loaded as fp16; scaling is applied
post-dot to match the index kernel).
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
    def _attn_reduce_buffer_fp16q_kernel(
        Q_ptr,
        M_idx_ptr,
        L_idx_ptr,
        O_idx_ptr,
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        Out_ptr,
        H_Q,
        NUM_SPLITS_RUNTIME,
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_TILE: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SPLITS_POW: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        group_tile = tl.program_id(1)

        g_local = tl.arange(0, GROUPS_TILE)
        g_idx = group_tile * GROUPS_TILE + g_local
        g_valid = g_idx < GROUPS
        hq_vec = kvh * GROUPS + g_idx

        d_range = tl.arange(0, D)
        dv = tl.arange(0, D_V)
        s_range = tl.arange(0, SPLITS_POW)
        s_valid = s_range < NUM_SPLITS

        q_f16 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )

        m_idx = tl.load(
            M_idx_ptr + hq_vec[:, None] * NUM_SPLITS_RUNTIME + s_range[None, :],
            mask=g_valid[:, None] & s_valid[None, :],
            other=-1.0e30,
        )
        l_idx = tl.load(
            L_idx_ptr + hq_vec[:, None] * NUM_SPLITS_RUNTIME + s_range[None, :],
            mask=g_valid[:, None] & s_valid[None, :],
            other=0.0,
        )
        m = tl.max(m_idx, axis=1)
        alpha_idx = tl.exp(m_idx - m[:, None])
        l_acc = tl.sum(alpha_idx * l_idx, axis=1)

        o_idx = tl.load(
            O_idx_ptr
            + (hq_vec[:, None, None] * NUM_SPLITS_RUNTIME + s_range[None, :, None]) * D_V
            + dv[None, None, :],
            mask=g_valid[:, None, None] & s_valid[None, :, None],
            other=0.0,
        )
        o_acc = tl.sum(alpha_idx[:, :, None] * o_idx, axis=1)

        cols_inner = tl.arange(0, BUF_COLS_PER_PROG)
        for c_start in range(0, L_BUF_MAX, BUF_COLS_PER_PROG):
            col_idx = c_start + cols_inner
            inv = tl.load(BufInvalid_ptr + kvh * L_BUF_MAX + col_idx)
            col_valid = inv == 0
            survive = g_valid[:, None] & col_valid[None, :]

            keys_tile = tl.load(
                BufKeysT_ptr
                + (kvh * D + d_range[:, None]) * L_BUF_MAX
                + col_idx[None, :],
                mask=col_valid[None, :],
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
                BufValues_ptr
                + (kvh * L_BUF_MAX + col_idx[:, None]) * D_V
                + dv[None, :],
                mask=col_valid[:, None],
                other=0.0,
            )
            o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
            m = m_new

        l_safe = tl.where(l_acc > 0.0, l_acc, 1.0)
        out = o_acc / l_safe[:, None]
        tl.store(
            Out_ptr + hq_vec[:, None] * D_V + dv[None, :],
            out,
            mask=g_valid[:, None],
        )


def run_attn_reduce_buffer_fp16q(
    q: torch.Tensor,
    m_idx: torch.Tensor,
    l_idx: torch.Tensor,
    o_idx: torch.Tensor,
    buf_keys_t_f16: torch.Tensor,
    buf_values_f16: torch.Tensor,
    buf_invalid_i8: torch.Tensor,
    h_kv_eff: int,
    groups: int,
    l_buf_max: int,
    buf_cols_per_prog: int,
    scale: float,
    out: torch.Tensor,
    groups_tile: int = 4,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    d = q.shape[1]
    d_v = o_idx.shape[-1]
    num_splits = m_idx.shape[1]
    splits_pow = 1
    while splits_pow < max(num_splits, 1):
        splits_pow *= 2
    grid = (h_kv_eff, triton.cdiv(groups, groups_tile))
    _attn_reduce_buffer_fp16q_kernel[grid](
        q,
        m_idx,
        l_idx,
        o_idx,
        buf_keys_t_f16,
        buf_values_f16,
        buf_invalid_i8,
        out,
        q.shape[0],
        num_splits,
        D=d,
        D_V=d_v,
        GROUPS=groups,
        GROUPS_TILE=groups_tile,
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_PROG=buf_cols_per_prog,
        NUM_SPLITS=num_splits,
        SPLITS_POW=splits_pow,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
