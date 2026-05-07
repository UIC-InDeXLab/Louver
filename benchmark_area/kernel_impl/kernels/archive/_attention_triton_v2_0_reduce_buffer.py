"""v2.0 fused reduce + buffer kernel — uses tl.exp2 throughout.

Combines index-partial reduction with buffer scanning, all using exp2
to stay consistent with the v2.0 index kernel's log2-score-space.
Caller must pass scale pre-multiplied by LOG2E.
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
    def _attn_reduce_buffer_v2_0_kernel(
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
        SCALE_LOG2E: tl.constexpr,
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
        alpha_idx = tl.exp2(m_idx - m[:, None])
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
            scores = tl.dot(q_f16, keys_tile) * SCALE_LOG2E
            scores = tl.where(survive, scores, -1.0e30)

            chunk_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp2(m - m_new)
            p = tl.exp2(scores - m_new[:, None])
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


def run_attn_reduce_buffer_v2_0(
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
    scale_log2e: float,
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
    _attn_reduce_buffer_v2_0_kernel[grid](
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
        SCALE_LOG2E=float(scale_log2e),
        num_warps=num_warps,
        num_stages=num_stages,
    )
