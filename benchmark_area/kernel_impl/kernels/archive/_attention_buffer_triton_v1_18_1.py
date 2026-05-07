"""Triton buffer-attention kernel for v1.18.1 with fp16 query inputs."""

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
    def _buffer_attn_fp16q_kernel(
        Q_ptr,
        BufKeysT_ptr,
        BufValues_ptr,
        BufInvalid_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        L_BUF_MAX: tl.constexpr,
        BUF_COLS_PER_PROG: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)

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
                + dv_range[None, :],
                mask=col_valid[:, None],
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


def run_buffer_attn_fp16q(
    q: torch.Tensor,
    buf_keys_t_f16: torch.Tensor,
    buf_values_f16: torch.Tensor,
    buf_invalid_i8: torch.Tensor,
    h_kv_eff: int,
    groups: int,
    groups_pow: int,
    l_buf_max: int,
    buf_cols_per_prog: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    d = q.shape[1]
    d_v = buf_values_f16.shape[-1]
    grid = (h_kv_eff,)
    _buffer_attn_fp16q_kernel[grid](
        q, buf_keys_t_f16, buf_values_f16, buf_invalid_i8,
        out_m, out_l, out_o,
        D=d, D_V=d_v,
        GROUPS=groups, GROUPS_POW=groups_pow,
        L_BUF_MAX=l_buf_max,
        BUF_COLS_PER_PROG=buf_cols_per_prog,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
