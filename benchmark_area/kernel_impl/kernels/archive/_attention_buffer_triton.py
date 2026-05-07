"""Triton buffer-attention kernel for attention_v1_16.

Computes per-head (m_buf, l_buf, o_buf) from a padded, fixed-bucket buffer
of keys/values. Output is written directly into the (H_q,) and (H_q, D_v)
slots consumed by `_attn_reduce_kernel`, so no extra reduce pass is needed
as long as one program owns the full buffer for its kv-head.

Layout expectations:
  BufKeysT:   (H_kv_eff, D,   L_BUF_MAX) f16  (D-major — matches index kernel)
  BufValues:  (H_kv_eff, L_BUF_MAX, D_v)  f16
  BufInvalid: (H_kv_eff, L_BUF_MAX)       i8   (1 == padded/invalid, 0 == valid)

L_BUF_MAX is declared `constexpr`, so Triton emits a separately-cached SASS
per bucket in {64, 128, 256, 512}. Actual buffer length is encoded entirely
in `BufInvalid` — the kernel always iterates the full bucket.
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
    def _buffer_attn_kernel(
        Q_ptr,                  # (H_q, D) f32
        BufKeysT_ptr,           # (H_kv_eff, D, L_BUF_MAX) f16
        BufValues_ptr,          # (H_kv_eff, L_BUF_MAX, D_v) f16
        BufInvalid_ptr,         # (H_kv_eff, L_BUF_MAX) i8
        M_out_ptr,              # (H_q,)       f32
        L_out_ptr,              # (H_q,)       f32
        O_out_ptr,              # (H_q, D_v)   f32
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

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        cols_inner = tl.arange(0, BUF_COLS_PER_PROG)

        for c_start in range(0, L_BUF_MAX, BUF_COLS_PER_PROG):
            col_idx = c_start + cols_inner  # all < L_BUF_MAX by construction

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
            scores = tl.dot(q_f16, keys_tile)
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
            p_f16 = p.to(tl.float16)
            o_acc = alpha[:, None] * o_acc + tl.dot(p_f16, v_tile)

            m = m_new

        tl.store(M_out_ptr + hq_vec, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + hq_vec[:, None] * D_V + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def run_buffer_attn(
    q: torch.Tensor,                    # (H_q, D) f32
    buf_keys_t_f16: torch.Tensor,       # (H_kv_eff, D, L_BUF_MAX) f16
    buf_values_f16: torch.Tensor,       # (H_kv_eff, L_BUF_MAX, D_v) f16
    buf_invalid_i8: torch.Tensor,       # (H_kv_eff, L_BUF_MAX) i8
    h_kv_eff: int,
    groups: int,
    groups_pow: int,
    l_buf_max: int,
    buf_cols_per_prog: int,
    scale: float,
    out_m: torch.Tensor,                # (H_q,) f32
    out_l: torch.Tensor,                # (H_q,) f32
    out_o: torch.Tensor,                # (H_q, D_v) f32
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    d = q.shape[1]
    d_v = buf_values_f16.shape[-1]
    grid = (h_kv_eff,)
    _buffer_attn_kernel[grid](
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
