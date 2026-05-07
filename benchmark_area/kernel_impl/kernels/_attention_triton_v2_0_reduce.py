"""v2.0 reduce kernel — uses tl.exp2 for consistency with exp2-based index kernel.

The m/l/o partials from the v2.0 index kernel live in "log2-score-space"
(scores scaled by LOG2E), so the reduce must also use exp2 when rescaling.
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
    def _attn_reduce_v2_0_kernel(
        M_idx_ptr,
        L_idx_ptr,
        O_idx_ptr,
        M_buf_ptr,
        L_buf_ptr,
        O_buf_ptr,
        Out_ptr,
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

        m_global = tl.maximum(tl.max(m_idx, axis=0), m_buf)

        alpha_idx = tl.exp2(m_idx - m_global)
        l_sum = tl.sum(alpha_idx * l_idx, axis=0) + tl.exp2(m_buf - m_global) * l_buf

        o_idx = tl.load(
            O_idx_ptr + (hq * NUM_SPLITS + s_range[:, None]) * D_V + dv[None, :],
            mask=s_valid[:, None],
            other=0.0,
        )
        o_sum = tl.sum(alpha_idx[:, None] * o_idx, axis=0)

        o_buf = tl.load(O_buf_ptr + hq * D_V + dv)
        o_sum = o_sum + tl.exp2(m_buf - m_global) * o_buf

        l_safe = tl.where(l_sum > 0.0, l_sum, 1.0)
        out = o_sum / l_safe
        tl.store(Out_ptr + hq * D_V + dv, out)


def run_attn_reduce_v2_0(
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
    _attn_reduce_v2_0_kernel[(h_q,)](
        m_idx, l_idx, o_idx,
        m_buf, l_buf, o_buf,
        out,
        NUM_SPLITS=num_splits,
        D_V=d_v,
        SPLITS_POW=splits_pow,
    )
