"""Tiny Triton copy kernel that copies two fp16 buffers in a single launch.

Replaces two separate ``cudaMemcpyAsync`` calls with one Triton kernel
launch.  On small tensors (~8 KB), the per-launch overhead (~0.5 µs) saved
by coalescing is a measurable fraction of total query time.
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
    def _fused_copy_kernel(
        Src_q_ptr,
        Src_th_ptr,
        Dst_q_ptr,
        Dst_th_ptr,
        Q_ELEMS: tl.constexpr,
        TH_ELEMS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)

        q_mask = offs < Q_ELEMS
        q_vals = tl.load(Src_q_ptr + offs, mask=q_mask, other=0.0)
        tl.store(Dst_q_ptr + offs, q_vals, mask=q_mask)

        th_mask = offs < TH_ELEMS
        th_vals = tl.load(Src_th_ptr + offs, mask=th_mask, other=0.0)
        tl.store(Dst_th_ptr + offs, th_vals, mask=th_mask)


def fused_copy_q_th(
    src_q: torch.Tensor,
    src_th: torch.Tensor,
    dst_q: torch.Tensor,
    dst_th: torch.Tensor,
) -> None:
    src_q_flat = src_q.view(-1)
    src_th_flat = src_th.view(-1)
    dst_q_flat = dst_q.view(-1)
    dst_th_flat = dst_th.view(-1)
    q_elems = src_q_flat.numel()
    th_elems = src_th_flat.numel()
    total = max(q_elems, th_elems)
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    _fused_copy_kernel[grid](
        src_q_flat,
        src_th_flat,
        dst_q_flat,
        dst_th_flat,
        Q_ELEMS=q_elems,
        TH_ELEMS=th_elems,
        BLOCK=BLOCK,
        num_warps=1,
    )
