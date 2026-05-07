"""v1.36 anchor-only cluster-pass kernel with tensor-core dot."""

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
    def _anchor_cluster_pass_rawq_fp16_v1_36_kernel(
        Q_ptr,
        QNormAnchor_ptr,
        ThAnchor_ptr,
        CentersAnchor_ptr,
        RadiiAnchor_ptr,
        Out_ptr,
        H_Q,
        K,
        DIM_OFFSET: tl.constexpr,
        D: tl.constexpr,
        WIDTH: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        k0 = tl.program_id(1) * BLOCK_K

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        d_range = tl.arange(0, WIDTH)
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        hq_vec = kvh * GROUPS + g_range

        q = tl.load(
            Q_ptr + hq_vec[:, None] * D + (DIM_OFFSET + d_range)[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        qn = tl.load(QNormAnchor_ptr + hq_vec, mask=g_valid, other=0.0)
        th = tl.load(ThAnchor_ptr + hq_vec, mask=g_valid, other=float("inf"))

        centers = tl.load(
            CentersAnchor_ptr + (kvh * K + k_range[:, None]) * WIDTH + d_range[None, :],
            mask=k_mask[:, None],
            other=0.0,
        )
        radii = tl.load(RadiiAnchor_ptr + kvh * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.dot(q, tl.trans(centers))
        ub = cdot + radii[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int8)

        out_offs = hq_vec[:, None] * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)


def triton_anchor_cluster_pass_rawq_fp16_tc(
    q: torch.Tensor,
    q_norm_anchor: torch.Tensor,
    th_anchor: torch.Tensor,
    centers_anchor: torch.Tensor,
    radii_anchor: torch.Tensor,
    dim_offset: int,
    groups: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    h_q, d = q.shape
    h_kv, k, width = centers_anchor.shape
    if out is None:
        out = torch.empty(h_q, k, device=q.device, dtype=torch.int8)

    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2

    block_k = 128
    grid = (h_kv, triton.cdiv(k, block_k))
    _anchor_cluster_pass_rawq_fp16_v1_36_kernel[grid](
        q,
        q_norm_anchor,
        th_anchor,
        centers_anchor,
        radii_anchor,
        out,
        h_q,
        k,
        DIM_OFFSET=dim_offset,
        D=d,
        WIDTH=width,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out
