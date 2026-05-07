"""v1.37 anchor-only cluster-pass kernel with bitpacked output."""

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
    def _anchor_cluster_pass_rawq_fp16_v1_37_kernel(
        Q_ptr,
        QNormAnchor_ptr,
        ThAnchor_ptr,
        CentersAnchor_ptr,
        RadiiAnchor_ptr,
        OutWords_ptr,
        H_Q,
        K,
        K_WORDS,
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

        cdot = tl.sum(q[:, None, :] * centers[None, :, :], axis=2)
        ub = cdot + radii[None, :] * qn[:, None]
        passed = (ub >= th[:, None]) & (g_valid[:, None] & k_mask[None, :])

        bit_range = tl.arange(0, 32)
        weights = (1 << bit_range).to(tl.int32)
        for word_rel in tl.static_range(0, BLOCK_K // 32):
            word_idx = k0 // 32 + word_rel
            word_mask = word_idx < K_WORDS
            bits = passed[:, word_rel * 32 + bit_range].to(tl.int32)
            packed = tl.sum(bits * weights[None, :], axis=1)
            tl.store(
                OutWords_ptr + hq_vec * K_WORDS + word_idx,
                packed,
                mask=g_valid & word_mask,
            )


def triton_anchor_cluster_pass_rawq_fp16_bitpack(
    q: torch.Tensor,
    q_norm_anchor: torch.Tensor,
    th_anchor: torch.Tensor,
    centers_anchor: torch.Tensor,
    radii_anchor: torch.Tensor,
    dim_offset: int,
    groups: int,
    out_words: torch.Tensor | None = None,
) -> torch.Tensor:
    h_q, d = q.shape
    h_kv, k, width = centers_anchor.shape
    k_words = (k + 31) // 32
    if out_words is None:
        out_words = torch.empty(h_q, k_words, device=q.device, dtype=torch.int32)

    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2
    block_k = 64

    grid = (h_kv, triton.cdiv(k, block_k))
    _anchor_cluster_pass_rawq_fp16_v1_37_kernel[grid](
        q,
        q_norm_anchor,
        th_anchor,
        centers_anchor,
        radii_anchor,
        out_words,
        h_q,
        k,
        k_words,
        DIM_OFFSET=dim_offset,
        D=d,
        WIDTH=width,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        BLOCK_K=block_k,
        num_warps=2,
    )
    return out_words
