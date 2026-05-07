"""v1.31 cluster-pass kernel with fp16 q/th/centers/radii/qnorms inputs."""

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
    def _fused_cluster_pass_rawq_fp16_v1_31_kernel(
        Q_ptr,
        QNorms_ptr,
        DimOffsets_ptr,
        DimWidths_ptr,
        Th_ptr,
        Centers_ptr,
        Radii_ptr,
        Out_ptr,
        H_Q,
        H_KV,
        K,
        D: tl.constexpr,
        S: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        MAX_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        s = tl.program_id(0)
        kvh = tl.program_id(1)
        k0 = tl.program_id(2) * BLOCK_K

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        d_range = tl.arange(0, MAX_D)
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        hq_vec = kvh * GROUPS + g_range

        dim_off = tl.load(DimOffsets_ptr + s)
        width = tl.load(DimWidths_ptr + s)
        d_valid = d_range < width

        q_load_mask = g_valid[:, None] & d_valid[None, :]
        qp = tl.load(
            Q_ptr + hq_vec[:, None] * D + (dim_off + d_range)[None, :],
            mask=q_load_mask,
            other=0.0,
        )
        qn = tl.load(QNorms_ptr + s * H_Q + hq_vec, mask=g_valid, other=0.0)

        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)

        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int8)

        out_offs = (s * H_Q + hq_vec[:, None]) * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)


def triton_fused_cluster_pass_rawq_fp16(
    q: torch.Tensor,
    q_norms: torch.Tensor,
    th: torch.Tensor,
    dim_offsets: torch.Tensor,
    dim_widths: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    groups: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    h_q, d = q.shape
    s, h_kv, k, max_d = centers.shape
    if out is None:
        out = torch.empty(s, h_q, k, device=q.device, dtype=torch.int8)

    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2
    block_k = 64

    grid = (s, h_kv, triton.cdiv(k, block_k))
    _fused_cluster_pass_rawq_fp16_v1_31_kernel[grid](
        q, q_norms, dim_offsets, dim_widths, th, centers, radii, out,
        h_q, h_kv, k,
        D=d, S=s, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out
