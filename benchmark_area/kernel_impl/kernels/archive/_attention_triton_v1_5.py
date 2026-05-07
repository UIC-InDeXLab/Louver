"""v1.5 cluster_pass kernel that consumes raw q (no q_pack).

Same gate logic as `_fused_cluster_pass_kernel` in `_search_triton`, but loads
each subspace's q slice directly from the raw `(H_q, D)` tensor using per-
subspace (offset, width) metadata. Computes `qn` (per-group ℓ2 norm) on the
fly. Avoids the 4-op Python `_pack_q` path (view/transpose/contiguous/norm).
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
    def _fused_cluster_pass_rawq_kernel(
        Q_ptr,             # (H_q, D)             f32
        DimOffsets_ptr,    # (S,)                 i32
        DimWidths_ptr,     # (S,)                 i32
        Th_ptr,            # (S, H_q)             f32
        Centers_ptr,       # (S, H_kv, K, MAX_D)  f32
        Radii_ptr,         # (S, H_kv, K)         f32
        Out_ptr,           # (S, H_q, K)          i8
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

        # Pull q slice from raw (H_q, D): qp[g, d] = q[hq_vec[g], dim_off + d_range[d]]
        q_load_mask = g_valid[:, None] & d_valid[None, :]
        qp = tl.load(
            Q_ptr + hq_vec[:, None] * D + (dim_off + d_range)[None, :],
            mask=q_load_mask,
            other=0.0,
        )
        qn = tl.sqrt(tl.sum(qp * qp, axis=1))  # (GROUPS_POW,)

        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)

        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)  # (GROUPS_POW, BLOCK_K)
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int8)

        out_offs = (s * H_Q + hq_vec[:, None]) * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)


def triton_fused_cluster_pass_rawq(
    q: torch.Tensor,                 # (H_q, D)
    th: torch.Tensor,                # (S, H_q)
    dim_offsets: torch.Tensor,       # (S,) int32
    dim_widths: torch.Tensor,        # (S,) int32
    centers: torch.Tensor,           # (S, H_kv, K, max_d)
    radii: torch.Tensor,             # (S, H_kv, K)
    groups: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    h_q, d = q.shape
    s, h_kv, k, max_d = centers.shape
    if out is None:
        out = torch.empty(s, h_q, k, device=q.device, dtype=torch.int8)

    groups_pow = 1
    while groups_pow < max(groups, 8):
        groups_pow *= 2
    block_k = 64

    grid = (s, h_kv, triton.cdiv(k, block_k))
    _fused_cluster_pass_rawq_kernel[grid](
        q, dim_offsets, dim_widths, th, centers, radii, out,
        h_q, h_kv, k,
        D=d, S=s, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out
