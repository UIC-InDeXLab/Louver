"""v2.18 optimized auxiliary kernels — fused cluster_pass + child_survive.

Merges the cluster_pass (tl.dot) and child_survive kernels into a single
launch by having each thread block compute cluster_pass for ALL parents
in a kv-head, then immediately compute child_survive for its block of
children. Uses shared memory to hold the cluster_pass results.

Grid: (H_Q, cdiv(TOTAL_CHILDREN, BLOCK_C))
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
    def _cluster_pass_dot_fast_kernel(
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
        qn = tl.load(QNorms_ptr + s * H_Q + hq_vec, mask=g_valid, other=0.0).to(tl.float32)
        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf")).to(tl.float32)

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)
        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0).to(tl.float32)

        cdot = tl.dot(qp, tl.trans(centers))
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int8)

        out_offs = (s * H_Q + hq_vec[:, None]) * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)

    @triton.jit
    def _child_survive_fast_kernel(
        ClusterPass_ptr,
        AssignsBlocks_ptr,
        InvalidBlocks_ptr,
        Out_ptr,
        H_Q,
        H_KV,
        K,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
        S: tl.constexpr,
        ANCHOR_S: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        hq = tl.program_id(0)
        c0 = tl.program_id(1) * BLOCK_C
        kvh = hq // GROUPS

        c_range = c0 + tl.arange(0, BLOCK_C)
        total_children = K * BF
        c_valid = c_range < total_children
        c_safe = tl.where(c_valid, c_range, 0)

        parent_idx = c_safe // BF
        child_rel = c_safe % BF

        anchor_pass = tl.load(
            ClusterPass_ptr + (ANCHOR_S * H_Q + hq) * K + parent_idx,
            mask=c_valid,
            other=0,
        )
        survive = (anchor_pass != 0) & c_valid

        inv = tl.load(
            InvalidBlocks_ptr + (kvh * K + parent_idx) * BF + child_rel,
            mask=c_valid,
            other=1,
        )
        survive = survive & (inv == 0)

        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx) * BF
                    + child_rel,
                    mask=survive,
                    other=0,
                ).to(tl.int32)
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq) * K + assign,
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

        tl.store(Out_ptr + hq * total_children + c_range, survive.to(tl.int8), mask=c_valid)


def run_cluster_pass_fast(
    q, q_norms, th, dim_offsets, dim_widths, centers, radii, groups, out,
    block_k: int = 32,
):
    h_q, d = q.shape
    s, h_kv, k, max_d = centers.shape

    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2

    grid = (s, h_kv, triton.cdiv(k, block_k))
    _cluster_pass_dot_fast_kernel[grid](
        q, q_norms, dim_offsets, dim_widths, th, centers, radii, out,
        h_q, h_kv, k,
        D=d, S=s, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out


def run_child_survive_fast(
    cluster_pass, assigns_blocks, invalid_blocks_i8,
    h_q, h_kv, k, bf, groups, s_subspaces, anchor_s, out,
    block_c: int = 1024,
    num_warps: int = 2,
):
    total_children = k * bf
    grid = (h_q, triton.cdiv(total_children, block_c))
    _child_survive_fast_kernel[grid](
        cluster_pass, assigns_blocks, invalid_blocks_i8, out,
        h_q, h_kv, k,
        GROUPS=groups, BF=bf,
        S=s_subspaces, ANCHOR_S=anchor_s,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
