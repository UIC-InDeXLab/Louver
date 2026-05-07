"""v2.24 fused cluster_pass + child_survive kernel.

Each block handles one (hq, child_block). For each non-anchor subspace,
it computes cluster_pass inline for the assigned parents, avoiding the
separate cluster_pass kernel launch entirely.

Pipeline: 1 launch instead of 2 (cluster_pass + child_survive).
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
    def _fused_survive_kernel(
        Q_ptr,
        QNorms_ptr,
        Th_ptr,
        DimOffsets_ptr,
        DimWidths_ptr,
        Centers_ptr,
        Radii_ptr,
        AssignsBlocks_ptr,
        InvalidBlocks_ptr,
        Out_ptr,
        H_Q,
        H_KV,
        K,
        D: tl.constexpr,
        S: tl.constexpr,
        ANCHOR_S: tl.constexpr,
        GROUPS: tl.constexpr,
        BF: tl.constexpr,
        MAX_D: tl.constexpr,
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
        d_range = tl.arange(0, MAX_D)

        # --- Anchor subspace check ---
        anchor_dim_off = tl.load(DimOffsets_ptr + ANCHOR_S)
        anchor_width = tl.load(DimWidths_ptr + ANCHOR_S)
        anchor_d_valid = d_range < anchor_width

        q_anchor = tl.load(
            Q_ptr + hq * D + (anchor_dim_off + d_range),
            mask=anchor_d_valid,
            other=0.0,
        )
        qn_anchor = tl.load(QNorms_ptr + ANCHOR_S * H_Q + hq).to(tl.float32)
        th_anchor = tl.load(Th_ptr + ANCHOR_S * H_Q + hq).to(tl.float32)

        # centers_anchor[parent_idx, :MAX_D]
        c_anchor_offs = (ANCHOR_S * H_KV + kvh) * K * MAX_D + parent_idx[:, None] * MAX_D + d_range[None, :]
        centers_a = tl.load(Centers_ptr + c_anchor_offs, mask=c_valid[:, None], other=0.0)
        r_a = tl.load(
            Radii_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx,
            mask=c_valid,
            other=0.0,
        ).to(tl.float32)

        dot_a = tl.sum(q_anchor[None, :] * centers_a, axis=1).to(tl.float32)
        ub_a = dot_a + r_a * qn_anchor
        survive = (ub_a >= th_anchor) & c_valid

        # --- Invalid check ---
        inv = tl.load(
            InvalidBlocks_ptr + (kvh * K + parent_idx) * BF + child_rel,
            mask=c_valid,
            other=1,
        )
        survive = survive & (inv == 0)

        # --- Non-anchor subspace checks ---
        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                dim_off_s = tl.load(DimOffsets_ptr + s)
                width_s = tl.load(DimWidths_ptr + s)
                d_valid_s = d_range < width_s

                q_s = tl.load(
                    Q_ptr + hq * D + (dim_off_s + d_range),
                    mask=d_valid_s,
                    other=0.0,
                )
                qn_s = tl.load(QNorms_ptr + s * H_Q + hq).to(tl.float32)
                th_s = tl.load(Th_ptr + s * H_Q + hq).to(tl.float32)

                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx) * BF
                    + child_rel,
                    mask=survive,
                    other=0,
                ).to(tl.int32)

                c_offs_s = (s * H_KV + kvh) * K * MAX_D + assign[:, None] * MAX_D + d_range[None, :]
                centers_s = tl.load(Centers_ptr + c_offs_s, mask=survive[:, None], other=0.0)
                r_s = tl.load(
                    Radii_ptr + (s * H_KV + kvh) * K + assign,
                    mask=survive,
                    other=0.0,
                ).to(tl.float32)

                dot_s = tl.sum(q_s[None, :] * centers_s, axis=1).to(tl.float32)
                ub_s = dot_s + r_s * qn_s
                survive = survive & (ub_s >= th_s)

        tl.store(Out_ptr + hq * total_children + c_range, survive.to(tl.int8), mask=c_valid)


def run_fused_survive(
    q, q_norms, th, dim_offsets, dim_widths,
    centers, radii, assigns_blocks, invalid_blocks_i8,
    h_q, h_kv, k, bf, groups, s_subspaces, anchor_s, out,
    block_c: int = 256,
    num_warps: int = 4,
):
    d = q.shape[1]
    max_d = centers.shape[3]
    total_children = k * bf
    grid = (h_q, triton.cdiv(total_children, block_c))
    _fused_survive_kernel[grid](
        q, q_norms, th, dim_offsets, dim_widths,
        centers, radii, assigns_blocks, invalid_blocks_i8, out,
        h_q, h_kv, k,
        D=d, S=s_subspaces, ANCHOR_S=anchor_s,
        GROUPS=groups, BF=bf, MAX_D=max_d,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )
