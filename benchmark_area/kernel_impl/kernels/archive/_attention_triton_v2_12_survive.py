"""v2.12 child_survive pre-computation kernel.

Fuses cluster_pass + assigns_blocks + invalid_blocks into a single
per-child, per-query-head int8 survive mask.  The main attention kernel
then only needs one cheap contiguous load per parent chunk instead of
2*(S-1) scattered gather loads for the multi-subspace AND filter.
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
    def _child_survive_kernel(
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


def run_child_survive(
    cluster_pass: torch.Tensor,
    assigns_blocks: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    h_q: int,
    h_kv: int,
    k: int,
    bf: int,
    groups: int,
    s_subspaces: int,
    anchor_s: int,
    out: torch.Tensor,
) -> None:
    total_children = k * bf
    BLOCK_C = 256
    grid = (h_q, triton.cdiv(total_children, BLOCK_C))
    _child_survive_kernel[grid](
        cluster_pass,
        assigns_blocks,
        invalid_blocks_i8,
        out,
        h_q,
        h_kv,
        k,
        GROUPS=groups,
        BF=bf,
        S=s_subspaces,
        ANCHOR_S=anchor_s,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
