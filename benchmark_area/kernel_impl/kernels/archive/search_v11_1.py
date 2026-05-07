"""search_v11.1 — v11 path with full-fp16 build/search."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from ._search_fp16_utils import (
    _get_layout_v15_fp16,
    _next_pow2,
    buffer_dot_to_dtype,
    pack_query_for_fp16_search,
)
from ._search_triton import triton_fused_cluster_pass

KERNEL_VERSION = "v11.1"
_PARENTS_PER_PROG = 8


if HAS_TRITON:

    @triton.jit
    def _fused_anchor_parent_batch_kernel_fp16(
        Q_ptr,              # (H_q, D) f16
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        ClusterPass_ptr,    # (S, H_q, K) int8
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        Out_ptr,            # (H_q, N_pad) f16
        H_Q,
        H_KV,
        K,
        N_PAD,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        parent_block = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        parent_idx = parent_block * PARENTS_PER_PROG + parent_rel
        col_valid = parent_idx < K
        parent_idx_safe = tl.where(col_valid, parent_idx, 0)
        child_idx = parent_idx_safe * BF + child_rel

        out_offs = hq_vec[:, None] * N_PAD + child_idx[None, :]
        out_mask = g_valid[:, None] & col_valid[None, :]

        anchor_pass = tl.load(
            ClusterPass_ptr + (ANCHOR_S * H_Q + hq_vec[:, None]) * K + parent_idx_safe[None, :],
            mask=out_mask,
            other=0,
        )
        survive = (anchor_pass != 0) & out_mask

        inv = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
            mask=col_valid,
            other=1,
        )
        survive = survive & (inv[None, :] == 0)

        for s in tl.static_range(0, S):
            if s != ANCHOR_S:
                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s * H_KV + kvh) * K + parent_idx_safe) * BF
                    + child_rel,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq_vec[:, None]) * K + assign[None, :],
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

        live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        d_range = tl.arange(0, D)
        q_full = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
            + child_rel[None, :],
            mask=live_cols[None, :],
            other=0.0,
        )
        acc = tl.dot(q_full, keys_tile)

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + out_offs, out, mask=out_mask)


def search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise RuntimeError("search_v11_1 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v11_1 requires build_v2-style state")

    layout = _get_layout_v15_fp16(state, q_head_to_kv, q, "_search_v11_1_cache")
    q_cast, q_packed, q_norm, th_packed = pack_query_for_fp16_search(q, th_per_subspace, layout)

    cluster_pass_flat = triton_fused_cluster_pass(
        q_packed,
        q_norm,
        th_packed,
        layout["centers"],
        layout["radii"],
        layout["groups"],
    )

    h_q = q_cast.shape[0]
    d = q_cast.shape[1]
    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    out = torch.empty(h_q, n_pad, device=q_cast.device, dtype=q_cast.dtype)
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))
    _fused_anchor_parent_batch_kernel_fp16[grid](
        q_cast.contiguous(),
        layout["keys_blocks_t_f16"],
        layout["assigns_blocks"],
        cluster_pass_flat,
        layout["invalid_blocks_i8"],
        out,
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=layout["num_subspaces"],
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
        num_warps=4,
    )

    buf_shim = {
        "mode": layout["mode"],
        "groups": groups,
        "base_heads": h_kv,
    }
    buf_dots = buffer_dot_to_dtype(q_cast, buffer_keys, q_head_to_kv, buf_shim, out.dtype)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
