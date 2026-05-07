"""search_v18.0 — v15 layout with on-demand cluster gating.

This variant keeps v15's fp16 key path and block-packed build state, but
stops materializing the full ``cluster_pass`` tensor. Instead, the search
kernel computes anchor and non-anchor upper bounds directly from
``q_packed`` / centers / radii while it is visiting each parent block.

It is intentionally isolated as a separate version so benchmark results can
show whether removing the intermediate pass buffer is actually a win.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False

from .._search_utils import buffer_dot
from .search_v15_0 import _get_layout_v15

KERNEL_VERSION = "v18.0"
_PARENTS_PER_PROG = 4


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


if HAS_TRITON:

    @triton.jit
    def _ondemand_cluster_gate_kernel(
        Q_ptr,              # (H_q, D) f32
        QPacked_ptr,        # (S, H_q, MAX_D) f32
        QNorm_ptr,          # (S, H_q) f32
        Th_ptr,             # (S, H_q) f32
        KeysBlocksT_ptr,    # (H_kv, K, D, BF) f16
        Centers_ptr,        # (S, H_kv, K, MAX_D) f32
        Radii_ptr,          # (S, H_kv, K) f32
        AssignsBlocks_ptr,  # (S, H_kv, K, BF)
        InvalidBlocks_ptr,  # (H_kv, K, BF) int8
        Out_ptr,            # (H_q, N_pad) f32
        H_Q,
        H_KV,
        K,
        N_PAD,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        MAX_D: tl.constexpr,
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

        d_sub = tl.arange(0, MAX_D)

        q_anchor = tl.load(
            QPacked_ptr + (ANCHOR_S * H_Q + hq_vec[:, None]) * MAX_D + d_sub[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        qn_anchor = tl.load(
            QNorm_ptr + ANCHOR_S * H_Q + hq_vec,
            mask=g_valid,
            other=0.0,
        )
        th_anchor = tl.load(
            Th_ptr + ANCHOR_S * H_Q + hq_vec,
            mask=g_valid,
            other=float("inf"),
        )

        anchor_centers = tl.load(
            Centers_ptr
            + (((ANCHOR_S * H_KV + kvh) * K + parent_idx_safe[:, None]) * MAX_D + d_sub[None, :]),
            mask=col_valid[:, None],
            other=0.0,
        )
        anchor_r = tl.load(
            Radii_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx_safe,
            mask=col_valid,
            other=0.0,
        )
        anchor_cdot = tl.sum(
            q_anchor[:, None, :] * anchor_centers[None, :, :],
            axis=2,
        )
        survive = (anchor_cdot + anchor_r[None, :] * qn_anchor[:, None] >= th_anchor[:, None]) & out_mask

        inv = tl.load(
            InvalidBlocks_ptr + ((kvh * K + parent_idx_safe) * BF + child_rel),
            mask=col_valid,
            other=1,
        )
        survive = survive & (inv[None, :] == 0)

        for s_idx in tl.static_range(0, S):
            if s_idx != ANCHOR_S:
                q_sub = tl.load(
                    QPacked_ptr + (s_idx * H_Q + hq_vec[:, None]) * MAX_D + d_sub[None, :],
                    mask=g_valid[:, None],
                    other=0.0,
                )
                qn_sub = tl.load(
                    QNorm_ptr + s_idx * H_Q + hq_vec,
                    mask=g_valid,
                    other=0.0,
                )
                th_sub = tl.load(
                    Th_ptr + s_idx * H_Q + hq_vec,
                    mask=g_valid,
                    other=float("inf"),
                )

                assign = tl.load(
                    AssignsBlocks_ptr
                    + ((s_idx * H_KV + kvh) * K + parent_idx_safe) * BF
                    + child_rel,
                    mask=col_valid,
                    other=0,
                ).to(tl.int32)
                child_centers = tl.load(
                    Centers_ptr
                    + (((s_idx * H_KV + kvh) * K + assign[:, None]) * MAX_D + d_sub[None, :]),
                    mask=col_valid[:, None],
                    other=0.0,
                )
                child_r = tl.load(
                    Radii_ptr + (s_idx * H_KV + kvh) * K + assign,
                    mask=col_valid,
                    other=0.0,
                )
                child_cdot = tl.sum(
                    q_sub[:, None, :] * child_centers[None, :, :],
                    axis=2,
                )
                survive = survive & (
                    child_cdot + child_r[None, :] * qn_sub[:, None] >= th_sub[:, None]
                )

        live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
        if tl.max(live_cols.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + out_offs, neg_inf, mask=out_mask)
            return

        d_full = tl.arange(0, D)
        q_full = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_full[None, :],
            mask=g_valid[:, None],
            other=0.0,
        ).to(tl.float16)

        keys_tile = tl.load(
            KeysBlocksT_ptr
            + ((kvh * K + parent_idx_safe[None, :]) * D + d_full[:, None]) * BF
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
        raise RuntimeError("search_v18_0 requires Triton")
    if "keys_reord" not in state:
        raise RuntimeError("search_v18_0 requires build_v2-style state")

    layout = _get_layout_v15(state, q_head_to_kv, q)
    h_q = q.shape[0]
    s = layout["num_subspaces"]
    max_d = layout["max_d"]
    d = q.shape[1]

    if d == s * max_d:
        q_packed = q.view(h_q, s, max_d).transpose(0, 1).contiguous()
    else:
        q_packed = q.new_zeros(s, h_q, max_d)
        for si, (s0, e0) in enumerate(layout["dim_slices"]):
            q_packed[si, :, : e0 - s0] = q[:, s0:e0]
        q_packed = q_packed.contiguous()
    q_norm = q_packed.norm(dim=-1).contiguous()
    th_packed = th_per_subspace.reshape(s, h_q).contiguous()

    h_kv = layout["base_heads"]
    k = layout["K"]
    bf = layout["bf"]
    n_pad = layout["N_pad"]
    groups = layout["groups"]
    groups_pow = max(_next_pow2(groups), 8)
    anchor_s = layout["anchor_subspace"]

    out = torch.empty(h_q, n_pad, device=q.device, dtype=torch.float32)
    grid = (h_kv, triton.cdiv(k, _PARENTS_PER_PROG))
    _ondemand_cluster_gate_kernel[grid](
        q.contiguous(),
        q_packed,
        q_norm,
        th_packed,
        layout["keys_blocks_t_f16"],
        layout["centers"],
        layout["radii"],
        layout["assigns_blocks"],
        layout["invalid_blocks_i8"],
        out,
        h_q,
        h_kv,
        k,
        n_pad,
        ANCHOR_S=anchor_s,
        D=d,
        MAX_D=max_d,
        BF=bf,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=s,
        PARENTS_PER_PROG=_PARENTS_PER_PROG,
        num_warps=4,
    )

    buf_shim = {"mode": layout["mode"], "groups": groups, "base_heads": h_kv}
    buf_dots = buffer_dot(q, buffer_keys, q_head_to_kv, buf_shim)
    return out if buf_dots is None else torch.cat([out, buf_dots], dim=1)


KERNEL = search
