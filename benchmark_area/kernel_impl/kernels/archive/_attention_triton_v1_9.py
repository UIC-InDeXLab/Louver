"""Packed-gate Triton kernels for fixed BF=4 sparse attention experiments."""

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
    def _fused_cluster_pass_packed_kernel(
        Q_ptr,
        DimOffsets_ptr,
        DimWidths_ptr,
        Th_ptr,
        Centers_ptr,
        Radii_ptr,
        Out_ptr,           # (S, H_KV, K) u8; low GROUPS bits pack pass[g]
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

        qp = tl.load(
            Q_ptr + hq_vec[:, None] * D + (dim_off + d_range)[None, :],
            mask=g_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        qn = tl.sqrt(tl.sum(qp * qp, axis=1))
        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)
        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)
        ub = cdot + r[None, :] * qn[:, None]
        passed = (ub >= th[:, None]).to(tl.int32)
        bit_weights = (1 << g_range).to(tl.int32)
        packed = tl.sum(passed * bit_weights[:, None], axis=0)

        tl.store(
            Out_ptr + (s * H_KV + kvh) * K + k_range,
            packed.to(tl.uint8),
            mask=k_mask,
        )

    @triton.jit
    def _fused_attn_index_packed_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        AssignsBlocks_ptr,
        PackedPass_ptr,      # (S, H_KV, K) u8
        InvalidBlocks_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
        K,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        S: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        split = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            parent_idx = p_chunk_start + parent_rel
            col_valid = parent_idx < p_end
            parent_idx_safe = tl.where(col_valid, parent_idx, 0)
            out_mask = g_valid[:, None] & col_valid[None, :]

            anchor_bits = tl.load(
                PackedPass_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx_safe,
                mask=col_valid,
                other=0,
            ).to(tl.int32)
            survive = (((anchor_bits[None, :] >> g_range[:, None]) & 1) != 0) & out_mask

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
                    passed_bits = tl.load(
                        PackedPass_ptr + (s * H_KV + kvh) * K + assign,
                        mask=col_valid,
                        other=0,
                    ).to(tl.int32)
                    passed = ((passed_bits[None, :] >> g_range[:, None]) & 1) != 0
                    survive = survive & passed

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0

            keys_tile = tl.load(
                KeysBlocksT_ptr
                + ((kvh * K + parent_idx_safe[None, :]) * D + d_range[:, None]) * BF
                + child_rel[None, :],
                mask=live_cols[None, :],
                other=0.0,
            )
            scores = tl.dot(q_f16, keys_tile)
            scores = tl.where(survive, scores, -1.0e30)

            chunk_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(survive, p, 0.0)
            l_acc = alpha * l_acc + tl.sum(p, axis=1)

            v_tile = tl.load(
                ValuesBlocks_ptr
                + ((kvh * K + parent_idx_safe[:, None]) * BF + child_rel[:, None]) * D_V
                + dv_range[None, :],
                mask=live_cols[:, None],
                other=0.0,
            )
            o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
            m = m_new

        tl.store(M_out_ptr + hq_vec * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + (hq_vec[:, None] * NUM_SPLITS + split) * D_V + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )

    @triton.jit
    def _compact_live_parents_kernel(
        AssignsBlocks_ptr,
        PackedPass_ptr,
        InvalidBlocks_ptr,
        ParentIds_ptr,       # (H_KV, K) i32
        ChildMasks_ptr,      # (H_KV, K) i32; low (GROUPS*BF) bits used
        Counts_ptr,          # (H_KV,) i32
        H_KV,
        K,
        ANCHOR_S: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        S: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        parent_idx = tl.program_id(1)
        if parent_idx >= K:
            return

        live_anchor = tl.load(
            PackedPass_ptr + (ANCHOR_S * H_KV + kvh) * K + parent_idx
        ).to(tl.int32)
        if live_anchor == 0:
            return

        packed_mask = 0
        for child in tl.static_range(0, BF):
            invalid = tl.load(
                InvalidBlocks_ptr + ((kvh * K + parent_idx) * BF + child)
            )
            if invalid == 0:
                live_bits = live_anchor
                for s in tl.static_range(0, S):
                    if s != ANCHOR_S:
                        assign = tl.load(
                            AssignsBlocks_ptr
                            + ((s * H_KV + kvh) * K + parent_idx) * BF
                            + child
                        ).to(tl.int32)
                        live_bits = live_bits & tl.load(
                            PackedPass_ptr + (s * H_KV + kvh) * K + assign
                        ).to(tl.int32)
                for g in tl.static_range(0, 8):
                    if g < GROUPS:
                        if ((live_bits >> g) & 1) != 0:
                            packed_mask = packed_mask | (1 << (g * BF + child))

        if packed_mask != 0:
            slot = tl.atomic_add(Counts_ptr + kvh, 1)
            tl.store(ParentIds_ptr + kvh * K + slot, parent_idx)
            tl.store(ChildMasks_ptr + kvh * K + slot, packed_mask)

    @triton.jit
    def _fused_attn_index_parentlist_kernel(
        Q_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        ParentIds_ptr,       # (H_KV, K) i32
        ChildMasks_ptr,      # (H_KV, K) i32
        Counts_ptr,          # (H_KV,) i32
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
        K,
        D: tl.constexpr,
        D_V: tl.constexpr,
        BF: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        PARENTS_PER_PROG: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        kvh = tl.program_id(0)
        split = tl.program_id(1)

        g_range = tl.arange(0, GROUPS_POW)
        g_valid = g_range < GROUPS
        hq_vec = kvh * GROUPS + g_range

        d_range = tl.arange(0, D)
        dv_range = tl.arange(0, D_V)

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        count = tl.load(Counts_ptr + kvh)
        parents_per_split = (count + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, count)

        cols = tl.arange(0, PARENTS_PER_PROG * BF)
        parent_rel = cols // BF
        child_rel = cols % BF

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            list_idx = p_chunk_start + parent_rel
            col_valid = list_idx < p_end
            list_idx_safe = tl.where(col_valid, list_idx, 0)

            parent_idx = tl.load(
                ParentIds_ptr + kvh * K + list_idx_safe,
                mask=col_valid,
                other=0,
            ).to(tl.int32)
            child_masks = tl.load(
                ChildMasks_ptr + kvh * K + list_idx_safe,
                mask=col_valid,
                other=0,
            ).to(tl.int32)

            shifts = g_range[:, None] * BF + child_rel[None, :]
            survive = (((child_masks[None, :] >> shifts) & 1) != 0) & (
                g_valid[:, None] & col_valid[None, :]
            )

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0

            keys_tile = tl.load(
                KeysBlocksT_ptr
                + ((kvh * K + parent_idx[None, :]) * D + d_range[:, None]) * BF
                + child_rel[None, :],
                mask=live_cols[None, :],
                other=0.0,
            )
            scores = tl.dot(q_f16, keys_tile)
            scores = tl.where(survive, scores, -1.0e30)

            chunk_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m, chunk_max)
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(survive, p, 0.0)
            l_acc = alpha * l_acc + tl.sum(p, axis=1)

            v_tile = tl.load(
                ValuesBlocks_ptr
                + ((kvh * K + parent_idx[:, None]) * BF + child_rel[:, None]) * D_V
                + dv_range[None, :],
                mask=live_cols[:, None],
                other=0.0,
            )
            o_acc = alpha[:, None] * o_acc + tl.dot(p.to(tl.float16), v_tile)
            m = m_new

        tl.store(M_out_ptr + hq_vec * NUM_SPLITS + split, m, mask=g_valid)
        tl.store(L_out_ptr + hq_vec * NUM_SPLITS + split, l_acc, mask=g_valid)
        tl.store(
            O_out_ptr + (hq_vec[:, None] * NUM_SPLITS + split) * D_V + dv_range[None, :],
            o_acc,
            mask=g_valid[:, None],
        )


def triton_fused_cluster_pass_packed(
    q: torch.Tensor,
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
        out = torch.empty(s, h_kv, k, device=q.device, dtype=torch.uint8)

    groups_pow = 1
    while groups_pow < max(groups, 4):
        groups_pow *= 2

    block_k = 64
    grid = (s, h_kv, triton.cdiv(k, block_k))
    _fused_cluster_pass_packed_kernel[grid](
        q,
        dim_offsets,
        dim_widths,
        th,
        centers,
        radii,
        out,
        h_q,
        h_kv,
        k,
        D=d,
        S=s,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        MAX_D=max_d,
        BLOCK_K=block_k,
        num_warps=2,
    )
    return out


def run_fused_attn_index_packed(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    assigns_blocks: torch.Tensor,
    packed_pass: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    s_subspaces: int,
    parents_per_prog: int,
    num_splits: int,
    anchor_s: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_packed_kernel[grid](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        assigns_blocks,
        packed_pass,
        invalid_blocks_i8,
        out_m,
        out_l,
        out_o,
        h_q,
        h_kv_eff,
        k,
        ANCHOR_S=anchor_s,
        D=d,
        D_V=d_v,
        BF=values_blocks_f16.shape[2],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        S=s_subspaces,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
    )


def run_compact_live_parents(
    assigns_blocks: torch.Tensor,
    packed_pass: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
    parent_ids: torch.Tensor,
    child_masks: torch.Tensor,
    counts: torch.Tensor,
    *,
    anchor_s: int,
    groups: int,
    s_subspaces: int,
) -> None:
    h_kv_eff = packed_pass.shape[1]
    k = packed_pass.shape[2]
    counts.zero_()
    grid = (h_kv_eff, k)
    _compact_live_parents_kernel[grid](
        assigns_blocks,
        packed_pass,
        invalid_blocks_i8,
        parent_ids,
        child_masks,
        counts,
        h_kv_eff,
        k,
        ANCHOR_S=anchor_s,
        BF=invalid_blocks_i8.shape[2],
        GROUPS=groups,
        S=s_subspaces,
        num_warps=2,
    )


def run_fused_attn_index_parentlist(
    q: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    parent_ids: torch.Tensor,
    child_masks: torch.Tensor,
    counts: torch.Tensor,
    h_q: int,
    h_kv_eff: int,
    k: int,
    groups: int,
    groups_pow: int,
    parents_per_prog: int,
    num_splits: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
    num_warps: int = 4,
) -> None:
    d = q.shape[1]
    d_v = values_blocks_f16.shape[-1]
    grid = (h_kv_eff, num_splits)
    _fused_attn_index_parentlist_kernel[grid](
        q,
        keys_blocks_t_f16,
        values_blocks_f16,
        parent_ids,
        child_masks,
        counts,
        out_m,
        out_l,
        out_o,
        h_q,
        h_kv_eff,
        k,
        D=d,
        D_V=d_v,
        BF=values_blocks_f16.shape[2],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
    )
