"""v1.27 attention kernel — on-demand raw-q gating without q packing.

This is a fixed-shape experiment for the current hot path:
  - bf = 4
  - S = 8
  - D = 128 with contiguous 16-wide subspaces

Unlike v1.22, the kernel does not rely on a prepacked q buffer. It loads each
subspace slice once per program from the raw q tensor, keeps the subspace
norms in registers, and derives the cluster-pass gates on demand.
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
    def _fused_attn_index_ondemand_rawq_fixed_kernel(
        Q_ptr,
        Th_ptr,
        KeysBlocksT_ptr,
        ValuesBlocks_ptr,
        Centers_ptr,
        Radii_ptr,
        AssignsBlocks_ptr,
        InvalidBlocks_ptr,
        M_out_ptr,
        L_out_ptr,
        O_out_ptr,
        H_Q,
        H_KV,
        K,
        D_V: tl.constexpr,
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

        d_full = tl.arange(0, 128)
        d_sub = tl.arange(0, 16)
        dv_range = tl.arange(0, D_V)

        q_full_f32 = tl.load(
            Q_ptr + hq_vec[:, None] * 128 + d_full[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        q_f16 = (q_full_f32 * SCALE).to(tl.float16)

        q0 = tl.load(Q_ptr + hq_vec[:, None] * 128 + d_sub[None, :], mask=g_valid[:, None], other=0.0)
        q1 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (16 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q2 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (32 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q3 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (48 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q4 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (64 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q5 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (80 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q6 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (96 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)
        q7 = tl.load(Q_ptr + hq_vec[:, None] * 128 + (112 + d_sub)[None, :], mask=g_valid[:, None], other=0.0)

        qn0 = tl.sqrt(tl.sum(q0 * q0, axis=1))
        qn1 = tl.sqrt(tl.sum(q1 * q1, axis=1))
        qn2 = tl.sqrt(tl.sum(q2 * q2, axis=1))
        qn3 = tl.sqrt(tl.sum(q3 * q3, axis=1))
        qn4 = tl.sqrt(tl.sum(q4 * q4, axis=1))
        qn5 = tl.sqrt(tl.sum(q5 * q5, axis=1))
        qn6 = tl.sqrt(tl.sum(q6 * q6, axis=1))
        qn7 = tl.sqrt(tl.sum(q7 * q7, axis=1))

        th0 = tl.load(Th_ptr + 0 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th1 = tl.load(Th_ptr + 1 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th2 = tl.load(Th_ptr + 2 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th3 = tl.load(Th_ptr + 3 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th4 = tl.load(Th_ptr + 4 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th5 = tl.load(Th_ptr + 5 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th6 = tl.load(Th_ptr + 6 * H_Q + hq_vec, mask=g_valid, other=float("inf"))
        th7 = tl.load(Th_ptr + 7 * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        m = tl.full([GROUPS_POW], -1.0e30, dtype=tl.float32)
        l_acc = tl.zeros([GROUPS_POW], dtype=tl.float32)
        o_acc = tl.zeros([GROUPS_POW, D_V], dtype=tl.float32)

        parents_per_split = (K + NUM_SPLITS - 1) // NUM_SPLITS
        p_start = split * parents_per_split
        p_end = tl.minimum(p_start + parents_per_split, K)

        cols = tl.arange(0, PARENTS_PER_PROG * 4)
        child_rel = cols % 4

        for p_chunk_start in range(p_start, p_end, PARENTS_PER_PROG):
            child_parent_idx = p_chunk_start + cols // 4
            child_valid = child_parent_idx < p_end
            child_parent_idx_safe = tl.where(child_valid, child_parent_idx, 0)
            out_mask = g_valid[:, None] & child_valid[None, :]

            anchor_centers = tl.load(
                Centers_ptr
                + ((0 * H_KV + kvh) * K + child_parent_idx_safe[:, None]) * 16
                + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            anchor_r = tl.load(
                Radii_ptr + (0 * H_KV + kvh) * K + child_parent_idx_safe,
                mask=child_valid,
                other=0.0,
            )
            anchor_cdot = tl.sum(q0[:, None, :] * anchor_centers[None, :, :], axis=2)
            survive = (
                anchor_cdot + anchor_r[None, :] * qn0[:, None] >= th0[:, None]
            ) & out_mask

            inv = tl.load(
                InvalidBlocks_ptr + ((kvh * K + child_parent_idx_safe) * 4 + child_rel),
                mask=child_valid,
                other=1,
            )
            survive = survive & (inv[None, :] == 0)

            assign1 = tl.load(
                AssignsBlocks_ptr + ((1 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers1 = tl.load(
                Centers_ptr + ((1 * H_KV + kvh) * K + assign1[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r1 = tl.load(Radii_ptr + (1 * H_KV + kvh) * K + assign1, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q1[:, None, :] * centers1[None, :, :], axis=2) + r1[None, :] * qn1[:, None]
                >= th1[:, None]
            )

            assign2 = tl.load(
                AssignsBlocks_ptr + ((2 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers2 = tl.load(
                Centers_ptr + ((2 * H_KV + kvh) * K + assign2[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r2 = tl.load(Radii_ptr + (2 * H_KV + kvh) * K + assign2, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q2[:, None, :] * centers2[None, :, :], axis=2) + r2[None, :] * qn2[:, None]
                >= th2[:, None]
            )

            assign3 = tl.load(
                AssignsBlocks_ptr + ((3 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers3 = tl.load(
                Centers_ptr + ((3 * H_KV + kvh) * K + assign3[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r3 = tl.load(Radii_ptr + (3 * H_KV + kvh) * K + assign3, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q3[:, None, :] * centers3[None, :, :], axis=2) + r3[None, :] * qn3[:, None]
                >= th3[:, None]
            )

            assign4 = tl.load(
                AssignsBlocks_ptr + ((4 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers4 = tl.load(
                Centers_ptr + ((4 * H_KV + kvh) * K + assign4[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r4 = tl.load(Radii_ptr + (4 * H_KV + kvh) * K + assign4, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q4[:, None, :] * centers4[None, :, :], axis=2) + r4[None, :] * qn4[:, None]
                >= th4[:, None]
            )

            assign5 = tl.load(
                AssignsBlocks_ptr + ((5 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers5 = tl.load(
                Centers_ptr + ((5 * H_KV + kvh) * K + assign5[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r5 = tl.load(Radii_ptr + (5 * H_KV + kvh) * K + assign5, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q5[:, None, :] * centers5[None, :, :], axis=2) + r5[None, :] * qn5[:, None]
                >= th5[:, None]
            )

            assign6 = tl.load(
                AssignsBlocks_ptr + ((6 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers6 = tl.load(
                Centers_ptr + ((6 * H_KV + kvh) * K + assign6[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r6 = tl.load(Radii_ptr + (6 * H_KV + kvh) * K + assign6, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q6[:, None, :] * centers6[None, :, :], axis=2) + r6[None, :] * qn6[:, None]
                >= th6[:, None]
            )

            assign7 = tl.load(
                AssignsBlocks_ptr + ((7 * H_KV + kvh) * K + child_parent_idx_safe) * 4 + child_rel,
                mask=child_valid,
                other=0,
            ).to(tl.int32)
            centers7 = tl.load(
                Centers_ptr + ((7 * H_KV + kvh) * K + assign7[:, None]) * 16 + d_sub[None, :],
                mask=child_valid[:, None],
                other=0.0,
            )
            r7 = tl.load(Radii_ptr + (7 * H_KV + kvh) * K + assign7, mask=child_valid, other=0.0)
            survive = survive & (
                tl.sum(q7[:, None, :] * centers7[None, :, :], axis=2) + r7[None, :] * qn7[:, None]
                >= th7[:, None]
            )

            live_cols = tl.max(survive.to(tl.int32), axis=0) != 0
            if tl.max(live_cols.to(tl.int32), axis=0) != 0:
                keys_tile = tl.load(
                    KeysBlocksT_ptr
                    + ((kvh * K + child_parent_idx_safe[None, :]) * 128 + d_full[:, None]) * 4
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
                    + ((kvh * K + child_parent_idx_safe[:, None]) * 4 + child_rel[:, None]) * D_V
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


def run_fused_attn_index_ondemand_rawq_fixed(
    q: torch.Tensor,
    th: torch.Tensor,
    keys_blocks_t_f16: torch.Tensor,
    values_blocks_f16: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    assigns_blocks: torch.Tensor,
    invalid_blocks_i8: torch.Tensor,
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
    num_stages: int = 3,
) -> None:
    d_v = values_blocks_f16.shape[-1]
    _fused_attn_index_ondemand_rawq_fixed_kernel[(h_kv_eff, num_splits)](
        q,
        th,
        keys_blocks_t_f16,
        values_blocks_f16,
        centers,
        radii,
        assigns_blocks,
        invalid_blocks_i8,
        out_m,
        out_l,
        out_o,
        h_q,
        h_kv_eff,
        k,
        D_V=d_v,
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        PARENTS_PER_PROG=parents_per_prog,
        NUM_SPLITS=num_splits,
        SCALE=float(scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
