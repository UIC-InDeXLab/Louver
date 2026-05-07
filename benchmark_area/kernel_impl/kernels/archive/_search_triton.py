"""Triton search kernels shared by search_v4+."""

from __future__ import annotations

import torch

from ._search_utils import cluster_pass_only, dense_index_search, flatten_cluster_pass, pack_query_subspaces

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _clusterpass_search_kernel(
        Q_ptr,
        KeysT_ptr,
        Assigns_ptr,
        ClusterPass_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        GROUPS,
        N,
        K,
        D: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        neg_inf = float("-inf")
        hq = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        ns = n0 + tl.arange(0, BLOCK_N)
        nmask = ns < N
        kvh = hq // GROUPS

        survive = nmask
        for s in range(NUM_SUBSPACES):
            assign = tl.load(
                Assigns_ptr + (s * BASE_HEADS + kvh) * N + ns,
                mask=survive,
                other=0,
            )
            passed = tl.load(
                ClusterPass_ptr + (s * H_Q + hq) * K + assign,
                mask=survive,
                other=0,
            )
            survive = survive & (passed != 0)

        if tl.max(survive.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + hq * N + ns, neg_inf, mask=nmask)
            return

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(D):
            qv = tl.load(Q_ptr + hq * D + d)
            kv = tl.load(
                KeysT_ptr + (kvh * D + d) * N + ns,
                mask=survive,
                other=0.0,
            )
            acc += qv * kv

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + hq * N + ns, out, mask=nmask)

    @triton.jit
    def _direct_search_kernel(
        Q_ptr,
        QPacked_ptr,
        QNorm_ptr,
        KeysT_ptr,
        Centers_ptr,
        Radii_ptr,
        Widths_ptr,
        Assigns_ptr,
        Th_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        GROUPS,
        N,
        K,
        D: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        MAX_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        neg_inf = float("-inf")
        gate_eps = 1e-6
        hq = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        ns = n0 + tl.arange(0, BLOCK_N)
        nmask = ns < N
        kvh = hq // GROUPS

        survive = nmask

        for s in range(NUM_SUBSPACES):
            width = tl.load(Widths_ptr + s)
            assign = tl.load(
                Assigns_ptr + (s * BASE_HEADS + kvh) * N + ns,
                mask=survive,
                other=0,
            )
            center_dot = tl.zeros([BLOCK_N], dtype=tl.float32)
            for d_idx in range(MAX_D):
                qv = tl.load(
                    QPacked_ptr + (s * H_Q + hq) * MAX_D + d_idx,
                    mask=d_idx < width,
                    other=0.0,
                )
                center = tl.load(
                    Centers_ptr
                    + ((s * BASE_HEADS + kvh) * K + assign) * MAX_D
                    + d_idx,
                    mask=survive & (d_idx < width),
                    other=0.0,
                )
                center_dot += center * qv
            ub = center_dot + tl.load(
                Radii_ptr + (s * BASE_HEADS + kvh) * K + assign,
                mask=survive,
                other=0.0,
            ) * tl.load(QNorm_ptr + s * H_Q + hq)
            survive = survive & (ub + gate_eps >= tl.load(Th_ptr + s * H_Q + hq))

        if tl.max(survive.to(tl.int32), axis=0) == 0:
            tl.store(Out_ptr + hq * N + ns, neg_inf, mask=nmask)
            return

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(D):
            qv = tl.load(Q_ptr + hq * D + d)
            kv = tl.load(
                KeysT_ptr + (kvh * D + d) * N + ns,
                mask=survive,
                other=0.0,
            )
            acc += qv * kv

        out = tl.where(survive, acc, neg_inf)
        tl.store(Out_ptr + hq * N + ns, out, mask=nmask)


if HAS_TRITON:
    @triton.jit
    def _anchor_cluster_kernel(
        Q_ptr,
        KeysT_ptr,
        ChildOrder_ptr,
        ChildOffsets_ptr,
        Assigns_ptr,
        ClusterPass_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        GROUPS,
        N,
        K,
        K_PLUS1,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        MAX_CC: tl.constexpr,
    ):
        hq = tl.program_id(0)
        c = tl.program_id(1)
        kvh = hq // GROUPS

        anchor_pass = tl.load(ClusterPass_ptr + (ANCHOR_S * H_Q + hq) * K + c)
        if anchor_pass == 0:
            return

        off_start = tl.load(
            ChildOffsets_ptr + (ANCHOR_S * BASE_HEADS + kvh) * K_PLUS1 + c
        )
        off_end = tl.load(
            ChildOffsets_ptr + (ANCHOR_S * BASE_HEADS + kvh) * K_PLUS1 + c + 1
        )
        count = off_end - off_start

        cc = tl.arange(0, MAX_CC)
        cc_mask = cc < count

        child_idx = tl.load(
            ChildOrder_ptr
            + (ANCHOR_S * BASE_HEADS + kvh) * N
            + off_start
            + cc,
            mask=cc_mask,
            other=0,
        )

        survive = cc_mask
        for s in range(NUM_SUBSPACES):
            if s != ANCHOR_S:
                assign = tl.load(
                    Assigns_ptr + (s * BASE_HEADS + kvh) * N + child_idx,
                    mask=survive,
                    other=0,
                )
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq) * K + assign,
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

        if tl.max(survive.to(tl.int32), axis=0) == 0:
            return

        acc = tl.zeros([MAX_CC], dtype=tl.float32)
        for d in range(D):
            qv = tl.load(Q_ptr + hq * D + d)
            kv = tl.load(
                KeysT_ptr + (kvh * D + d) * N + child_idx,
                mask=survive,
                other=0.0,
            )
            acc += qv * kv

        tl.store(Out_ptr + hq * N + child_idx, acc, mask=survive)


if HAS_TRITON:
    @triton.jit
    def _anchor_cluster_batched_kernel(
        Q_ptr,
        KeysT_ptr,
        ChildOrder_ptr,
        ChildOffsets_ptr,
        Assigns_ptr,
        ClusterPass_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        GROUPS,
        N,
        K,
        K_PLUS1,
        ANCHOR_S: tl.constexpr,
        D: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        MAX_CC: tl.constexpr,
        CLUSTERS_PER_PROG: tl.constexpr,
    ):
        hq = tl.program_id(0)
        cb = tl.program_id(1)
        kvh = hq // GROUPS

        c0 = cb * CLUSTERS_PER_PROG
        for ci in range(CLUSTERS_PER_PROG):
            c = c0 + ci
            if c < K:
                anchor_pass = tl.load(
                    ClusterPass_ptr + (ANCHOR_S * H_Q + hq) * K + c
                )
                if anchor_pass != 0:
                    off_start = tl.load(
                        ChildOffsets_ptr
                        + (ANCHOR_S * BASE_HEADS + kvh) * K_PLUS1
                        + c
                    )
                    off_end = tl.load(
                        ChildOffsets_ptr
                        + (ANCHOR_S * BASE_HEADS + kvh) * K_PLUS1
                        + c
                        + 1
                    )
                    count = off_end - off_start

                    cc = tl.arange(0, MAX_CC)
                    cc_mask = cc < count
                    child_idx = tl.load(
                        ChildOrder_ptr
                        + (ANCHOR_S * BASE_HEADS + kvh) * N
                        + off_start
                        + cc,
                        mask=cc_mask,
                        other=0,
                    )

                    survive = cc_mask
                    for s in range(NUM_SUBSPACES):
                        if s != ANCHOR_S:
                            assign = tl.load(
                                Assigns_ptr
                                + (s * BASE_HEADS + kvh) * N
                                + child_idx,
                                mask=survive,
                                other=0,
                            )
                            passed = tl.load(
                                ClusterPass_ptr
                                + (s * H_Q + hq) * K
                                + assign,
                                mask=survive,
                                other=0,
                            )
                            survive = survive & (passed != 0)

                    if tl.max(survive.to(tl.int32), axis=0) != 0:
                        acc = tl.zeros([MAX_CC], dtype=tl.float32)
                        for d in range(D):
                            qv = tl.load(Q_ptr + hq * D + d)
                            kv = tl.load(
                                KeysT_ptr + (kvh * D + d) * N + child_idx,
                                mask=survive,
                                other=0.0,
                            )
                            acc += qv * kv
                        tl.store(
                            Out_ptr + hq * N + child_idx, acc, mask=survive
                        )


def triton_anchor_cluster_batched_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    anchor_s: int = 0,
    max_cc: int = 32,
    clusters_per_prog: int = 8,
    num_warps: int = 4,
) -> torch.Tensor:
    if not HAS_TRITON or "child_order" not in layout:
        return dense_index_search(q, th_per_subspace, layout)

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout).to(torch.int8).contiguous()

    N = layout["num_points"]
    H_Q = q.shape[0]
    K = int(layout["centers"].shape[2])

    out = q.new_full((H_Q, N), float("-inf"))

    num_groups = (K + clusters_per_prog - 1) // clusters_per_prog
    grid = (H_Q, num_groups)
    _anchor_cluster_batched_kernel[grid](
        q.contiguous(),
        layout["keys_t"],
        layout["child_order"].contiguous(),
        layout["child_offsets"].contiguous(),
        layout["assigns_i32"],
        cluster_pass_flat,
        out,
        H_Q,
        layout["base_heads"],
        layout["groups"],
        N,
        K,
        K + 1,
        ANCHOR_S=anchor_s,
        D=q.shape[1],
        NUM_SUBSPACES=layout["num_subspaces"],
        MAX_CC=max_cc,
        CLUSTERS_PER_PROG=clusters_per_prog,
        num_warps=num_warps,
    )
    return out


if HAS_TRITON:
    @triton.jit
    def _fused_cluster_pass_kernel(
        QPacked_ptr,    # (S, H_q, MAX_D)        f32
        QNorm_ptr,      # (S, H_q)               f32
        Th_ptr,         # (S, H_q)               f32
        Centers_ptr,    # (S, H_kv, K, MAX_D)    f32
        Radii_ptr,      # (S, H_kv, K)           f32
        Out_ptr,        # (S, H_q, K)            i8
        H_Q,
        H_KV,
        K,
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

        hq_vec = kvh * GROUPS + g_range  # (GROUPS_POW,)

        # Load packed query for this (s, hq_vec): (GROUPS_POW, MAX_D)
        qp = tl.load(
            QPacked_ptr + (s * H_Q + hq_vec[:, None]) * MAX_D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )
        qn = tl.load(QNorm_ptr + s * H_Q + hq_vec, mask=g_valid, other=0.0)
        th = tl.load(Th_ptr + s * H_Q + hq_vec, mask=g_valid, other=float("inf"))

        # Load centers tile: (BLOCK_K, MAX_D) for this (s, kvh, k_range)
        c_offs = (s * H_KV + kvh) * K * MAX_D + k_range[:, None] * MAX_D + d_range[None, :]
        centers = tl.load(Centers_ptr + c_offs, mask=k_mask[:, None], other=0.0)

        # Radii: (BLOCK_K,)
        r = tl.load(Radii_ptr + (s * H_KV + kvh) * K + k_range, mask=k_mask, other=0.0)

        # Compute cluster_dot[g, k] = sum_d qp[g, d] * centers[k, d]
        # Use elementwise broadcast + sum.
        cdot = tl.sum(qp[:, None, :] * centers[None, :, :], axis=2)  # (GROUPS_POW, BLOCK_K)
        ub = cdot + r[None, :] * qn[:, None]  # (GROUPS_POW, BLOCK_K)
        passed = (ub >= th[:, None]).to(tl.int8)

        # Store (S, H_q, K)[s, hq_vec, k_range]
        out_offs = (s * H_Q + hq_vec[:, None]) * K + k_range[None, :]
        out_mask = g_valid[:, None] & k_mask[None, :]
        tl.store(Out_ptr + out_offs, passed, mask=out_mask)


def triton_fused_cluster_pass(
    q_packed: torch.Tensor,   # (S, H_q, max_d)
    q_norm: torch.Tensor,     # (S, H_q)
    th: torch.Tensor,         # (S, H_q)
    centers: torch.Tensor,    # (S, H_kv, K, max_d)
    radii: torch.Tensor,      # (S, H_kv, K)
    groups: int,
) -> torch.Tensor:
    """Returns int8 cluster_pass (S, H_q, K)."""
    S, H_q, max_d = q_packed.shape
    H_kv = centers.shape[1]
    K = centers.shape[2]
    out = torch.empty(S, H_q, K, device=q_packed.device, dtype=torch.int8)

    triton_fused_cluster_pass_out(
        q_packed=q_packed,
        q_norm=q_norm,
        th=th,
        centers=centers,
        radii=radii,
        groups=groups,
        out=out,
    )
    return out


def triton_fused_cluster_pass_out(
    q_packed: torch.Tensor,   # (S, H_q, max_d)
    q_norm: torch.Tensor,     # (S, H_q)
    th: torch.Tensor,         # (S, H_q)
    centers: torch.Tensor,    # (S, H_kv, K, max_d)
    radii: torch.Tensor,      # (S, H_kv, K)
    groups: int,
    out: torch.Tensor,        # (S, H_q, K)
) -> torch.Tensor:
    """Writes int8 cluster_pass into ``out`` and returns it."""
    S, H_q, max_d = q_packed.shape
    H_kv = centers.shape[1]
    K = centers.shape[2]

    groups_pow = 1
    while groups_pow < max(groups, 8):
        groups_pow *= 2
    block_k = 64

    grid = (S, H_kv, triton.cdiv(K, block_k))
    _fused_cluster_pass_kernel[grid](
        q_packed, q_norm, th, centers, radii, out,
        H_q, H_kv, K,
        S=S, GROUPS=groups, GROUPS_POW=groups_pow,
        MAX_D=max_d, BLOCK_K=block_k,
        num_warps=2,
    )
    return out


if HAS_TRITON:
    @triton.jit
    def _gqa_clusterpass_search_kernel(
        Q_ptr,
        KeysT_ptr,
        Assigns_ptr,
        ClusterPass_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        GROUPS: tl.constexpr,
        N,
        K,
        D: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        ns = n0 + tl.arange(0, BLOCK_N)
        nmask = ns < N

        # Load keys tile once for this kv head: (D, BLOCK_N)
        d_range = tl.arange(0, D)
        keys_tile = tl.load(
            KeysT_ptr + (kvh * D + d_range[:, None]) * N + ns[None, :],
            mask=nmask[None, :],
            other=0.0,
        )

        # Load per-subspace assigns for this kv head once: each (BLOCK_N,)
        # We reload per subspace below — keep simple.

        for g in range(GROUPS):
            hq = kvh * GROUPS + g

            # Compute survive mask for this hq
            survive = nmask
            for s in range(NUM_SUBSPACES):
                assign = tl.load(
                    Assigns_ptr + (s * BASE_HEADS + kvh) * N + ns,
                    mask=survive,
                    other=0,
                )
                passed = tl.load(
                    ClusterPass_ptr + (s * H_Q + hq) * K + assign,
                    mask=survive,
                    other=0,
                )
                survive = survive & (passed != 0)

            # Load query for this hq and compute dot with reused keys_tile
            q_vec = tl.load(Q_ptr + hq * D + d_range)
            # dot: sum over D of q_vec[d] * keys_tile[d, n]
            acc = tl.sum(q_vec[:, None] * keys_tile, axis=0)

            out_vals = tl.where(survive, acc, neg_inf)
            tl.store(Out_ptr + hq * N + ns, out_vals, mask=nmask)


if HAS_TRITON:
    @triton.jit
    def _gqa_tcore_kernel(
        Q_ptr,
        KeysT_ptr,
        Assigns_ptr,
        ClusterPass_ptr,
        Out_ptr,
        H_Q,
        BASE_HEADS,
        N,
        K,
        D: tl.constexpr,
        GROUPS: tl.constexpr,
        GROUPS_POW: tl.constexpr,
        NUM_SUBSPACES: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        neg_inf = float("-inf")
        kvh = tl.program_id(0)
        n0 = tl.program_id(1) * BLOCK_N
        ns = n0 + tl.arange(0, BLOCK_N)
        nmask = ns < N

        d_range = tl.arange(0, D)
        gp_range = tl.arange(0, GROUPS_POW)
        g_valid = gp_range < GROUPS

        # Load keys tile: (D, BLOCK_N). Shared across all groups of this kvh.
        keys_tile = tl.load(
            KeysT_ptr + (kvh * D + d_range[:, None]) * N + ns[None, :],
            mask=nmask[None, :],
            other=0.0,
        )

        # Load all group queries at once: (GROUPS_POW, D), mask invalid rows.
        hq_vec = kvh * GROUPS + gp_range
        q_block = tl.load(
            Q_ptr + hq_vec[:, None] * D + d_range[None, :],
            mask=g_valid[:, None],
            other=0.0,
        )

        # Tensor-core dot: (GROUPS_POW, D) x (D, BLOCK_N) -> (GROUPS_POW, BLOCK_N)
        acc = tl.dot(q_block, keys_tile, allow_tf32=True)

        # Vectorized gate across groups.
        survive2d = g_valid[:, None] & nmask[None, :]  # (GROUPS_POW, BLOCK_N)
        for s in range(NUM_SUBSPACES):
            # assigns are kv-level — single (BLOCK_N,) load shared by groups
            assign = tl.load(
                Assigns_ptr + (s * BASE_HEADS + kvh) * N + ns,
                mask=nmask,
                other=0,
            )
            passed = tl.load(
                ClusterPass_ptr
                + (s * H_Q + hq_vec[:, None]) * K
                + assign[None, :],
                mask=survive2d,
                other=0,
            )
            survive2d = survive2d & (passed != 0)

        out_vals = tl.where(survive2d, acc, neg_inf)
        tl.store(
            Out_ptr + hq_vec[:, None] * N + ns[None, :],
            out_vals,
            mask=g_valid[:, None] & nmask[None, :],
        )


def triton_gqa_tcore_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    block_n: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    if not HAS_TRITON or layout["mode"] != "grouped":
        return triton_clusterpass_search(q, th_per_subspace, layout, block_n, num_warps)

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout).to(torch.int8)
    out = torch.empty(
        q.shape[0],
        layout["num_points"],
        device=q.device,
        dtype=torch.float32,
    )
    N = layout["num_points"]
    groups = layout["groups"]
    groups_pow = 1
    while groups_pow < max(groups, 8):
        groups_pow *= 2
    grid = (layout["base_heads"], triton.cdiv(N, block_n))
    _gqa_tcore_kernel[grid](
        q.contiguous(),
        layout["keys_t"],
        layout["assigns_i32"],
        cluster_pass_flat,
        out,
        q.shape[0],
        layout["base_heads"],
        N,
        layout["centers"].shape[2],
        D=q.shape[1],
        GROUPS=groups,
        GROUPS_POW=groups_pow,
        NUM_SUBSPACES=layout["num_subspaces"],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def triton_gqa_clusterpass_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    block_n: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    if not HAS_TRITON or layout["mode"] != "grouped":
        return triton_clusterpass_search(q, th_per_subspace, layout, block_n, num_warps)

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout).to(torch.int8)
    out = torch.empty(
        q.shape[0],
        layout["num_points"],
        device=q.device,
        dtype=torch.float32,
    )
    N = layout["num_points"]
    grid = (layout["base_heads"], triton.cdiv(N, block_n))
    _gqa_clusterpass_search_kernel[grid](
        q.contiguous(),
        layout["keys_t"],
        layout["assigns_i32"],
        cluster_pass_flat,
        out,
        q.shape[0],
        layout["base_heads"],
        GROUPS=layout["groups"],
        N=N,
        K=layout["centers"].shape[2],
        D=q.shape[1],
        NUM_SUBSPACES=layout["num_subspaces"],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def triton_anchor_cluster_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    anchor_s: int = 0,
    max_cc: int = 32,
    num_warps: int = 2,
) -> torch.Tensor:
    """Anchor-subspace cluster-level search: skip entire clusters whose
    anchor-subspace gate fails. Requires child_order/child_offsets in layout.
    """
    if not HAS_TRITON or "child_order" not in layout:
        return dense_index_search(q, th_per_subspace, layout)

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout).to(torch.int8).contiguous()

    N = layout["num_points"]
    H_Q = q.shape[0]
    K = int(layout["centers"].shape[2])

    out = q.new_full((H_Q, N), float("-inf"))

    grid = (H_Q, K)
    _anchor_cluster_kernel[grid](
        q.contiguous(),
        layout["keys_t"],
        layout["child_order"].contiguous(),
        layout["child_offsets"].contiguous(),
        layout["assigns_i32"],
        cluster_pass_flat,
        out,
        H_Q,
        layout["base_heads"],
        layout["groups"],
        N,
        K,
        K + 1,
        ANCHOR_S=anchor_s,
        D=q.shape[1],
        NUM_SUBSPACES=layout["num_subspaces"],
        MAX_CC=max_cc,
        num_warps=num_warps,
    )
    return out


def triton_clusterpass_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    block_n: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    if not HAS_TRITON:
        return dense_index_search(q, th_per_subspace, layout)

    cluster_pass, _ = cluster_pass_only(q, th_per_subspace, layout)
    cluster_pass_flat = flatten_cluster_pass(cluster_pass, layout).to(torch.int8)
    out = torch.empty(
        q.shape[0],
        layout["num_points"],
        device=q.device,
        dtype=torch.float32,
    )
    grid = (q.shape[0], triton.cdiv(layout["num_points"], block_n))
    _clusterpass_search_kernel[grid](
        q.contiguous(),
        layout["keys_t"],
        layout["assigns_i32"],
        cluster_pass_flat,
        out,
        q.shape[0],
        layout["base_heads"],
        layout["groups"],
        layout["num_points"],
        layout["centers"].shape[2],
        D=q.shape[1],
        NUM_SUBSPACES=layout["num_subspaces"],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out


def triton_direct_search(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    layout: dict,
    block_n: int | None = None,
    num_warps: int = 4,
) -> torch.Tensor:
    if not HAS_TRITON:
        return dense_index_search(q, th_per_subspace, layout)

    _, q_packed, q_norm = pack_query_subspaces(q, layout)
    q_packed = q_packed.reshape(
        layout["num_subspaces"],
        q.shape[0],
        layout["max_d"],
    ).contiguous()
    q_norm = q_norm.reshape(layout["num_subspaces"], q.shape[0]).contiguous()
    out = torch.empty(
        q.shape[0],
        layout["num_points"],
        device=q.device,
        dtype=torch.float32,
    )
    if block_n is None:
        block_n = 32 if layout["num_subspaces"] >= 16 else 64
    grid = (q.shape[0], triton.cdiv(layout["num_points"], block_n))
    _direct_search_kernel[grid](
        q.contiguous(),
        q_packed,
        q_norm,
        layout["keys_t"],
        layout["centers"],
        layout["radii"],
        layout["widths"],
        layout["assigns_i32"],
        th_per_subspace.contiguous(),
        out,
        q.shape[0],
        layout["base_heads"],
        layout["groups"],
        layout["num_points"],
        layout["centers"].shape[2],
        D=q.shape[1],
        NUM_SUBSPACES=layout["num_subspaces"],
        MAX_D=layout["max_d"],
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out
