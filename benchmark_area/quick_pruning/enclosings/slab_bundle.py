"""Slab bundle: multiple random-direction halfspace bounds."""

from __future__ import annotations

import torch


def enclose_slab_bundle(keys, assign, centers, K, bf, n_dirs: int = 8):
    """
    Project all keys onto M random unit directions. For each cluster and
    each direction, store [min, max] projection.

    For query q and direction d_m, the contribution to q·k from the d_m
    component is (q·d_m)(k·d_m). Given k·d_m ∈ [a, b], this is bounded by
    max((q·d_m)*a, (q·d_m)*b).

    We build a "virtual AABB" in the random-direction coordinate system.
    If the M directions span the space well, this is like an oriented
    bounding box without per-cluster SVD.

    Intersected with standard ball bound.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict.
    """
    H, N, D = keys.shape
    device = keys.device
    M = n_dirs
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Random orthonormal directions (shared across heads) ──
    # Use QR decomposition for orthogonality
    rand_mat = torch.randn(D, M, device=device, dtype=keys.dtype)
    dirs, _ = torch.linalg.qr(rand_mat)  # (D, M) orthonormal columns

    # ── Project keys onto directions ──
    # keys: (H, N, D), dirs: (D, M) -> projs: (H, N, M)
    projs = keys @ dirs  # (H, N, M)

    # ── Per-cluster min/max projections ──
    idx_exp_m = assign.unsqueeze(-1).expand(-1, -1, M)  # (H, N, M)
    lo = torch.full((H, K, M), float("inf"), device=device)
    hi = torch.full((H, K, M), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp_m, projs, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp_m, projs, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # ── Ball radii for intersection ──
    parent_for_child = centers.gather(1, idx_exp)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(
        1, assign, (keys - parent_for_child).norm(dim=-1),
        reduce="amax", include_self=True,
    )

    # ── Residual energy: directions only span M out of D dims ──
    # Project centers onto directions
    center_projs = centers @ dirs  # (H, K, M)

    # Residual norm: total key energy not captured by the M projections
    # For each cluster, bound the residual as a ball
    key_in_dirs = projs @ dirs.T  # (H, N, D) — reconstruction in M-subspace
    residual_norms = (keys - key_in_dirs).norm(dim=-1)  # (H, N)
    max_residual = torch.full((H, K), 0.0, device=device)
    max_residual.scatter_reduce_(1, assign, residual_norms, reduce="amax", include_self=True)

    def gate(q, th):
        th_exp = th.unsqueeze(-1)

        # Project query onto M directions
        q_proj = (q @ dirs).unsqueeze(1)  # (H, 1, M)

        # AABB in projected space: sum_m max(q_proj_m * lo_m, q_proj_m * hi_m)
        slab_score = torch.maximum(q_proj * lo, q_proj * hi).sum(dim=-1)  # (H, K)

        # Add residual bound: ||q_perp|| * max_residual_norm
        q_in_dirs = (q @ dirs) @ dirs.T  # (H, D)
        q_perp_norm = (q - q_in_dirs).norm(dim=-1, keepdim=True)  # (H, 1)
        slab_upper = slab_score + q_perp_norm * max_residual  # (H, K)

        slab_pass = slab_upper > th_exp

        # Ball intersection
        ball_scores = torch.einsum("hkd,hd->hk", centers, q)
        ball_pass = (ball_scores + ball_radii) > th_exp

        return slab_pass & ball_pass

    return gate, {
        "n_dirs": M,
        "max_residual_mean": float(max_residual.mean()),
        "ball_r_mean": float(ball_radii.mean()),
    }
