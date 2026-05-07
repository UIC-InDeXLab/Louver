"""Bisect-AABB: split each cluster via 2-means, then AABB each half."""

from __future__ import annotations

import torch


def enclose_bisect_aabb(keys, assign, centers, K, bf):
    """
    For each cluster, run 2-means (3 iterations) to split into two
    sub-clusters, then build AABB per sub-cluster.

    Unlike split_aabb (which splits along the widest axis-aligned dim),
    this splits along the direction of maximum variance, giving more
    balanced and tighter sub-boxes.

    Intersected with ball bound for safety.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Ball radii (for intersection) ──
    parent_for_child = centers.gather(1, idx_exp)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(
        1, assign, (keys - parent_for_child).norm(dim=-1),
        reduce="amax", include_self=True,
    )

    # ── 2-means per cluster ──
    # Offset each key relative to its cluster centroid
    diff = keys - parent_for_child  # (H, N, D)

    # Pick two initial sub-centers: farthest pair heuristic
    # For efficiency, use the direction of max variance per cluster
    # Compute per-cluster covariance dominant direction via power iteration

    # Simple approach: split using the sign of projection onto
    # (farthest point from centroid) direction
    dist_from_center = diff.norm(dim=-1)  # (H, N)

    # For each cluster, find the farthest key
    big_neg = torch.full((H, K), -1.0, device=device)
    big_neg.scatter_reduce_(1, assign, dist_from_center, reduce="amax", include_self=True)
    farthest_dist = big_neg.gather(1, assign)  # (H, N) — max dist in each key's cluster

    is_farthest = (dist_from_center >= farthest_dist - 1e-8)  # approximate

    # Direction = centroid-to-farthest (per cluster)
    # Use scatter to accumulate farthest directions
    farthest_dir = torch.zeros(H, K, D, device=device)
    # Weight by is_farthest so only farthest points contribute
    weighted = diff * is_farthest.unsqueeze(-1).float()
    farthest_dir.scatter_add_(1, idx_exp, weighted)
    farthest_dir = farthest_dir / farthest_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Project each key onto its cluster's farthest direction
    dir_for_child = farthest_dir.gather(1, idx_exp)  # (H, N, D)
    proj = (diff * dir_for_child).sum(dim=-1)  # (H, N)

    # Split at projection = 0 (centroid)
    sub_assign = (proj >= 0).long()  # (H, N): 0 or 1

    # ── 2-means refinement: 3 iterations ──
    combined = assign * 2 + sub_assign  # (H, N)
    K2 = K * 2
    for _ in range(3):
        comb_exp = combined.unsqueeze(-1).expand(-1, -1, D)
        sub_centers = torch.zeros(H, K2, D, device=device, dtype=keys.dtype)
        sub_counts = torch.zeros(H, K2, device=device)
        sub_centers.scatter_add_(1, comb_exp, keys)
        sub_counts.scatter_add_(1, combined, torch.ones(H, N, device=device))
        sub_counts = sub_counts.clamp_min(1)
        sub_centers /= sub_counts.unsqueeze(-1)

        # Reassign within each cluster to nearest sub-center
        sc0 = sub_centers[:, 0::2, :]  # (H, K, D) — sub 0
        sc1 = sub_centers[:, 1::2, :]  # (H, K, D) — sub 1

        sc0_for_key = sc0.gather(1, idx_exp)
        sc1_for_key = sc1.gather(1, idx_exp)
        d0 = (keys - sc0_for_key).square().sum(dim=-1)
        d1 = (keys - sc1_for_key).square().sum(dim=-1)
        sub_assign = (d1 < d0).long()
        combined = assign * 2 + sub_assign

    # ── Build AABBs for each sub-cluster ──
    comb_exp = combined.unsqueeze(-1).expand(-1, -1, D)
    lo_sub = torch.full((H, K2, D), float("inf"), device=device)
    hi_sub = torch.full((H, K2, D), float("-inf"), device=device)
    lo_sub.scatter_reduce_(1, comb_exp, keys, reduce="amin", include_self=False)
    hi_sub.scatter_reduce_(1, comb_exp, keys, reduce="amax", include_self=False)
    empty_sub = lo_sub[:, :, 0].isinf()
    if empty_sub.any():
        lo_sub[empty_sub] = 0.0
        hi_sub[empty_sub] = 0.0

    lo_a = lo_sub[:, 0::2, :]
    hi_a = hi_sub[:, 0::2, :]
    lo_b = lo_sub[:, 1::2, :]
    hi_b = hi_sub[:, 1::2, :]

    def gate(q, th):
        th_exp = th.unsqueeze(-1)
        q_exp = q.unsqueeze(1)

        # Split AABB: OR over sub-boxes
        score_a = torch.maximum(q_exp * lo_a, q_exp * hi_a).sum(dim=-1)
        score_b = torch.maximum(q_exp * lo_b, q_exp * hi_b).sum(dim=-1)
        split_pass = (score_a > th_exp) | (score_b > th_exp)

        # Ball intersection
        ball_scores = torch.einsum("hkd,hd->hk", centers, q)
        ball_pass = (ball_scores + ball_radii) > th_exp

        return split_pass & ball_pass

    return gate, {"ball_r_mean": float(ball_radii.mean())}
