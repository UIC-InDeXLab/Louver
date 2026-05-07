"""Tight hybrid enclosure from complementary support bounds."""

from __future__ import annotations

import torch


def enclose_tight_hybrid(keys, assign, centers, K, bf, rank: int = 2, num_anchors: int = 2):
    """
    Combine the strongest complementary bounds:
      1. centroid ball for a safe baseline,
      2. axis-aligned box for per-dimension extremal structure,
      3. low-rank subspace box for rotated anisotropy,
      4. small multi-ball cover for local multimodality.

    The gate uses the minimum valid upper bound across all four.
    """
    H, N, D = keys.shape
    device = keys.device
    R = max(1, rank)
    M = max(1, num_anchors)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent_for_child = centers.gather(1, idx_exp)
    diff = keys - parent_for_child

    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(1, assign, diff.norm(dim=-1), reduce="amax", include_self=True)

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    basis = torch.zeros(H, K, R, D, device=device, dtype=keys.dtype)
    alpha = torch.zeros(H, K, R, device=device, dtype=keys.dtype)
    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)

    anchor_centers = torch.zeros(H, K, M, D, device=device, dtype=keys.dtype)
    anchor_radii = torch.zeros(H, K, M, device=device, dtype=keys.dtype)

    for h in range(H):
        for k in range(K):
            idx = (assign[h] == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            points = keys[h, idx]
            centered = points - centers[h, k]

            local_rank = min(R, max(0, int(points.shape[0]) - 1), D)
            if local_rank > 0 and float(centered.square().sum()) > 1e-12:
                try:
                    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                    axes = vh[:local_rank]
                except RuntimeError:
                    var = centered.square().mean(dim=0)
                    axes = torch.zeros(local_rank, D, device=device, dtype=keys.dtype)
                    axes[0, var.argmax()] = 1.0

                coeff = centered @ axes.transpose(0, 1)
                basis[h, k, :local_rank] = axes
                alpha[h, k, :local_rank] = coeff.abs().amax(dim=0)
                residual[h, k] = (centered - coeff @ axes).norm(dim=-1).amax()

            local_m = min(M, int(points.shape[0]))
            chosen = torch.empty(local_m, dtype=torch.long, device=device)
            mean = points.mean(dim=0, keepdim=True)
            chosen[0] = (points - mean).norm(dim=-1).argmax()
            min_sqdist = (points - points[chosen[0]]).square().sum(dim=-1)
            for m in range(1, local_m):
                chosen[m] = min_sqdist.argmax()
                cand_sqdist = (points - points[chosen[m]]).square().sum(dim=-1)
                min_sqdist = torch.minimum(min_sqdist, cand_sqdist)

            local_centers = points[chosen]
            dists = torch.cdist(points.unsqueeze(0), local_centers.unsqueeze(0)).squeeze(0)
            local_assign = dists.argmin(dim=-1)
            anchor_centers[h, k, :local_m] = local_centers
            for m in range(local_m):
                mask = local_assign == m
                if mask.any():
                    anchor_radii[h, k, m] = dists[mask, m].max()

    def gate(q, th):
        q_norm = q.norm(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)

        ball_upper = center_scores + ball_radii * q_norm

        q_exp = q.unsqueeze(1)
        aabb_upper = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)

        proj = torch.einsum("hkrd,hd->hkr", basis, q)
        box_slack = (proj.abs() * alpha).sum(dim=-1)
        perp_norm = (q_norm.square() - proj.square().sum(dim=-1)).clamp_min(0).sqrt()
        subspace_upper = center_scores + box_slack + residual * perp_norm

        anchor_scores = torch.einsum("hkmd,hd->hkm", anchor_centers, q)
        multi_upper = (anchor_scores + anchor_radii * q_norm.unsqueeze(1)).amax(dim=-1)

        upper = torch.minimum(ball_upper, aabb_upper)
        upper = torch.minimum(upper, subspace_upper)
        upper = torch.minimum(upper, multi_upper)
        return upper > th.unsqueeze(-1)

    return gate, {
        "rank": R,
        "anchors": M,
        "ball_r_mean": float(ball_radii.mean()),
        "residual_mean": float(residual.mean()),
        "anchor_r_mean": float(anchor_radii.mean()),
    }
