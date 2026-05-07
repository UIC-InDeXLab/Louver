"""Small union-of-balls enclosure."""

from __future__ import annotations

import torch


def enclose_multi_ball(keys, assign, centers, K, bf, num_anchors: int = 2):
    """
    Cover each cluster with a small union of balls and use the best anchor.

    This keeps the per-parent gate cheap while reducing the slack from a single
    worst-case radius. The final upper bound is clamped by the centroid ball so
    it is never weaker than the baseline ball enclosure.
    """
    H, N, D = keys.shape
    device = keys.device
    M = max(1, num_anchors)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent_for_child = centers.gather(1, idx_exp)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(
        1,
        assign,
        (keys - parent_for_child).norm(dim=-1),
        reduce="amax",
        include_self=True,
    )

    anchor_centers = torch.zeros(H, K, M, D, device=device, dtype=keys.dtype)
    anchor_radii = torch.zeros(H, K, M, device=device, dtype=keys.dtype)

    for h in range(H):
        for k in range(K):
            idx = (assign[h] == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            points = keys[h, idx]
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
        ball_upper = torch.einsum("hkd,hd->hk", centers, q) + ball_radii * q_norm

        anchor_scores = torch.einsum("hkmd,hd->hkm", anchor_centers, q)
        multi_upper = (anchor_scores + anchor_radii * q_norm.unsqueeze(1)).amax(dim=-1)

        return torch.minimum(ball_upper, multi_upper) > th.unsqueeze(-1)

    return gate, {
        "anchors": M,
        "ball_r_mean": float(ball_radii.mean()),
        "anchor_r_mean": float(anchor_radii.mean()),
    }
