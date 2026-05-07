"""K-center clustering via farthest-point insertion."""

from __future__ import annotations

import math

import torch


def cluster_kcenter(keys, bf, refine_iter=5):
    """
    K-center clustering: greedy farthest-point seeding followed by
    k-means refinement.

    Phase 1 (farthest-point): Selects centers that minimize the maximum
    distance from any point to its nearest center. This is a
    2-approximation of the NP-hard k-center problem and directly
    optimizes for tight ball enclosures.

    Phase 2 (refinement): A few Lloyd's iterations to polish centroids,
    keeping the good initialization from phase 1.

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) cluster centroids.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    # ── Phase 1: farthest-point seeding ──
    center_indices = torch.zeros(H, K, dtype=torch.long, device=device)
    center_indices[:, 0] = torch.randint(0, N, (H,), device=device)

    first = keys.gather(
        1, center_indices[:, :1].unsqueeze(-1).expand(-1, 1, D)
    )  # (H, 1, D)
    min_dist = (keys - first).norm(dim=-1)  # (H, N)

    for i in range(1, K):
        farthest = min_dist.argmax(dim=1)  # (H,)
        center_indices[:, i] = farthest
        new_center = keys.gather(
            1, farthest.view(H, 1, 1).expand(-1, 1, D)
        )  # (H, 1, D)
        new_dist = (keys - new_center).norm(dim=-1)  # (H, N)
        min_dist = torch.min(min_dist, new_dist)

    centers = keys.gather(
        1, center_indices.unsqueeze(-1).expand(-1, -1, D)
    )  # (H, K, D)

    # ── Phase 2: k-means refinement ──
    for _ in range(refine_iter):
        dists = torch.cdist(keys, centers)  # (H, N, K)
        assign = dists.argmin(dim=2)  # (H, N)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(
            1, assign.unsqueeze(-1).expand(-1, -1, D), keys
        )
        counts = torch.zeros(H, K, device=device)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        # Handle empty clusters
        empty_mask = counts == 0
        counts = counts.clamp_min(1)
        new_centers = new_centers / counts.unsqueeze(-1)

        if empty_mask.any():
            cur_dists = torch.cdist(keys, new_centers).min(dim=2).values
            for h in range(H):
                for k_idx in empty_mask[h].nonzero(as_tuple=True)[0]:
                    far = cur_dists[h].argmax()
                    new_centers[h, k_idx] = keys[h, far]
                    cur_dists[h, far] = 0.0

        centers = new_centers

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers
