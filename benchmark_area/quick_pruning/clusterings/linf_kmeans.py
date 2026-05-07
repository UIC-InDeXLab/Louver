"""L-infinity k-means: optimizes for tight AABB enclosures."""

from __future__ import annotations

import math

import torch


def cluster_linf_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    K-means using Chebyshev (L∞) distance instead of L2.

    Assignment uses max absolute deviation per dimension from center,
    which directly minimizes the worst-case AABB span.

    Centers are computed as the midrange (avg of min/max per dim per cluster),
    which is the L∞ optimal center.

    Args:
        keys: (H, N, D).
        bf: branching factor.
        max_iter: iterations.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Initialize with k-means++ style seeding
    centers = torch.empty(H, K, D, device=device, dtype=keys.dtype)
    idx0 = torch.randint(N, (H,), device=device)
    centers[:, 0, :] = keys[torch.arange(H, device=device), idx0]

    min_dist = torch.full((H, N), float("inf"), device=device)
    for j in range(1, K):
        # L∞ distance to nearest center so far
        d = (keys - centers[:, j - 1 : j, :]).abs().amax(dim=-1)  # (H, N)
        min_dist = torch.minimum(min_dist, d)
        probs = min_dist / min_dist.sum(dim=1, keepdim=True).clamp_min(1e-30)
        chosen = torch.multinomial(probs, 1).squeeze(-1)
        centers[:, j, :] = keys[torch.arange(H, device=device), chosen]

    for _ in range(max_iter):
        # Assign by L∞ distance
        # (H, N, 1, D) - (H, 1, K, D) -> (H, N, K, D) -> max over D -> (H, N, K)
        diff = (keys.unsqueeze(2) - centers.unsqueeze(1)).abs()
        linf_dists = diff.amax(dim=-1)  # (H, N, K)
        assign = linf_dists.argmin(dim=2)

        # Update centers as midrange (optimal for L∞)
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        lo = torch.full((H, K, D), float("inf"), device=device)
        hi = torch.full((H, K, D), float("-inf"), device=device)
        lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
        hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

        empty = lo[:, :, 0].isinf()
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    lo[h, ek] = keys[h, far_idx]
                    hi[h, ek] = keys[h, far_idx]

        centers = (lo + hi) / 2

    assign = (keys.unsqueeze(2) - centers.unsqueeze(1)).abs().amax(dim=-1).argmin(dim=2)
    return assign, centers
