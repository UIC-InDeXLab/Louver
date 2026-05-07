"""Nearest-neighbor greedy pairing: create the tightest possible small clusters.

For bf=2: pair each key with its nearest neighbor (greedy, non-overlapping).
For bf>2: greedily grow clusters from the tightest seed.

This minimizes within-cluster radius by construction, giving the tightest
possible ball and AABB enclosings.
"""

from __future__ import annotations

import math
import torch


def cluster_nn_greedy(keys: torch.Tensor, bf: int):
    """
    Greedy nearest-neighbor clustering.

    For bf=2: find mutual nearest neighbors, pair them. Remaining points
    paired greedily.

    For bf>2: pick the point with the smallest bf-NN radius, assign its
    bf nearest available neighbors, repeat.

    Fully batched across heads for bf=2 (efficient). Per-head loop for bf>2.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    if bf == 2:
        return _nn_pair_batched(keys, K)
    else:
        return _nn_greedy_general(keys, bf, K)


def _nn_pair_batched(keys: torch.Tensor, K: int):
    """Batched NN pairing for bf=2."""
    H, N, D = keys.shape
    device = keys.device

    # Pairwise distances
    dists = torch.cdist(keys, keys)  # (H, N, N)
    dists.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))  # exclude self

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        available = torch.ones(N, dtype=torch.bool, device=device)
        d = dists[h].clone()
        group_id = 0

        while available.sum() >= 2 and group_id < K:
            # Find the closest pair among available points
            d_masked = d.clone()
            d_masked[~available] = float("inf")
            d_masked[:, ~available] = float("inf")

            # Find global minimum
            flat_idx = d_masked.argmin()
            i = flat_idx // N
            j = flat_idx % N

            assign[h, i] = group_id
            assign[h, j] = group_id
            available[i] = False
            available[j] = False
            group_id += 1

        # Assign remaining singletons
        remaining = available.nonzero(as_tuple=True)[0]
        for idx in remaining:
            assign[h, idx] = min(group_id, K - 1)
            group_id += 1

    # Clamp to K-1
    assign = assign.clamp(0, K - 1)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)

    return assign, centers


def _nn_greedy_general(keys: torch.Tensor, bf: int, K: int):
    """General NN-greedy clustering for any bf."""
    H, N, D = keys.shape
    device = keys.device

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        keys_h = keys[h]
        dists = torch.cdist(keys_h.unsqueeze(0), keys_h.unsqueeze(0)).squeeze(0)

        # bf-NN radius per point
        nn_dists, _ = dists.topk(bf, dim=1, largest=False)
        ball_radii = nn_dists[:, -1]

        available = torch.ones(N, dtype=torch.bool, device=device)
        sorted_seeds = ball_radii.argsort()
        group_id = 0

        for seed in sorted_seeds:
            seed = seed.item()
            if not available[seed]:
                continue

            avail_idx = available.nonzero(as_tuple=True)[0]
            seed_dists = dists[seed, avail_idx]
            n_take = min(bf, len(avail_idx))
            _, top_local = seed_dists.topk(n_take, largest=False)
            chosen = avail_idx[top_local]

            assign[h, chosen] = group_id
            available[chosen] = False
            group_id += 1

            if not available.any():
                break

        # Assign any remaining
        remaining = available.nonzero(as_tuple=True)[0]
        for idx in remaining:
            assign[h, idx] = min(group_id, K - 1)
            group_id += 1

    assign = assign.clamp(0, K - 1)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)

    return assign, centers


def cluster_sort_partition(keys: torch.Tensor, bf: int):
    """
    Sort keys by projection onto their first PC, then partition into
    consecutive groups of size bf.

    Very fast (O(N log N) per head). Keys in each group are close along
    the highest-variance direction, giving tight AABBs along that axis.
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    # First PC per head
    mu = keys.mean(dim=1, keepdim=True)
    centered = keys - mu
    # Power iteration for top eigenvector (fast)
    v = torch.randn(H, D, 1, device=device, dtype=keys.dtype)
    for _ in range(10):
        v = torch.bmm(centered.transpose(1, 2), torch.bmm(centered, v))
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # Project onto first PC
    proj = torch.bmm(centered, v).squeeze(-1)  # (H, N)

    # Sort by projection and assign consecutive groups
    sorted_idx = proj.argsort(dim=1)
    assign = torch.zeros(H, N, dtype=torch.long, device=device)
    for h in range(H):
        for i, idx in enumerate(sorted_idx[h]):
            assign[h, idx] = min(i // bf, K - 1)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)

    return assign, centers
