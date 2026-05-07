"""L1-distance batch-NN clustering for any bf.

Groups bf nearest points by L1 distance, which directly minimizes
the AABB span (since AABB span = element-wise |max - min| and the
total span sum = L1 diameter of the group).

For bf=2: equivalent to l1_nn_pairing
For bf=3+: greedy L1-NN grouping (pick tightest groups first)
"""

from __future__ import annotations

import math
import torch


def cluster_l1_batch_nn(keys: torch.Tensor, bf: int):
    """L1-distance batch-NN clustering."""
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        keys_h = keys[h]  # (N, D)

        # L1 distance matrix
        dists = torch.cdist(keys_h.unsqueeze(0), keys_h.unsqueeze(0), p=1.0).squeeze(0)

        # For each point, find bf nearest neighbors
        nn_dists, nn_idx = dists.topk(bf, dim=1, largest=False)  # (N, bf)

        # Score: max L1 distance in the group (tighter = better for AABB)
        ball_radii = nn_dists[:, -1]  # (N,)

        available = torch.ones(N, dtype=torch.bool, device=device)
        group_id = 0

        # Sort by tightness
        sorted_candidates = ball_radii.argsort()

        for seed in sorted_candidates:
            seed = seed.item()
            if not available[seed]:
                continue

            # Get bf nearest available neighbors by L1
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

    # Remap
    for h in range(H):
        _, inverse = assign[h].unique(sorted=True, return_inverse=True)
        assign[h] = inverse.clamp(max=K - 1)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)

    return assign, centers
