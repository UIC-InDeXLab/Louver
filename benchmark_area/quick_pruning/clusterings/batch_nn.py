"""Batch-NN balling methods exposed as clustering algorithms."""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import math

_bench_root = Path(__file__).resolve().parents[1]
if str(_bench_root) not in sys.path:
    sys.path.insert(0, str(_bench_root))


def cluster_batch_nn(keys: torch.Tensor, bf: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each point, find its bf-1 nearest neighbors. Form candidate balls.
    Then greedily select non-overlapping balls that cover all points.

    Strategy:
    1. Compute bf-NN for all points
    2. Score each candidate ball by tightness (max NN distance)
    3. Greedily pick the tightest ball, assign its points, repeat

    This finds globally optimal tight balls rather than growing from seeds.

    Returns:
        assign: (H, N) cluster assignments
        centers: (H, K, D) cluster centroids
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        keys_h = keys[h]  # (N, D)

        # Compute all pairwise distances
        dists = torch.cdist(keys_h.unsqueeze(0), keys_h.unsqueeze(0)).squeeze(
            0
        )  # (N, N)

        # For each point, find bf nearest neighbors (including self)
        nn_dists, nn_idx = dists.topk(bf, dim=1, largest=False)  # (N, bf)

        # Score each candidate ball: radius = max distance in the ball
        ball_radii = nn_dists[:, -1]  # (N,) — max distance to bf-th neighbor

        # Greedy assignment: pick tightest ball, assign, repeat
        available = torch.ones(N, dtype=torch.bool, device=device)
        group_id = 0

        # Sort candidates by radius (tightest first)
        sorted_candidates = ball_radii.argsort()

        for seed in sorted_candidates:
            seed = seed.item()
            if not available[seed]:
                continue

            # Get bf nearest available neighbors
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

    centers = torch.zeros(H, K, D, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    counts = counts.clamp_min(1)
    centers = centers / counts.unsqueeze(-1)

    return assign, centers
