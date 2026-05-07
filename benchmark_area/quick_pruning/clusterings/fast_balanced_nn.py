"""Fast balanced nearest-neighbor pairing.

Uses mutual nearest neighbors + greedy refinement.
Much faster than the full nn_greedy while maintaining near-optimal pair quality.

Core algorithm for bf=2:
1. Compute all pairwise distances (fast GPU matmul via cdist)
2. Find mutual nearest neighbors (vectorized)
3. Match mutual NNs in one shot
4. For remaining points, iteratively find next-best mutual NNs
5. Pair remaining stragglers greedily

For bf>2: use a balanced partition approach with greedy seed expansion.
"""

from __future__ import annotations

import math
import torch


def cluster_fast_balanced_nn(keys: torch.Tensor, bf: int):
    """Fast balanced NN clustering. Optimized for bf=2 but works for any bf."""
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    if bf == 2:
        return _fast_pair_bf2(keys, K)
    else:
        return _fast_balanced_general(keys, bf, K)


def _fast_pair_bf2(keys: torch.Tensor, K: int):
    """
    Fast NN pairing for bf=2 using iterative mutual-NN matching.

    O(N² D) for cdist + O(N * rounds) for matching, where rounds ≈ 3-5.
    The cdist dominates and is fully GPU-parallelized.
    """
    H, N, D = keys.shape
    device = keys.device

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        dists = torch.cdist(keys[h:h+1], keys[h:h+1]).squeeze(0)  # (N, N)
        dists.diagonal().fill_(float("inf"))

        available = torch.ones(N, dtype=torch.bool, device=device)
        group_id = 0
        max_rounds = 20

        for _ in range(max_rounds):
            avail_idx = available.nonzero(as_tuple=True)[0]
            n_avail = len(avail_idx)
            if n_avail < 2:
                break

            # Sub-distance matrix for available points
            sub_dists = dists[avail_idx][:, avail_idx]  # (n_avail, n_avail)

            # Find nearest neighbor for each available point
            nn_local = sub_dists.argmin(dim=1)  # (n_avail,)

            # Find mutual nearest neighbors
            mutual = (nn_local[nn_local] == torch.arange(n_avail, device=device))

            # Among mutual NNs, ensure we only count each pair once (i < j)
            mutual_pairs_i = mutual.nonzero(as_tuple=True)[0]
            mutual_pairs_j = nn_local[mutual_pairs_i]
            # Keep only i < j to avoid double-counting
            valid = mutual_pairs_i < mutual_pairs_j
            pair_i = mutual_pairs_i[valid]
            pair_j = mutual_pairs_j[valid]

            if len(pair_i) == 0:
                # No mutual NNs found — force-pair the closest available pair
                flat_min = sub_dists.argmin()
                i_local = flat_min // n_avail
                j_local = flat_min % n_avail
                i_global = avail_idx[i_local]
                j_global = avail_idx[j_local]
                assign[h, i_global] = group_id
                assign[h, j_global] = group_id
                available[i_global] = False
                available[j_global] = False
                group_id += 1
                continue

            # Assign all mutual NN pairs at once
            for k in range(len(pair_i)):
                i_global = avail_idx[pair_i[k]]
                j_global = avail_idx[pair_j[k]]
                assign[h, i_global] = group_id
                assign[h, j_global] = group_id
                available[i_global] = False
                available[j_global] = False
                group_id += 1

        # Handle remaining singletons
        remaining = available.nonzero(as_tuple=True)[0]
        for idx in remaining:
            assign[h, idx] = min(group_id, K - 1)
            group_id += 1

    assign = assign.clamp(0, K - 1)
    centers = _compute_centers(keys, assign, K)
    return assign, centers


def _fast_balanced_general(keys: torch.Tensor, bf: int, K: int):
    """
    Balanced NN clustering for bf > 2.

    Strategy:
    1. Sort pairwise distances globally (flattened)
    2. Greedily build clusters: for each unassigned point, find its nearest
       unassigned neighbor and add to the same cluster (up to bf points).

    This is a simplified version that produces balanced clusters.
    """
    H, N, D = keys.shape
    device = keys.device

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        dists = torch.cdist(keys[h:h+1], keys[h:h+1]).squeeze(0)
        dists.diagonal().fill_(float("inf"))

        # For each point, find its nearest neighbor distance
        nn_dists = dists.min(dim=1).values  # (N,)

        # Process points in order of increasing NN distance (tightest cores first)
        order = nn_dists.argsort()
        available = torch.ones(N, dtype=torch.bool, device=device)
        cluster_count = torch.zeros(K, dtype=torch.long, device=device)
        group_id = 0

        for seed_idx in order:
            seed = seed_idx.item()
            if not available[seed]:
                continue
            if group_id >= K:
                break

            # Find bf-1 nearest available neighbors
            avail_idx = available.nonzero(as_tuple=True)[0]
            seed_dists = dists[seed, avail_idx]
            n_take = min(bf, len(avail_idx))
            _, top_local = seed_dists.topk(n_take, largest=False)
            chosen = avail_idx[top_local]

            for idx in chosen:
                assign[h, idx] = group_id
                available[idx] = False

            group_id += 1

        # Assign remaining
        remaining = available.nonzero(as_tuple=True)[0]
        if len(remaining) > 0:
            for idx in remaining:
                assign[h, idx] = min(group_id, K - 1)
                group_id += 1

    assign = assign.clamp(0, K - 1)
    centers = _compute_centers(keys, assign, K)
    return assign, centers


def _compute_centers(keys, assign, K):
    H, N, D = keys.shape
    device = keys.device
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)
    return centers
