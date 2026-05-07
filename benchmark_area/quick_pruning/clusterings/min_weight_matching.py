"""Minimum-weight perfect matching clustering (bf=2 only).

Finds n/2 disjoint pairs that (approximately) minimize the total
pairwise L2 distance, using scipy's linear_sum_assignment on the
full distance matrix.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_min_weight_matching(keys: torch.Tensor, bf: int):
    """
    Minimum-weight perfect matching for bf=2.

    Parameters
    ----------
    keys : (H, N, D) tensor
    bf   : branching factor, must be 2

    Returns
    -------
    assign  : (H, N) long tensor – cluster id for each key
    centers : (H, K, D) tensor – cluster centroids
    """
    assert bf == 2, f"min_weight_matching only supports bf=2, got bf={bf}"

    H, N, D = keys.shape
    device = keys.device
    K = math.ceil(N / 2)

    dists = torch.cdist(keys, keys)  # (H, N, N)

    assign = torch.empty(H, N, dtype=torch.long, device=device)

    for h in range(H):
        d = dists[h].cpu().numpy().astype(np.float64)
        np.fill_diagonal(d, 1e18)

        row_ind, col_ind = linear_sum_assignment(d)

        # Extract pairs from the permutation
        visited = np.zeros(N, dtype=bool)
        group_id = 0
        a_h = np.empty(N, dtype=np.int64)

        for i in range(N):
            if visited[i]:
                continue
            j = col_ind[i]
            a_h[i] = group_id
            visited[i] = True
            if not visited[j]:
                a_h[j] = group_id
                visited[j] = True
            group_id += 1

        assign[h] = torch.from_numpy(a_h).to(device)

    assign = assign.clamp(0, K - 1)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    centers /= counts.clamp_min(1).unsqueeze(-1)

    return assign, centers
