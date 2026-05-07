"""L_inf-distance nearest-neighbor pairing for bf=2.

For bf=2, AABB span = |a - b| element-wise. The AABB upper bound looseness
depends on max(|q_d| * |a_d - b_d|). Since |q_d| varies, the optimal
distance metric depends on q. However, L_inf minimizes the maximum
dimension span, which is a good proxy for AABB tightness.

Also implements "weighted L1" pairing where dimensions are weighted
by their importance for AABB pruning (estimated from key statistics).
"""

from __future__ import annotations

import math
import torch


def cluster_linf_nn(keys: torch.Tensor, bf: int):
    """L_inf-distance NN pairing. Only for bf=2."""
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    if bf != 2:
        raise ValueError("linf_nn_pairing is only for bf=2")

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        # L_inf distance matrix
        dists = torch.cdist(keys[h:h+1], keys[h:h+1], p=float("inf")).squeeze(0)
        dists.diagonal().fill_(float("inf"))

        available = torch.ones(N, dtype=torch.bool, device=device)
        group_id = 0

        for _ in range(20):
            avail_idx = available.nonzero(as_tuple=True)[0]
            n_avail = len(avail_idx)
            if n_avail < 2:
                break

            sub_dists = dists[avail_idx][:, avail_idx]
            nn_local = sub_dists.argmin(dim=1)
            mutual = nn_local[nn_local] == torch.arange(n_avail, device=device)

            mutual_i = mutual.nonzero(as_tuple=True)[0]
            mutual_j = nn_local[mutual_i]
            valid = mutual_i < mutual_j
            pair_i = mutual_i[valid]
            pair_j = mutual_j[valid]

            if len(pair_i) == 0:
                flat_min = sub_dists.argmin()
                i_local = flat_min // n_avail
                j_local = flat_min % n_avail
                assign[h, avail_idx[i_local]] = group_id
                assign[h, avail_idx[j_local]] = group_id
                available[avail_idx[i_local]] = False
                available[avail_idx[j_local]] = False
                group_id += 1
                continue

            for k in range(len(pair_i)):
                assign[h, avail_idx[pair_i[k]]] = group_id
                assign[h, avail_idx[pair_j[k]]] = group_id
                available[avail_idx[pair_i[k]]] = False
                available[avail_idx[pair_j[k]]] = False
                group_id += 1

        remaining = available.nonzero(as_tuple=True)[0]
        for idx in remaining:
            assign[h, idx] = min(group_id, K - 1)
            group_id += 1

    assign = assign.clamp(0, K - 1)
    centers = _compute_centers(keys, assign, K)
    return assign, centers


def cluster_weighted_l1_nn(keys: torch.Tensor, bf: int):
    """Weighted L1-distance NN pairing.

    Weights each dimension by 1/std(key_d) to equalize contribution.
    Dimensions with high variance get lower weight (harder to make tight),
    dimensions with low variance get higher weight (easy to make tight,
    so pair to minimize their span).
    """
    H, N, D = keys.shape
    device = keys.device
    K = max(1, math.ceil(N / bf))

    if bf != 2:
        raise ValueError("weighted_l1_nn_pairing is only for bf=2")

    assign = torch.full((H, N), -1, dtype=torch.long, device=device)

    for h in range(H):
        # Compute per-dimension weights: 1/std
        std = keys[h].std(dim=0).clamp_min(1e-6)  # (D,)
        weights = 1.0 / std  # (D,)
        weights = weights / weights.sum() * D  # normalize so sum = D

        # Weighted L1 distance
        weighted_keys = keys[h] * weights.unsqueeze(0)  # (N, D)
        dists = torch.cdist(weighted_keys.unsqueeze(0), weighted_keys.unsqueeze(0), p=1.0).squeeze(0)
        dists.diagonal().fill_(float("inf"))

        available = torch.ones(N, dtype=torch.bool, device=device)
        group_id = 0

        for _ in range(20):
            avail_idx = available.nonzero(as_tuple=True)[0]
            n_avail = len(avail_idx)
            if n_avail < 2:
                break

            sub_dists = dists[avail_idx][:, avail_idx]
            nn_local = sub_dists.argmin(dim=1)
            mutual = nn_local[nn_local] == torch.arange(n_avail, device=device)

            mutual_i = mutual.nonzero(as_tuple=True)[0]
            mutual_j = nn_local[mutual_i]
            valid = mutual_i < mutual_j
            pair_i = mutual_i[valid]
            pair_j = mutual_j[valid]

            if len(pair_i) == 0:
                flat_min = sub_dists.argmin()
                i_local = flat_min // n_avail
                j_local = flat_min % n_avail
                assign[h, avail_idx[i_local]] = group_id
                assign[h, avail_idx[j_local]] = group_id
                available[avail_idx[i_local]] = False
                available[avail_idx[j_local]] = False
                group_id += 1
                continue

            for k in range(len(pair_i)):
                assign[h, avail_idx[pair_i[k]]] = group_id
                assign[h, avail_idx[pair_j[k]]] = group_id
                available[avail_idx[pair_i[k]]] = False
                available[avail_idx[pair_j[k]]] = False
                group_id += 1

        remaining = available.nonzero(as_tuple=True)[0]
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
