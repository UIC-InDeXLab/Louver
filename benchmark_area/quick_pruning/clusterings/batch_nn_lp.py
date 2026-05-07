"""Lp-aware batch nearest-neighbor grouping."""

from __future__ import annotations

import math

import torch

from ._lp_utils import cdist_lp, recompute_centers_lp, validate_p


def cluster_batch_nn_lp(keys: torch.Tensor, bf: int, p: float = 2.0):
    """
    Greedy batch nearest-neighbor grouping under Lp distance.

    For each point, form a candidate bf-neighborhood under the requested
    metric, score it by the bf-th neighbor distance, then greedily accept
    the tightest non-overlapping groups first.
    """
    p = validate_p(p)
    h, n, d = keys.shape
    device = keys.device
    k = max(1, math.ceil(n / bf))

    assign = torch.full((h, n), -1, dtype=torch.long, device=device)

    for head in range(h):
        points = keys[head]
        dists = cdist_lp(points.unsqueeze(0), points.unsqueeze(0), p).squeeze(0)
        nn_dists, _ = dists.topk(bf, dim=1, largest=False)
        candidate_radii = nn_dists[:, -1]
        available = torch.ones(n, dtype=torch.bool, device=device)
        group_id = 0

        for seed in candidate_radii.argsort():
            seed = seed.item()
            if not available[seed]:
                continue

            avail_idx = available.nonzero(as_tuple=True)[0]
            seed_dists = dists[seed, avail_idx]
            n_take = min(bf, avail_idx.numel())
            _, top_local = seed_dists.topk(n_take, largest=False)
            chosen = avail_idx[top_local]

            assign[head, chosen] = group_id
            available[chosen] = False
            group_id += 1

            if not available.any():
                break

    for head in range(h):
        _, inverse = assign[head].unique(sorted=True, return_inverse=True)
        assign[head] = inverse.clamp(max=k - 1)

    centers = recompute_centers_lp(keys, assign, k, p)
    return assign, centers


def make_cluster_batch_nn_lp(p: float):
    """Factory that binds a concrete p value into the clustering callable."""
    p = validate_p(p)

    def _cluster_batch_nn_lp(keys: torch.Tensor, bf: int):
        return cluster_batch_nn_lp(keys, bf, p=p)

    return _cluster_batch_nn_lp
