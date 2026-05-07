"""K-center assignments with minimum-enclosing-ball style centers."""

from __future__ import annotations

import math

import torch

from .kcenter import cluster_kcenter


def _approx_meb_center(points: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    n, d = points.shape
    if n == 0:
        raise ValueError('empty cluster')
    if n == 1:
        return points[0]

    center = points.mean(dim=0)
    for t in range(1, n_iter + 1):
        dists = (points - center).norm(dim=-1)
        farthest = points[dists.argmax()]
        center = center + (farthest - center) / (t + 1)
    return center


def cluster_kcenter_meb(keys: torch.Tensor, bf: int, refine_iter: int = 5, meb_iter: int = 10):
    """
    Use k-center for assignments, then replace each centroid with an approximate
    minimum-enclosing-ball center for that cluster.

    This targets the actual ball enclosure more directly than a plain mean.
    """
    assign, _ = cluster_kcenter(keys, bf, refine_iter=refine_iter)

    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    centers = torch.zeros(h, k, d, device=keys.device, dtype=keys.dtype)

    for head in range(h):
        for cid in range(k):
            idx = (assign[head] == cid).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                centers[head, cid] = keys[head, torch.randint(n, (1,), device=keys.device)]
                continue
            centers[head, cid] = _approx_meb_center(keys[head, idx], n_iter=meb_iter)

    return assign, centers
