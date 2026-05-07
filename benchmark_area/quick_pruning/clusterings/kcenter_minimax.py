"""K-center refinement that keeps the minimax-ball objective."""

from __future__ import annotations

import math

import torch


def _kcenter_init(points: torch.Tensor, k: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.empty(k, d, device=points.device, dtype=points.dtype)

    idx0 = torch.randint(n, (1,), device=points.device)
    centers[0] = points[idx0]
    min_dist = (points - centers[:1]).norm(dim=-1)

    for j in range(1, k):
        farthest = min_dist.argmax()
        centers[j] = points[farthest]
        dist = (points - centers[j : j + 1]).norm(dim=-1)
        min_dist = torch.minimum(min_dist, dist)

    return centers


def _approx_meb_center(points: torch.Tensor, n_iter: int) -> torch.Tensor:
    n, d = points.shape
    if n == 1:
        return points[0]

    center = points.mean(dim=0)
    for t in range(1, n_iter + 1):
        dists = (points - center).norm(dim=-1)
        farthest = points[dists.argmax()]
        center = center + (farthest - center) / (t + 1)
    return center


def _recompute_meb_centers(points: torch.Tensor, assign: torch.Tensor, k: int, n_iter: int) -> torch.Tensor:
    n, d = points.shape
    centers = torch.empty(k, d, device=points.device, dtype=points.dtype)

    for cid in range(k):
        idx = (assign == cid).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            centers[cid] = points[torch.randint(n, (1,), device=points.device)]
            continue
        centers[cid] = _approx_meb_center(points[idx], n_iter=n_iter)

    return centers


def cluster_kcenter_minimax(keys: torch.Tensor, bf: int, refine_iter: int = 6, meb_iter: int = 10):
    """
    Farthest-point k-center followed by repeated 1-center style recentering.

    Unlike plain ``kcenter``, this does not switch to mean-based Lloyd updates.
    It keeps optimizing a max-radius proxy, which is the right objective for
    downstream ball enclosures.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    assign = torch.empty(h, n, dtype=torch.long, device=keys.device)
    centers = torch.empty(h, k, d, dtype=keys.dtype, device=keys.device)

    for head in range(h):
        points = keys[head]
        centers_h = _kcenter_init(points, k)
        assign_h = torch.cdist(points.unsqueeze(0), centers_h.unsqueeze(0)).squeeze(0).argmin(dim=1)

        for _ in range(refine_iter):
            next_centers = _recompute_meb_centers(points, assign_h, k, n_iter=meb_iter)
            next_assign = torch.cdist(points.unsqueeze(0), next_centers.unsqueeze(0)).squeeze(0).argmin(dim=1)
            centers_h = next_centers
            if torch.equal(next_assign, assign_h):
                break
            assign_h = next_assign

        assign[head] = assign_h
        centers[head] = centers_h

    return assign, centers
