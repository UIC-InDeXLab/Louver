"""PQ initialization followed by span-aware refinement."""

from __future__ import annotations

import math

import torch


def cluster_pq_span(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int = 4,
    pq_iter: int = 8,
    refine_iter: int = 8,
):
    """
    Use PQ-subspace hashing for an AABB-friendly initialization, then refine
    with a span-extension objective.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    assign = _pq_init_assign(keys, K, n_subspaces=n_subspaces, max_iter=pq_iter)
    centers = _centers_from_assign(keys, assign, K)

    for _ in range(refine_iter):
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        lo = torch.full((H, K, D), float("inf"), device=device, dtype=keys.dtype)
        hi = torch.full((H, K, D), float("-inf"), device=device, dtype=keys.dtype)
        lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
        hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

        empty = lo[:, :, 0].isinf()
        if empty.any():
            lo[empty] = 0.0
            hi[empty] = 0.0

        over = (keys.unsqueeze(2) - hi.unsqueeze(1)).clamp_min(0)
        under = (lo.unsqueeze(1) - keys.unsqueeze(2)).clamp_min(0)
        span_cost = (over + under).sum(dim=-1)

        # Small centroid regularizer prevents drift when several boxes give
        # nearly identical span cost.
        center_pull = ((keys.unsqueeze(2) - centers.unsqueeze(1)) ** 2).sum(dim=-1).sqrt()
        assign = (span_cost + 0.05 * center_pull).argmin(dim=2)
        centers = _centers_from_assign(keys, assign, K)

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def _pq_init_assign(keys: torch.Tensor, K: int, n_subspaces: int, max_iter: int) -> torch.Tensor:
    H, N, D = keys.shape
    device = keys.device
    sub_dim = D // n_subspaces
    remainder = D % n_subspaces
    sub_k = max(2, int(round(K ** (1.0 / n_subspaces))))

    sub_assigns = []
    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = keys[:, :, offset : offset + sd].contiguous()
        offset += sd

        perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
        centers = sub_keys.gather(1, perm[:, :sub_k].unsqueeze(-1).expand(-1, -1, sd)).clone()

        for _ in range(max_iter):
            dists = torch.cdist(sub_keys, centers)
            sa = dists.argmin(dim=2)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(H, sub_k, device=device, dtype=keys.dtype)
            new_centers.scatter_add_(1, sa.unsqueeze(-1).expand(-1, -1, sd), sub_keys)
            counts.scatter_add_(1, sa, torch.ones(H, N, device=device, dtype=keys.dtype))
            mask = counts > 0
            new_centers[mask] /= counts[mask].unsqueeze(-1)
            new_centers[~mask] = centers[~mask]
            centers = new_centers

        sub_assigns.append(torch.cdist(sub_keys, centers).argmin(dim=2))

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k
    return composite % K


def _centers_from_assign(keys: torch.Tensor, assign: torch.Tensor, K: int) -> torch.Tensor:
    H, N, D = keys.shape
    device = keys.device
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))

    empty = counts == 0
    if empty.any():
        for h in range(H):
            empty_ids = empty[h].nonzero(as_tuple=False).flatten()
            if empty_ids.numel() == 0:
                continue
            refill = torch.randperm(N, device=device)[: empty_ids.numel()]
            centers[h, empty_ids] = keys[h, refill]
            counts[h, empty_ids] = 1

    return centers / counts.clamp_min(1).unsqueeze(-1)
