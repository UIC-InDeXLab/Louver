"""PQ subspace init + L2 k-means refinement."""

from __future__ import annotations

import math

import torch


def cluster_pq_l2(keys: torch.Tensor, bf: int, pq_iter: int = 10, refine_iter: int = 10):
    """
    Two-phase:
    1. PQ-subspace (4 subspaces) for AABB-friendly initialization
    2. Standard L2 k-means refinement to polish

    The PQ init gives clusters with tight per-dimension spreads.
    L2 refinement moves points between clusters to reduce overall
    variance, which further tightens the AABB.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # ── Phase 1: PQ init ──
    n_subspaces = 4
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
        sc = sub_keys.gather(1, perm[:, :sub_k].unsqueeze(-1).expand(-1, -1, sd)).clone()

        for _ in range(pq_iter):
            dists = torch.cdist(sub_keys, sc)
            sa = dists.argmin(dim=2)
            new_sc = torch.zeros_like(sc)
            cnt = torch.zeros(H, sub_k, device=device)
            new_sc.scatter_add_(1, sa.unsqueeze(-1).expand(-1, -1, sd), sub_keys)
            cnt.scatter_add_(1, sa, torch.ones(H, N, device=device))
            m = cnt > 0
            new_sc[m] /= cnt[m].unsqueeze(-1)
            new_sc[~m] = sc[~m]
            sc = new_sc

        sub_assigns.append(torch.cdist(sub_keys, sc).argmin(dim=2))

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k

    assign = composite % K

    # Compute initial centers
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    # ── Phase 2: L2 k-means refinement ──
    for _ in range(refine_iter):
        dists = torch.cdist(keys, centers)
        assign = dists.argmin(dim=2)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, K, device=device)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        empty = counts == 0
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    new_centers[h, ek] = keys[h, far_idx]
                    counts[h, ek] = 1

        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        centers = new_centers

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers
