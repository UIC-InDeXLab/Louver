"""PQ subspace clustering followed by L∞ refinement."""

from __future__ import annotations

import math

import torch


def cluster_pq_linf(keys: torch.Tensor, bf: int, pq_iter: int = 10, refine_iter: int = 10):
    """
    Two-phase clustering:
    1. PQ-subspace (4 subspaces) to get a good AABB-aware initialization
    2. L∞ refinement to directly minimize max per-dimension span

    Combines PQ's subspace awareness with L∞'s direct AABB optimization.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # ── Phase 1: PQ-subspace initialization ──
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

    # Compute initial centers as midrange (L∞ optimal)
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                lo[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                hi[h, ek] = lo[h, ek]
    centers = (lo + hi) / 2

    # ── Phase 2: L∞ refinement ──
    for _ in range(refine_iter):
        diff = (keys.unsqueeze(2) - centers.unsqueeze(1)).abs()
        linf_dists = diff.amax(dim=-1)
        assign = linf_dists.argmin(dim=2)

        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        lo = torch.full((H, K, D), float("inf"), device=device)
        hi = torch.full((H, K, D), float("-inf"), device=device)
        lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
        hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

        empty = lo[:, :, 0].isinf()
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    lo[h, ek] = keys[h, far_idx]
                    hi[h, ek] = keys[h, far_idx]

        centers = (lo + hi) / 2

    assign = (keys.unsqueeze(2) - centers.unsqueeze(1)).abs().amax(dim=-1).argmin(dim=2)

    # Recompute centroids as means (needed for ball-based enclosings)
    mean_centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    mean_centers.scatter_add_(1, idx_exp, keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    counts = counts.clamp_min(1)
    mean_centers /= counts.unsqueeze(-1)

    return assign, mean_centers
