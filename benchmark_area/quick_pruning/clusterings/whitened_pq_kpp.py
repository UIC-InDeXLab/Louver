"""Whitened PQ with k-means++ initialization per subspace."""

from __future__ import annotations

import math

import torch


def cluster_whitened_pq_kpp(keys: torch.Tensor, bf: int, n_subspaces: int = 4, max_iter: int = 15):
    """
    Whitened PQ-subspace clustering with k-means++ initialization for
    each subspace's mini-k-means, and extra iterations.

    k-means++ gives better initial centers → fewer bad local minima →
    tighter sub-clusters → tighter AABB.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    mean = keys.mean(dim=1, keepdim=True)
    scale = keys.std(dim=1, keepdim=True).clamp_min(1e-4)
    norm_keys = (keys - mean) / scale

    sub_dim = D // n_subspaces
    remainder = D % n_subspaces
    sub_k = max(2, int(round(K ** (1.0 / n_subspaces))))

    sub_assigns = []
    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = norm_keys[:, :, offset : offset + sd].contiguous()
        offset += sd

        # k-means++ initialization
        sc = torch.empty(H, sub_k, sd, device=device, dtype=keys.dtype)
        idx0 = torch.randint(N, (H,), device=device)
        sc[:, 0, :] = sub_keys[torch.arange(H, device=device), idx0]

        min_dist_sq = torch.full((H, N), float("inf"), device=device)
        for j in range(1, sub_k):
            d = ((sub_keys - sc[:, j - 1 : j, :]) ** 2).sum(dim=-1)
            min_dist_sq = torch.minimum(min_dist_sq, d)
            probs = min_dist_sq / min_dist_sq.sum(dim=1, keepdim=True).clamp_min(1e-30)
            chosen = torch.multinomial(probs, 1).squeeze(-1)
            sc[:, j, :] = sub_keys[torch.arange(H, device=device), chosen]

        for _ in range(max_iter):
            dists = torch.cdist(sub_keys, sc)
            sa = dists.argmin(dim=2)
            new_sc = torch.zeros_like(sc)
            cnt = torch.zeros(H, sub_k, device=device, dtype=keys.dtype)
            new_sc.scatter_add_(1, sa.unsqueeze(-1).expand(-1, -1, sd), sub_keys)
            cnt.scatter_add_(1, sa, torch.ones(H, N, device=device, dtype=keys.dtype))
            mask = cnt > 0
            new_sc[mask] /= cnt[mask].unsqueeze(-1)
            new_sc[~mask] = sc[~mask]
            sc = new_sc

        sub_assigns.append(torch.cdist(sub_keys, sc).argmin(dim=2))

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k
    assign = composite % K

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device, dtype=keys.dtype))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    centers = centers / counts.clamp_min(1).unsqueeze(-1)
    norm_centers = (centers - mean) / scale
    assign = torch.cdist(norm_keys, norm_centers).argmin(dim=2)
    return assign, centers
