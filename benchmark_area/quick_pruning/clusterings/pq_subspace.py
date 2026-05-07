"""Product-quantization-inspired subspace clustering."""

from __future__ import annotations

import math

import torch


def cluster_pq_subspace(keys: torch.Tensor, bf: int, n_subspaces: int = 4, max_iter: int = 10):
    """
    Split D dims into subspaces, cluster each independently with mini k-means,
    combine assignments via composite hash.

    Returns:
        assign: (H, N) cluster assignments.
        centers: (H, K, D) cluster centroids.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
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
        sc = sub_keys.gather(1, perm[:, :sub_k].unsqueeze(-1).expand(-1, -1, sd)).clone()

        for _ in range(max_iter):
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

    # Composite hash
    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k

    assign = composite % K

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

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers
