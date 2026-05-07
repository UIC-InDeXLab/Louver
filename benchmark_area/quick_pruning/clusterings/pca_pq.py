"""Full PCA decorrelation + PQ subspace clustering."""

from __future__ import annotations

import math

import torch


def cluster_pca_pq(keys: torch.Tensor, bf: int, n_subspaces: int = 4, max_iter: int = 10):
    """
    Full PCA rotation to decorrelate ALL cross-dimension correlations,
    then PQ-subspace clustering in decorrelated space.

    Unlike whitened_pq (which only scales per-dim variance), PCA removes
    cross-dimension correlations entirely. After PCA, each subspace's
    dimensions are truly independent, giving the tightest possible
    per-subspace clusters and thus the tightest AABB bounds in original space.

    Returns:
        assign: (H, N) long.
        centers: (H, K, D).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # ── Full PCA per head ──
    key_mean = keys.mean(dim=1, keepdim=True)
    keys_c = keys - key_mean

    _, S, Vt = torch.linalg.svd(keys_c, full_matrices=False)
    # Rotate to PCA space and whiten (unit variance per PC)
    S_inv = 1.0 / S.clamp_min(1e-8)  # (H, min(N,D))
    keys_pca = torch.bmm(keys_c, Vt.transpose(-2, -1)) * S_inv.unsqueeze(1)

    # ── PQ-subspace in PCA-whitened space ──
    sub_dim = D // n_subspaces
    remainder = D % n_subspaces
    sub_k = max(2, int(round(K ** (1.0 / n_subspaces))))

    sub_assigns = []
    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = keys_pca[:, :, offset : offset + sd].contiguous()
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

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k
    assign = composite % K

    # ── Centers on original keys, reassign in PCA-whitened space ──
    centers_orig = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers_orig.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers_orig[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    mask = counts > 0
    centers_orig[mask] /= counts[mask].unsqueeze(-1)

    # Final reassignment in PCA-whitened space
    centers_pca = torch.bmm(centers_orig - key_mean, Vt.transpose(-2, -1)) * S_inv.unsqueeze(1)
    assign = torch.cdist(keys_pca, centers_pca).argmin(dim=2)

    # Recompute final centers
    centers_final = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers_final.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    counts = counts.clamp_min(1)
    centers_final /= counts.unsqueeze(-1)

    return assign, centers_final
