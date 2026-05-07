"""Norm-shell directional clustering."""

from __future__ import annotations

import math

import torch


def cluster_shell_kmeans(
    keys: torch.Tensor,
    bf: int,
    n_shells: int | None = None,
    max_iter: int = 8,
):
    """
    Partition each head into norm shells, then cluster directions inside each
    shell. This directly tightens angular and norm spread at the same time.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    key_norms = keys.norm(dim=-1)
    dir_keys = keys / key_norms.unsqueeze(-1).clamp_min(1e-12)

    shell_count = n_shells
    if shell_count is None:
        shell_count = min(8, max(2, int(round(math.sqrt(K)))))
    shell_count = min(shell_count, K, N)

    assign = torch.zeros(H, N, device=device, dtype=torch.long)

    for h in range(H):
        sorted_idx = key_norms[h].argsort()
        shell_slices = _equal_slices(N, shell_count)
        shell_sizes = [shell_slice.stop - shell_slice.start for shell_slice in shell_slices]
        shell_clusters = _allocate_clusters(shell_sizes, K)

        next_cluster = 0
        for shell_slice, local_k in zip(shell_slices, shell_clusters):
            shell_idx = sorted_idx[shell_slice]
            if shell_idx.numel() == 0:
                continue

            if local_k <= 1 or shell_idx.numel() <= 1:
                assign[h, shell_idx] = next_cluster
                next_cluster += 1
                continue

            shell_dirs = dir_keys[h, shell_idx]
            local_assign = _single_head_kmeans(shell_dirs, local_k, max_iter=max_iter)
            assign[h, shell_idx] = local_assign + next_cluster
            next_cluster += local_k

    centers = _centers_from_assign(keys, assign, K)
    return assign, centers


def _equal_slices(n: int, parts: int) -> list[slice]:
    edges = torch.linspace(0, n, parts + 1, dtype=torch.int64)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(parts)]


def _allocate_clusters(shell_sizes: list[int], K: int) -> list[int]:
    total = max(1, sum(shell_sizes))
    raw = [max(1, size) / total * K for size in shell_sizes]
    base = [max(1, int(math.floor(x))) for x in raw]
    current = sum(base)

    if current > K:
        order = sorted(range(len(base)), key=lambda i: (base[i], raw[i] - base[i]), reverse=True)
        for idx in order:
            if current <= K:
                break
            if base[idx] > 1:
                base[idx] -= 1
                current -= 1
    elif current < K:
        order = sorted(range(len(base)), key=lambda i: raw[i] - base[i], reverse=True)
        ptr = 0
        while current < K:
            base[order[ptr % len(order)]] += 1
            current += 1
            ptr += 1

    return base


def _single_head_kmeans(points: torch.Tensor, K: int, max_iter: int) -> torch.Tensor:
    n, d = points.shape
    device = points.device

    centers = torch.empty(K, d, device=device, dtype=points.dtype)
    first = torch.randint(n, (1,), device=device).item()
    centers[0] = points[first]

    min_dist_sq = torch.full((n,), float("inf"), device=device, dtype=points.dtype)
    for j in range(1, K):
        dist_sq = (points - centers[j - 1 : j]).square().sum(dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        probs = min_dist_sq / min_dist_sq.sum().clamp_min(1e-30)
        chosen = torch.multinomial(probs, 1).item()
        centers[j] = points[chosen]

    for _ in range(max_iter):
        dist = torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)
        assign = dist.argmin(dim=1)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(K, device=device, dtype=points.dtype)
        new_centers.scatter_add_(0, assign.unsqueeze(-1).expand(-1, d), points)
        counts.scatter_add_(0, assign, torch.ones(n, device=device, dtype=points.dtype))

        empty = counts == 0
        if empty.any():
            refill = torch.randperm(n, device=device)[: int(empty.sum().item())]
            new_centers[empty] = points[refill]
            counts[empty] = 1

        centers = new_centers / counts.clamp_min(1).unsqueeze(-1)

    return torch.cdist(points.unsqueeze(0), centers.unsqueeze(0)).squeeze(0).argmin(dim=1)


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
