"""Shared helpers for balanced quick-pruning clusterings."""

from __future__ import annotations

import math

import torch


def target_cluster_sizes(n_points: int, bf: int, device: torch.device) -> torch.Tensor:
    """Return exact per-cluster capacities that sum to ``n_points``."""
    k = max(1, math.ceil(n_points / bf))
    base = n_points // k
    extra = n_points % k
    sizes = torch.full((k,), base, device=device, dtype=torch.long)
    if extra > 0:
        sizes[:extra] += 1
    return sizes


def recompute_centers(keys_h: torch.Tensor, assign_h: torch.Tensor, k: int) -> torch.Tensor:
    """Compute per-cluster means for one head."""
    n, d = keys_h.shape
    centers = torch.zeros(k, d, device=keys_h.device, dtype=keys_h.dtype)
    centers.scatter_add_(0, assign_h.unsqueeze(-1).expand(-1, d), keys_h)
    counts = torch.bincount(assign_h, minlength=k).to(keys_h.dtype).clamp_min(1)
    return centers / counts.unsqueeze(-1)


def dominant_axis(points: torch.Tensor, power_iters: int = 6) -> torch.Tensor:
    """Approximate the top PCA direction with short power iteration."""
    _, d = points.shape
    centered = points - points.mean(dim=0, keepdim=True)
    energy = float(centered.square().sum().item())
    if centered.shape[0] <= 1 or energy <= 1e-12:
        axis = torch.zeros(d, device=points.device, dtype=points.dtype)
        axis[0] = 1.0
        return axis

    v = centered[torch.randint(centered.shape[0], (1,), device=points.device)].transpose(0, 1)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(power_iters):
        proj = centered @ v
        v = centered.transpose(0, 1) @ proj
        v = v / v.norm().clamp_min(1e-12)
    return v.squeeze(-1)


def pairwise_sq_dists(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix for one head."""
    return (points[:, None, :] - centers[None, :, :]).square().sum(dim=-1)


def balanced_assign_from_cost(cost: torch.Tensor, target_sizes: torch.Tensor) -> torch.Tensor:
    """
    Greedy exact-capacity assignment.

    Points with the largest gap between first and second choices are assigned
    first, which usually reduces later forced placements.
    """
    n, k = cost.shape
    device = cost.device

    prefs = cost.argsort(dim=1)
    best = cost.gather(1, prefs[:, :1]).squeeze(1)
    if k > 1:
        second = cost.gather(1, prefs[:, 1:2]).squeeze(1)
        regret = second - best
    else:
        regret = torch.zeros_like(best)

    order = torch.argsort(regret, descending=True)
    assign = torch.full((n,), -1, dtype=torch.long, device=device)
    remaining = target_sizes.clone()

    for idx in order.tolist():
        ranked = prefs[idx]
        feasible = remaining[ranked] > 0
        chosen = ranked[feasible][0]
        assign[idx] = chosen
        remaining[chosen] -= 1

    if int(remaining.sum().item()) != 0:
        raise RuntimeError("Balanced assignment did not consume all cluster capacities.")
    return assign


def balanced_refine(
    keys_h: torch.Tensor,
    assign_h: torch.Tensor,
    target_sizes: torch.Tensor,
    n_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a few balanced Lloyd-style refinement passes for one head."""
    k = int(target_sizes.numel())
    centers = recompute_centers(keys_h, assign_h, k)
    for _ in range(n_iter):
        cost = pairwise_sq_dists(keys_h, centers)
        assign_next = balanced_assign_from_cost(cost, target_sizes)
        centers_next = recompute_centers(keys_h, assign_next, k)
        if torch.equal(assign_next, assign_h):
            assign_h = assign_next
            centers = centers_next
            break
        assign_h = assign_next
        centers = centers_next
    return assign_h, centers


def batched_centers(keys: torch.Tensor, assign: torch.Tensor, k: int) -> torch.Tensor:
    """Compute cluster means for all heads."""
    h, _, d = keys.shape
    centers = torch.zeros(h, k, d, device=keys.device, dtype=keys.dtype)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, d), keys)
    counts = torch.stack(
        [torch.bincount(assign_i, minlength=k) for assign_i in assign], dim=0
    ).to(keys.dtype)
    return centers / counts.clamp_min(1).unsqueeze(-1)
