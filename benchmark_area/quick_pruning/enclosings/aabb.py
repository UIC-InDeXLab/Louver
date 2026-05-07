"""Axis-aligned bounding box enclosure."""

from __future__ import annotations

import torch


def enclose_aabb(keys, assign, centers, K, bf):
    """
    Per-cluster axis-aligned bounding box (min/max per dimension).
    Gate computes the maximum possible dot product with the query by
    picking the best corner per dimension.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)  # (H, N, D)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    def gate(q, th):
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        max_dot = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H, K)
        return max_dot > th.unsqueeze(-1)

    vol = (hi - lo).clamp_min(0).prod(dim=-1)
    return gate, {"vol_mean": float(vol.mean()), "vol_max": float(vol.max())}
