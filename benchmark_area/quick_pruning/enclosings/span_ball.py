"""Span-ball: Cauchy-Schwarz approximation of AABB with ball-level gate cost."""

from __future__ import annotations

import torch


def enclose_span_ball(keys, assign, centers, K, bf):
    """
    Uses the AABB half-span vector to define a tighter ball than ball_centroid.

    AABB score = q · mid + Σ_d |q_d| · half_span_d
    By Cauchy-Schwarz: Σ |q_d| · h_d ≤ ||q|| · ||h||₂

    For unit q: upper bound = q · mid + ||half_span||₂

    This is a ball centered at the midrange (not the mean!) with radius
    = L2 norm of half-spans.  Typically tighter than ball_centroid because:
    - midrange is the optimal center for minimax distance
    - ||half_span||₂ ≤ max-distance-from-center (unless all dims have equal span)

    Gate cost: 1 dot product + 1 add = g ≈ 1.0

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # Per-cluster min/max per dimension
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    mid = (lo + hi) / 2                       # (H, K, D) — midrange center
    half_span = (hi - lo) / 2                  # (H, K, D)
    radius = half_span.norm(dim=-1)            # (H, K) — ||half_span||₂

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", mid, q)  # (H, K)
        return (scores + radius) > th.unsqueeze(-1)

    return gate, {
        "radii_mean": float(radius.mean()),
        "radii_max": float(radius.max()),
    }
