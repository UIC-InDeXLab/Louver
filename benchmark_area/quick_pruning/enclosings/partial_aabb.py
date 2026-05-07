"""Partial AABB: exact intervals on the d widest AABB dimensions, Cauchy-Schwarz ball on the rest.

Tighter than span_ball, nearly as cheap (g ≈ 1.0 + d/(2D)).
Looser than full AABB but at ~half the gate cost.

For d dimensions with exact intervals:
  UB = q·mid + Σ_{i∈S} |q_i|·half_i + ||q_{S̄}||·||half_{S̄}||

vs span_ball: UB = q·mid + ||half||₂
vs full AABB: UB = q·mid + Σ_i |q_i|·half_i
"""

from __future__ import annotations

import torch


def _make_partial_aabb(d: int):
    def enclose(keys, assign, centers, K, bf):
        return enclose_partial_aabb(keys, assign, centers, K, bf, d=d)
    enclose.__name__ = f"enclose_partial_aabb_d{d}"
    return enclose


def enclose_partial_aabb(keys, assign, centers, K, bf, d: int = 8):
    H, N, D = keys.shape
    device = keys.device
    d = min(d, D)
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    lo = torch.full((H, K, D), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, D), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    mid = (lo + hi) / 2          # (H, K, D)
    half = (hi - lo) / 2         # (H, K, D)

    # Select top-d dimensions per (head, cluster) by half-span magnitude
    # Use global selection: top-d dims averaged across clusters per head
    half_importance = half.mean(dim=1)  # (H, D) — mean half-span per dim per head
    _, top_dims = half_importance.topk(d, dim=-1)  # (H, d)

    # Gather half-spans for selected dims per cluster
    top_dims_exp = top_dims.unsqueeze(1).expand(-1, K, -1)  # (H, K, d)
    half_sel = half.gather(2, top_dims_exp)  # (H, K, d) — half-spans for selected dims

    # Precompute ||half_{S̄}||₂ per cluster: residual half-span norm
    # Set selected dims to 0, then compute L2 norm of remaining
    half_residual = half.clone()
    half_residual.scatter_(2, top_dims_exp, 0.0)
    half_resid_norm = half_residual.norm(dim=-1)  # (H, K)

    def gate(q, th):
        # q·mid: standard centroid-like dot product (1 dp)
        base_score = torch.einsum("hkd,hd->hk", mid, q)  # (H, K)

        # |q| for selected dims
        q_abs_sel = q.abs().unsqueeze(1).expand(-1, K, -1).gather(2, top_dims_exp)  # (H, K, d)
        exact_part = (q_abs_sel * half_sel).sum(dim=-1)  # (H, K)

        # ||q_{S̄}||: query norm in non-selected dims
        q_sq = q.square()  # (H, D)
        q_sel_sq = q_sq.gather(1, top_dims).sum(dim=-1)  # (H,)
        q_resid_norm = (q_sq.sum(dim=-1) - q_sel_sq).clamp_min(0).sqrt()  # (H,)

        # Cauchy-Schwarz for remaining dims
        ball_part = q_resid_norm.unsqueeze(-1) * half_resid_norm  # (H, K)

        upper = base_score + exact_part + ball_part
        return upper > th.unsqueeze(-1)

    half_norm = half.norm(dim=-1)  # span_ball radius
    return gate, {
        "d": d,
        "half_norm_mean": float(half_norm.mean()),
        "half_resid_norm_mean": float(half_resid_norm.mean()),
    }


enclose_partial_aabb_d4 = _make_partial_aabb(4)
enclose_partial_aabb_d8 = _make_partial_aabb(8)
enclose_partial_aabb_d16 = _make_partial_aabb(16)
enclose_partial_aabb_d32 = _make_partial_aabb(32)
enclose_partial_aabb_d64 = _make_partial_aabb(64)
