"""Selected-dimension AABB with residual ball enclosure."""

from __future__ import annotations

import torch


def enclose_topk_aabb_residual(keys, assign, centers, K, bf, rank: int = 4):
    """
    Use exact AABB support on a few salient dimensions and a residual L2 bound
    for the remaining dimensions.

    This is much cheaper to build than per-cluster low-rank SVD bounds:
    everything is derived from scatter-reduced min/max statistics.
    """
    H, N, D = keys.shape
    device = keys.device
    R = min(max(1, rank), D)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    pos_dev = (hi - centers).clamp_min(0)
    neg_dev = (centers - lo).clamp_min(0)
    delta = torch.maximum(pos_dev, neg_dev)
    span = hi - lo

    top_idx = span.topk(R, dim=-1).indices
    top_lo = lo.gather(2, top_idx)
    top_hi = hi.gather(2, top_idx)
    top_center = centers.gather(2, top_idx)
    top_delta = delta.gather(2, top_idx)

    residual_sq = (delta.square().sum(dim=-1) - top_delta.square().sum(dim=-1)).clamp_min(0)

    parent_for_child = centers.gather(1, idx_exp)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(
        1,
        assign,
        (keys - parent_for_child).norm(dim=-1),
        reduce="amax",
        include_self=True,
    )

    def gate(q, th):
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)

        q_exp = q.unsqueeze(1).expand(-1, K, -1)
        top_q = q_exp.gather(2, top_idx)
        exact_top = torch.maximum(top_q * top_lo, top_q * top_hi)
        top_extra = exact_top - (top_q * top_center)

        residual_q_sq = (q_norm_sq - top_q.square().sum(dim=-1)).clamp_min(0)
        residual_upper = center_scores + top_extra.sum(dim=-1) + residual_sq.sqrt() * residual_q_sq.sqrt()

        ball_upper = center_scores + ball_radii * q_norm_sq.sqrt()
        return torch.minimum(residual_upper, ball_upper) > th.unsqueeze(-1)

    return gate, {
        "rank": R,
        "span_mean": float(span.mean()),
        "residual_mean": float(residual_sq.sqrt().mean()),
        "ball_r_mean": float(ball_radii.mean()),
    }
