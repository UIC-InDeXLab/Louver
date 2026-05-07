"""Origin-based single-axis interval bound."""

from __future__ import annotations

import torch


def enclose_axis_interval(keys, assign, centers, K, bf):
    """
    Bound each cluster by an interval on a local axis plus an orthogonal ball.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dir_keys = keys / key_norms

    axis = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    axis.scatter_add_(1, idx_exp, dir_keys)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_child = axis.gather(1, idx_exp)

    coeff = (keys * axis_child).sum(dim=-1)
    perp = (keys.square().sum(dim=-1) - coeff.square()).clamp_min(0).sqrt()

    coeff_lo = torch.full((H, K), float("inf"), device=device, dtype=keys.dtype)
    coeff_hi = torch.full((H, K), float("-inf"), device=device, dtype=keys.dtype)
    coeff_lo.scatter_reduce_(1, assign, coeff, reduce="amin", include_self=False)
    coeff_hi.scatter_reduce_(1, assign, coeff, reduce="amax", include_self=False)

    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    residual.scatter_reduce_(1, assign, perp, reduce="amax", include_self=True)

    empty = coeff_lo.isinf()
    if empty.any():
        coeff_lo[empty] = 0.0
        coeff_hi[empty] = 0.0

    def gate(q, th):
        proj = torch.einsum("hkd,hd->hk", axis, q).clamp(-1, 1)
        proj_perp = (1.0 - proj.square()).clamp_min(0).sqrt()
        upper = torch.maximum(proj * coeff_lo, proj * coeff_hi) + residual * proj_perp
        return upper > th.unsqueeze(-1)

    return gate, {
        "coeff_span_mean": float((coeff_hi - coeff_lo).mean()),
        "residual_mean": float(residual.mean()),
    }
