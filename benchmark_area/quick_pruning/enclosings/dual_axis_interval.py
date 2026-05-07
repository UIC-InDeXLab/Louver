"""Origin-based two-axis interval bound."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def enclose_dual_axis_interval(keys, assign, centers, K, bf):
    """
    Extend the single-axis interval bound with a second local axis and a small
    residual ball for the remaining orthogonal energy.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    dir_keys = keys / key_norms

    axis_u = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    axis_u.scatter_add_(1, idx_exp, dir_keys)
    axis_u = axis_u / axis_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_u_child = axis_u.gather(1, idx_exp)

    coeff_u = (keys * axis_u_child).sum(dim=-1)
    resid_u = keys - coeff_u.unsqueeze(-1) * axis_u_child

    max_abs = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    max_abs.scatter_reduce_(1, idx_exp, resid_u.abs(), reduce="amax", include_self=True)
    pivot_dim = max_abs.argmax(dim=-1)
    pivot_child = pivot_dim.gather(1, assign)
    pivot_vals = resid_u.gather(2, pivot_child.unsqueeze(-1)).squeeze(-1)
    pivot_sign = torch.where(pivot_vals < 0, -torch.ones_like(pivot_vals), torch.ones_like(pivot_vals))

    axis_v_raw = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    axis_v_raw.scatter_add_(1, idx_exp, resid_u * pivot_sign.unsqueeze(-1))
    axis_v_raw = axis_v_raw - (axis_v_raw * axis_u).sum(dim=-1, keepdim=True) * axis_u

    fallback = F.one_hot(pivot_dim, num_classes=D).to(device=device, dtype=keys.dtype)
    fallback = fallback - (fallback * axis_u).sum(dim=-1, keepdim=True) * axis_u

    raw_norm = axis_v_raw.norm(dim=-1, keepdim=True)
    fallback_norm = fallback.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis_v = torch.where(
        raw_norm > 1e-6,
        axis_v_raw / raw_norm.clamp_min(1e-12),
        fallback / fallback_norm,
    )
    axis_v_child = axis_v.gather(1, idx_exp)

    coeff_v = (keys * axis_v_child).sum(dim=-1)
    residual = (
        keys.square().sum(dim=-1) - coeff_u.square() - coeff_v.square()
    ).clamp_min(0).sqrt()

    lo_u = torch.full((H, K), float("inf"), device=device, dtype=keys.dtype)
    hi_u = torch.full((H, K), float("-inf"), device=device, dtype=keys.dtype)
    lo_v = torch.full((H, K), float("inf"), device=device, dtype=keys.dtype)
    hi_v = torch.full((H, K), float("-inf"), device=device, dtype=keys.dtype)
    lo_u.scatter_reduce_(1, assign, coeff_u, reduce="amin", include_self=False)
    hi_u.scatter_reduce_(1, assign, coeff_u, reduce="amax", include_self=False)
    lo_v.scatter_reduce_(1, assign, coeff_v, reduce="amin", include_self=False)
    hi_v.scatter_reduce_(1, assign, coeff_v, reduce="amax", include_self=False)

    max_residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    max_residual.scatter_reduce_(1, assign, residual, reduce="amax", include_self=True)

    empty = lo_u.isinf()
    if empty.any():
        lo_u[empty] = 0.0
        hi_u[empty] = 0.0
        lo_v[empty] = 0.0
        hi_v[empty] = 0.0

    def gate(q, th):
        proj_u = torch.einsum("hkd,hd->hk", axis_u, q).clamp(-1, 1)
        proj_v = torch.einsum("hkd,hd->hk", axis_v, q).clamp(-1, 1)
        proj_perp = (1.0 - proj_u.square() - proj_v.square()).clamp_min(0).sqrt()

        upper = torch.maximum(proj_u * lo_u, proj_u * hi_u)
        upper = upper + torch.maximum(proj_v * lo_v, proj_v * hi_v)
        upper = upper + max_residual * proj_perp
        return upper > th.unsqueeze(-1)

    return gate, {
        "u_span_mean": float((hi_u - lo_u).mean()),
        "v_span_mean": float((hi_v - lo_v).mean()),
        "residual_mean": float(max_residual.mean()),
    }
