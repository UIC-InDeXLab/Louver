"""Global PCA subspace box with residual ball fallback."""

from __future__ import annotations

import torch


def enclose_global_pca_box(keys, assign, centers, K, bf, rank: int = 3):
    """
    Use a small global PCA basis per head, then store a per-cluster box in that
    rotated subspace plus an orthogonal residual ball.

    This keeps the build fully batched while capturing correlated variation
    that axis-aligned AABB misses.
    """
    H, _, D = keys.shape
    device = keys.device
    R = min(max(1, rank), D)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent = centers.gather(1, idx_exp)
    diff = keys - parent

    key_mean = keys.mean(dim=1, keepdim=True)
    centered = keys - key_mean
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    basis = vt[:, :R, :].transpose(-2, -1).contiguous()

    diff_proj = torch.bmm(diff, basis)
    idx_proj = assign.unsqueeze(-1).expand(-1, -1, R)

    lo = torch.full((H, K, R), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, R), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_proj, diff_proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_proj, diff_proj, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    diff_sq = diff.square().sum(dim=-1)
    proj_sq = diff_proj.square().sum(dim=-1)
    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)
    residual.scatter_reduce_(
        1,
        assign,
        (diff_sq - proj_sq).clamp_min(0).sqrt(),
        reduce="amax",
        include_self=True,
    )

    ball_radii = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_radii.scatter_reduce_(1, assign, diff_sq.sqrt(), reduce="amax", include_self=True)

    def gate(q, th):
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)
        q_proj = torch.bmm(q.unsqueeze(1), basis).squeeze(1)
        q_proj_exp = q_proj.unsqueeze(1)
        q_perp = (q_norm_sq - q_proj.square().sum(dim=-1, keepdim=True)).clamp_min(0).sqrt()

        proj_upper = torch.maximum(q_proj_exp * lo, q_proj_exp * hi).sum(dim=-1)
        subspace_upper = center_scores + proj_upper + residual * q_perp

        ball_upper = center_scores + ball_radii * q_norm_sq.sqrt()
        return torch.minimum(subspace_upper, ball_upper) > th.unsqueeze(-1)

    return gate, {
        "rank": R,
        "residual_mean": float(residual.mean()),
        "ball_r_mean": float(ball_radii.mean()),
    }
