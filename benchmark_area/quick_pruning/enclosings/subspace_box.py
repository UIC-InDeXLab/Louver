"""Low-rank oriented box with residual ball enclosure."""

from __future__ import annotations

import torch


def enclose_subspace_box(keys, assign, centers, K, bf, rank: int = 2):
    """
    Bound each cluster in an oriented low-rank box plus an orthogonal residual.

    For any query q, the support bound is:
      mu.q + sum_i |u_i.q| * alpha_i + rho * ||q_perp||
    where u_i are the principal axes, alpha_i are max absolute coefficients
    along those axes, and rho is the max residual norm outside the subspace.
    """
    H, N, D = keys.shape
    device = keys.device
    R = max(1, rank)

    basis = torch.zeros(H, K, R, D, device=device, dtype=keys.dtype)
    alpha = torch.zeros(H, K, R, device=device, dtype=keys.dtype)
    residual = torch.zeros(H, K, device=device, dtype=keys.dtype)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent_for_child = centers.gather(1, idx_exp)
    ball_radii = torch.full((H, K), 0.0, device=device)
    ball_radii.scatter_reduce_(
        1,
        assign,
        (keys - parent_for_child).norm(dim=-1),
        reduce="amax",
        include_self=True,
    )

    for h in range(H):
        for k in range(K):
            idx = (assign[h] == k).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue

            points = keys[h, idx]
            centered = points - centers[h, k]
            local_rank = min(R, max(0, int(points.shape[0]) - 1), D)
            if local_rank == 0 or float(centered.square().sum()) <= 1e-12:
                continue

            try:
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                axes = vh[:local_rank]
            except RuntimeError:
                var = centered.square().mean(dim=0)
                axes = torch.zeros(local_rank, D, device=device, dtype=keys.dtype)
                axes[0, var.argmax()] = 1.0

            coeff = centered @ axes.transpose(0, 1)
            recon = coeff @ axes

            basis[h, k, :local_rank] = axes
            alpha[h, k, :local_rank] = coeff.abs().amax(dim=0)
            residual[h, k] = (centered - recon).norm(dim=-1).amax()

    def gate(q, th):
        q_norm_sq = q.square().sum(dim=-1, keepdim=True)
        center_scores = torch.einsum("hkd,hd->hk", centers, q)

        proj = torch.einsum("hkrd,hd->hkr", basis, q)
        box_slack = (proj.abs() * alpha).sum(dim=-1)
        perp_norm = (q_norm_sq - proj.square().sum(dim=-1)).clamp_min(0).sqrt()
        subspace_upper = center_scores + box_slack + residual * perp_norm

        ball_upper = center_scores + ball_radii * q_norm_sq.sqrt()
        return torch.minimum(subspace_upper, ball_upper) > th.unsqueeze(-1)

    return gate, {
        "rank": R,
        "alpha_mean": float(alpha.mean()),
        "residual_mean": float(residual.mean()),
        "ball_r_mean": float(ball_radii.mean()),
    }
