"""Query-adaptive gate: use observed query statistics to build tighter bounds.

Key insight: attention queries in LLMs are NOT uniformly distributed — they
concentrate in a low-dimensional subspace. If we know the typical query
direction d, we can build per-cluster intervals along d that are much tighter
than an isotropic ball.

Approach:
1. Observe queries and compute m principal directions of the query distribution.
2. For each cluster, compute intervals [lo, hi] of key projections onto those
   query directions, plus residual ball norm.
3. Gate: project query onto those m directions, use intervals + ball.

The crucial difference from PCA-of-keys: we use the QUERY subspace, not the
key subspace. Queries define the halfspace directions we need to prune against.

Gate cost: 1 dot product (centroid) + m/D additional per cluster ≈ 1.0-1.1.

This requires access to queries at build time (or online updating). In the
benchmark, we use oracle access to queries for proof-of-concept.
"""

from __future__ import annotations

import torch


def make_query_adaptive_gate(keys, assign, centers, K, bf, queries, m: int = 4):
    """
    Build a query-adaptive gate using the top-m PCA directions of the queries.

    Args:
        keys: (H, N, D) key tensor
        assign: (H, N) cluster assignments
        centers: (H, K, D) cluster centers
        K: number of clusters
        bf: branching factor
        queries: (H, T, D) observed queries (for computing query PCA)
        m: number of query PCA directions to use

    Returns:
        gate: callable(q, th) -> (H, K) bool
        info: dict
    """
    H, N, D = keys.shape
    device = keys.device
    m = min(m, D, queries.shape[1])

    # Compute query PCA directions
    q_mean = queries.mean(dim=1, keepdim=True)
    q_centered = queries - q_mean
    _, _, Vt = torch.linalg.svd(q_centered, full_matrices=False)
    Q_dirs = Vt[:, :m, :]  # (H, m, D) — top-m query PCA directions

    # Project keys onto query PCA directions
    key_proj = torch.bmm(keys, Q_dirs.transpose(-2, -1))  # (H, N, m)

    # Per-cluster intervals on query directions
    idx_proj = assign.unsqueeze(-1).expand(-1, -1, m)
    lo = torch.full((H, K, m), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K, m), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, idx_proj, key_proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_proj, key_proj, reduce="amax", include_self=False)
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Residual: max ||key - projection onto query subspace|| per cluster
    key_recon = torch.bmm(key_proj, Q_dirs)
    resid_norm = (keys - key_recon).norm(dim=-1)
    res_max = torch.zeros(H, K, device=device, dtype=keys.dtype)
    res_max.scatter_reduce_(1, assign, resid_norm, reduce="amax", include_self=True)

    # Also store ball radius for min(adaptive, ball) tightening
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent = centers.gather(1, idx_exp)
    ball_r = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_r.scatter_reduce_(
        1, assign, (keys - parent).norm(dim=-1), reduce="amax", include_self=True
    )

    def gate(q, th):
        # Project query onto query PCA directions
        p = torch.bmm(q.unsqueeze(1), Q_dirs.transpose(-2, -1)).squeeze(1)  # (H, m)
        q_perp_norm = (q.square().sum(-1) - p.square().sum(-1)).clamp_min(0).sqrt()

        # Interval bound
        p_exp = p.unsqueeze(1)
        interval_score = torch.maximum(p_exp * lo, p_exp * hi).sum(dim=-1)
        adaptive_ub = interval_score + res_max * q_perp_norm.unsqueeze(-1)

        # Ball bound (centroid)
        c_score = torch.einsum("hkd,hd->hk", centers, q)
        ball_ub = c_score + ball_r

        # Take the tighter of the two
        upper = torch.minimum(adaptive_ub, ball_ub)
        return upper > th.unsqueeze(-1)

    # Variance explained by query PCA
    q_var = queries.square().sum(-1).mean()
    q_proj_var = torch.bmm(queries, Q_dirs.transpose(-2, -1)).square().sum(-1).mean()
    q_explained = float(q_proj_var / q_var.clamp_min(1e-12))

    return gate, {
        "m": m,
        "q_var_explained": q_explained,
        "res_max_mean": float(res_max.mean()),
        "ball_r_mean": float(ball_r.mean()),
    }


def make_query_mean_gate(keys, assign, centers, K, bf, queries):
    """
    Simplest query-adaptive gate: project keys onto the mean query direction,
    store per-cluster [lo, hi] + residual ball.

    This is a 1D "AABB" along the mean query direction. If queries are
    concentrated, this is nearly as good as exact but at g ≈ 1.0.
    """
    H, N, D = keys.shape
    device = keys.device

    # Mean query direction (unit)
    q_mean = queries.mean(dim=1)
    q_dir = q_mean / q_mean.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # (H, D)

    # Project keys onto mean query direction
    proj = (keys * q_dir.unsqueeze(1)).sum(dim=-1)  # (H, N)

    # Per-cluster [lo, hi]
    lo = torch.full((H, K), float("inf"), device=device, dtype=keys.dtype)
    hi = torch.full((H, K), float("-inf"), device=device, dtype=keys.dtype)
    lo.scatter_reduce_(1, assign, proj, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, assign, proj, reduce="amax", include_self=False)
    empty = lo.isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Residual norm: ||key - (key·d)d|| = ||key||²-(key·d)² for each key
    resid_norm = (keys.square().sum(-1) - proj.square()).clamp_min(0).sqrt()
    res_max = torch.zeros(H, K, device=device, dtype=keys.dtype)
    res_max.scatter_reduce_(1, assign, resid_norm, reduce="amax", include_self=True)

    # Ball fallback
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    parent = centers.gather(1, idx_exp)
    ball_r = torch.zeros(H, K, device=device, dtype=keys.dtype)
    ball_r.scatter_reduce_(
        1, assign, (keys - parent).norm(dim=-1), reduce="amax", include_self=True
    )

    def gate(q, th):
        # Project query onto mean direction
        q_proj = (q * q_dir).sum(dim=-1)  # (H,)
        q_perp_norm = (q.square().sum(-1) - q_proj.square()).clamp_min(0).sqrt()

        # 1D interval bound
        interval_ub = torch.maximum(q_proj.unsqueeze(-1) * lo, q_proj.unsqueeze(-1) * hi)
        adaptive_ub = interval_ub + res_max * q_perp_norm.unsqueeze(-1)

        # Ball fallback
        c_score = torch.einsum("hkd,hd->hk", centers, q)
        ball_ub = c_score + ball_r

        upper = torch.minimum(adaptive_ub, ball_ub)
        return upper > th.unsqueeze(-1)

    return gate, {
        "res_max_mean": float(res_max.mean()),
        "ball_r_mean": float(ball_r.mean()),
    }
