"""AABB with hot key bypass.

Identifies keys with the largest norms (most likely to be in top-k) and
always includes them without gating. The remaining keys are clustered
and gated with AABB. This makes the remaining clusters tighter since
high-norm outliers are removed.

The hot keys have zero gate cost but are always scanned.
"""

from __future__ import annotations

import torch


def _make_hot_key_aabb(n_hot_frac: float):
    def enclose(keys, assign, centers, K, bf):
        return enclose_hot_key_aabb(keys, assign, centers, K, bf,
                                     n_hot_frac=n_hot_frac)
    enclose.__name__ = f"enclose_hot_key_aabb_{int(n_hot_frac*100)}pct"
    return enclose


def enclose_hot_key_aabb(keys, assign, centers, K, bf, n_hot_frac: float = 0.03):
    """
    AABB with hot key bypass.

    Args:
        n_hot_frac: fraction of keys to treat as "hot" (always included)
    """
    H, N, D = keys.shape
    device = keys.device

    n_hot = max(1, int(N * n_hot_frac))

    # Identify hot keys: those with largest norms (most likely to appear in top-k)
    norms = keys.norm(dim=-1)  # (H, N)
    _, hot_indices = norms.topk(n_hot, dim=-1)  # (H, n_hot)

    # Build hot key mask
    hot_mask = torch.zeros(H, N, dtype=torch.bool, device=device)
    hot_mask.scatter_(1, hot_indices, True)

    # Build AABB only on non-hot keys
    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)  # (H, N, D)

    # Mask out hot keys: set their contribution to neutral
    keys_cold = keys.clone()
    assign_cold = assign.clone()

    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Now build tighter AABB excluding hot keys
    lo_tight = torch.full((H, K, D), float("inf"), device=device)
    hi_tight = torch.full((H, K, D), float("-inf"), device=device)

    # Only scatter non-hot keys
    cold_mask = ~hot_mask  # (H, N)
    cold_mask_exp = cold_mask.unsqueeze(-1).expand(-1, -1, D)
    keys_masked = keys.where(cold_mask_exp, torch.full_like(keys, float("inf")))
    lo_tight.scatter_reduce_(1, idx_exp, keys_masked, reduce="amin", include_self=False)
    keys_masked = keys.where(cold_mask_exp, torch.full_like(keys, float("-inf")))
    hi_tight.scatter_reduce_(1, idx_exp, keys_masked, reduce="amax", include_self=False)

    # For clusters that become empty (all hot), use original bounds
    still_empty = lo_tight[:, :, 0].isinf()
    lo_tight[still_empty] = lo[still_empty]
    hi_tight[still_empty] = hi[still_empty]

    def gate(q, th):
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        # AABB gate on tight (cold) bounds
        max_dot = torch.maximum(q_exp * lo_tight, q_exp * hi_tight).sum(dim=-1)  # (H, K)
        cluster_pass = max_dot > th.unsqueeze(-1)

        # Hot keys always pass: mark their clusters as passing
        hot_clusters = assign[torch.arange(H, device=device).unsqueeze(1).expand_as(hot_indices),
                              hot_indices.clamp(0, N-1)]  # (H, n_hot)
        for h in range(H):
            cluster_pass[h].scatter_(0, hot_clusters[h], True)

        return cluster_pass

    vol = (hi_tight - lo_tight).clamp_min(0).prod(dim=-1)
    return gate, {
        "n_hot": n_hot,
        "hot_frac": n_hot / N,
        "vol_mean": float(vol.mean()),
    }


enclose_hot_key_aabb_3pct = _make_hot_key_aabb(0.03)
enclose_hot_key_aabb_5pct = _make_hot_key_aabb(0.05)
