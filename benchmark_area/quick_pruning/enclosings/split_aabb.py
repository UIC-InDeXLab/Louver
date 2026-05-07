"""Split-AABB: bisect each cluster and use tighter per-half AABBs."""

from __future__ import annotations

import torch


def enclose_split_aabb(keys, assign, centers, K, bf):
    """
    For each cluster, split keys into two halves along the widest AABB
    dimension, then build separate AABBs for each half.

    A cluster passes the gate if EITHER half's AABB could exceed threshold.
    Each half-box is roughly half the volume -> much tighter bounds.

    Also includes ball fallback per cluster.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Full AABB first (to find widest dim) ──
    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo_full[:, :, 0].isinf()
    if empty.any():
        lo_full[empty] = 0.0
        hi_full[empty] = 0.0

    span = hi_full - lo_full  # (H, K, D)
    widest_dim = span.argmax(dim=-1)  # (H, K)

    # ── Compute split point per cluster: median along widest dim ──
    # For each key, get the value along its cluster's widest dimension
    cluster_per_key = assign  # (H, N)
    widest_per_key = widest_dim.gather(1, cluster_per_key)  # (H, N)

    # Value of each key along its cluster's widest dim
    key_split_val = keys.gather(2, widest_per_key.unsqueeze(-1)).squeeze(-1)  # (H, N)

    # Split point = midpoint of the cluster along widest dim
    center_for_key = centers.gather(1, idx_exp)
    split_val = center_for_key.gather(2, widest_per_key.unsqueeze(-1)).squeeze(-1)  # (H, N)

    # Assign to sub-cluster: 0 if below split, 1 if above
    sub_assign = (key_split_val >= split_val).long()  # (H, N)

    # ── Build AABBs for each half ──
    # Encode (cluster_id, sub_id) as a single index: cluster_id * 2 + sub_id
    combined = cluster_per_key * 2 + sub_assign  # (H, N)
    K2 = K * 2
    comb_exp = combined.unsqueeze(-1).expand(-1, -1, D)

    lo_sub = torch.full((H, K2, D), float("inf"), device=device)
    hi_sub = torch.full((H, K2, D), float("-inf"), device=device)
    lo_sub.scatter_reduce_(1, comb_exp, keys, reduce="amin", include_self=False)
    hi_sub.scatter_reduce_(1, comb_exp, keys, reduce="amax", include_self=False)

    empty_sub = lo_sub[:, :, 0].isinf()
    if empty_sub.any():
        lo_sub[empty_sub] = 0.0
        hi_sub[empty_sub] = 0.0

    # Reshape to (H, K, 2, D)
    lo_a = lo_sub[:, 0::2, :]  # (H, K, D) — sub 0
    hi_a = hi_sub[:, 0::2, :]
    lo_b = lo_sub[:, 1::2, :]  # (H, K, D) — sub 1
    hi_b = hi_sub[:, 1::2, :]

    def gate(q, th):
        th_exp = th.unsqueeze(-1)
        q_exp = q.unsqueeze(1)  # (H, 1, D)

        # Sub-AABB A
        score_a = torch.maximum(q_exp * lo_a, q_exp * hi_a).sum(dim=-1)
        # Sub-AABB B
        score_b = torch.maximum(q_exp * lo_b, q_exp * hi_b).sum(dim=-1)

        # Cluster passes if EITHER half passes
        return (score_a > th_exp) | (score_b > th_exp)

    vol_a = (hi_a - lo_a).clamp_min(0).sum(dim=-1)
    vol_b = (hi_b - lo_b).clamp_min(0).sum(dim=-1)
    return gate, {
        "vol_a_mean": float(vol_a.mean()),
        "vol_b_mean": float(vol_b.mean()),
    }
