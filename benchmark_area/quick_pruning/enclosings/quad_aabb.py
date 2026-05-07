"""Quad-AABB: recursively bisect each cluster twice for 4 sub-boxes."""

from __future__ import annotations

import torch


def enclose_quad_aabb(keys, assign, centers, K, bf):
    """
    Two levels of median-split along the widest AABB dimension, producing
    4 sub-boxes per cluster.  A cluster passes if ANY of its 4 sub-boxes
    could exceed the threshold.

    Roughly 4x the box-count of plain AABB but each sub-box is ~1/4 volume.

    Returns:
        gate: callable(q, th) -> (H, K) bool.
        info: dict with diagnostic stats.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    # ── Level 1: split each cluster into 2 halves ──
    lo_full = torch.full((H, K, D), float("inf"), device=device)
    hi_full = torch.full((H, K, D), float("-inf"), device=device)
    lo_full.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi_full.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)
    empty = lo_full[:, :, 0].isinf()
    if empty.any():
        lo_full[empty] = 0.0
        hi_full[empty] = 0.0

    span = hi_full - lo_full
    widest1 = span.argmax(dim=-1)  # (H, K)

    # Split value: center along widest dim
    center_for_key = centers.gather(1, idx_exp)
    widest1_per_key = widest1.gather(1, assign)  # (H, N)
    key_val1 = keys.gather(2, widest1_per_key.unsqueeze(-1)).squeeze(-1)
    split_val1 = center_for_key.gather(2, widest1_per_key.unsqueeze(-1)).squeeze(-1)
    sub1 = (key_val1 >= split_val1).long()  # 0 or 1

    # ── Level 2: split each of the 2 halves again ──
    K2 = K * 2
    combined1 = assign * 2 + sub1
    comb1_exp = combined1.unsqueeze(-1).expand(-1, -1, D)

    lo2 = torch.full((H, K2, D), float("inf"), device=device)
    hi2 = torch.full((H, K2, D), float("-inf"), device=device)
    lo2.scatter_reduce_(1, comb1_exp, keys, reduce="amin", include_self=False)
    hi2.scatter_reduce_(1, comb1_exp, keys, reduce="amax", include_self=False)
    empty2 = lo2[:, :, 0].isinf()
    if empty2.any():
        lo2[empty2] = 0.0
        hi2[empty2] = 0.0

    span2 = hi2 - lo2
    widest2 = span2.argmax(dim=-1)  # (H, K2)

    # Centers of K2 sub-clusters
    centers2 = (lo2 + hi2) / 2
    centers2_for_key = centers2.gather(1, comb1_exp)
    widest2_per_key = widest2.gather(1, combined1)
    key_val2 = keys.gather(2, widest2_per_key.unsqueeze(-1)).squeeze(-1)
    split_val2 = centers2_for_key.gather(2, widest2_per_key.unsqueeze(-1)).squeeze(-1)
    sub2 = (key_val2 >= split_val2).long()

    # ── Build 4 sub-AABBs per cluster ──
    K4 = K * 4
    combined_final = assign * 4 + sub1 * 2 + sub2  # (H, N)
    cfinal_exp = combined_final.unsqueeze(-1).expand(-1, -1, D)

    lo4 = torch.full((H, K4, D), float("inf"), device=device)
    hi4 = torch.full((H, K4, D), float("-inf"), device=device)
    lo4.scatter_reduce_(1, cfinal_exp, keys, reduce="amin", include_self=False)
    hi4.scatter_reduce_(1, cfinal_exp, keys, reduce="amax", include_self=False)
    empty4 = lo4[:, :, 0].isinf()
    if empty4.any():
        lo4[empty4] = 0.0
        hi4[empty4] = 0.0

    # Reshape to (H, K, 4, D) for gate
    lo4 = lo4.view(H, K, 4, D)
    hi4 = hi4.view(H, K, 4, D)

    def gate(q, th):
        th_exp = th.unsqueeze(-1)  # (H, 1)
        q_exp = q.unsqueeze(1).unsqueeze(2)  # (H, 1, 1, D)

        scores = torch.maximum(q_exp * lo4, q_exp * hi4).sum(dim=-1)  # (H, K, 4)
        # Cluster passes if ANY sub-box passes
        passes = scores > th_exp.unsqueeze(-1)  # (H, K, 4)
        return passes.any(dim=-1)  # (H, K)

    return gate, {"n_subboxes": 4}
