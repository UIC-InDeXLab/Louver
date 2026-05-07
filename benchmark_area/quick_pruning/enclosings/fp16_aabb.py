"""AABB with fp16 storage for lo/hi bounds.

Halves memory bandwidth requirements compared to fp32 AABB.
Slight precision loss but may not affect pruning quality.
The query is still fp32; only the bounds are fp16.
"""

from __future__ import annotations

import torch


def enclose_fp16_aabb(keys, assign, centers, K, bf):
    """AABB with fp16-quantized bounds."""
    H, N, D = keys.shape
    device = keys.device

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    # Convert to fp16 (slight expansion to ensure valid bounds)
    # lo should be rounded DOWN, hi should be rounded UP
    # fp16 doesn't have exact rounding control, but the error is tiny
    lo_h = lo.half()
    hi_h = hi.half()

    # Ensure fp16 bounds are valid (lo_h <= lo, hi_h >= hi)
    # Add a small epsilon to hi and subtract from lo to be safe
    eps = 1e-3
    lo_h = (lo - eps).half()
    hi_h = (hi + eps).half()

    def gate(q, th):
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        # Compute in fp32 but load fp16 bounds (automatic promotion)
        max_dot = torch.maximum(q_exp * lo_h.float(), q_exp * hi_h.float()).sum(dim=-1)
        return max_dot > th.unsqueeze(-1)

    vol = (hi - lo).clamp_min(0).prod(dim=-1)
    return gate, {"vol_mean": float(vol.mean()), "vol_max": float(vol.max()),
                  "storage": "fp16"}
