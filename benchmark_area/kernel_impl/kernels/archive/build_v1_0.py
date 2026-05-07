"""build_v1.0 — torch-vectorized subspace k-center + ball_centroid build.

Input:
    keys: (H, N, D) float32 on GPU
    bf:   branching factor
    n_subspaces: number of contiguous dim splits
    refine_iter: Lloyd iterations per subspace

Output:
    dim_slices: list[(start, end)] of length n_subspaces
    assigns:    list[(H, N) int64] — point -> cluster for each subspace
    centers:    list[(H, K, d_s) float32]
    radii:      list[(H, K) float32]
    child_order:   list[(H, N) int64]   — parent-major child permutation
    child_offsets: list[(H, K + 1) int32] — offsets into child_order
    child_counts:  list[(H, K) int32]   — members per parent
"""

from __future__ import annotations

import math

import torch

KERNEL_VERSION = "v1.0"


def _split_contiguous(D: int, S: int):
    sub = D // S
    rem = D % S
    out, off = [], 0
    for s in range(S):
        d = sub + (1 if s < rem else 0)
        out.append((off, off + d))
        off += d
    return out


def _kcenter_subspace(keys_sub: torch.Tensor, K: int, refine_iter: int):
    """Farthest-point seeding + Lloyd refinement.

    keys_sub: (H, N, d). returns (assign (H,N), centers (H,K,d))
    """
    H, N, d = keys_sub.shape
    device = keys_sub.device

    # Farthest-point seeding (vectorized over H)
    center_idx = torch.empty(H, K, dtype=torch.long, device=device)
    center_idx[:, 0] = torch.randint(0, N, (H,), device=device)
    first = keys_sub.gather(1, center_idx[:, :1, None].expand(-1, 1, d))
    min_dist = (keys_sub - first).norm(dim=-1)  # (H, N)
    for i in range(1, K):
        farthest = min_dist.argmax(dim=1)
        center_idx[:, i] = farthest
        new_c = keys_sub.gather(1, farthest.view(H, 1, 1).expand(-1, 1, d))
        min_dist = torch.minimum(min_dist, (keys_sub - new_c).norm(dim=-1))
    centers = keys_sub.gather(1, center_idx[..., None].expand(-1, -1, d))

    # Lloyd refinement
    ones_hn = torch.ones(H, N, device=device, dtype=keys_sub.dtype)
    for _ in range(refine_iter):
        # cdist is batched; for large N this is (H,N,K) memory
        dists = torch.cdist(keys_sub, centers)
        assign = dists.argmin(dim=2)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(
            1, assign[..., None].expand(-1, -1, d), keys_sub
        )
        counts = torch.zeros(H, K, device=device, dtype=keys_sub.dtype)
        counts.scatter_add_(1, assign, ones_hn)

        empty = counts == 0
        counts = counts.clamp_min(1.0)
        new_centers = new_centers / counts.unsqueeze(-1)

        if empty.any():
            # Re-seed empty clusters from farthest points.
            cur_d = torch.cdist(keys_sub, new_centers).min(dim=2).values  # (H, N)
            # Python fallback — empty clusters are rare with fpc seeding.
            for h in range(H):
                for k_idx in empty[h].nonzero(as_tuple=True)[0]:
                    far = cur_d[h].argmax()
                    new_centers[h, k_idx] = keys_sub[h, far]
                    cur_d[h, far] = 0.0

        centers = new_centers

    assign = torch.cdist(keys_sub, centers).argmin(dim=2)
    return assign, centers


def _ball_centroid(keys_sub, assign, centers, K):
    """Max distance from centroid to any member (per cluster)."""
    H, N, d = keys_sub.shape
    device = keys_sub.device
    parent = centers.gather(1, assign[..., None].expand(-1, -1, d))
    dists = (keys_sub - parent).norm(dim=-1)  # (H, N)
    radii = torch.zeros(H, K, device=device, dtype=keys_sub.dtype)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)
    return radii


def _parent_major_layout(assign: torch.Tensor, K: int):
    """Build a per-head parent-major child order for the current assignments.

    Exact fixed-width [i*bf:(i+1)*bf] blocks are not valid for the current
    unconstrained k-center assignment because cluster sizes can exceed bf.
    This stores the next-best layout: each parent's children are contiguous in
    a variable-length range [offset[i]:offset[i+1]].
    """
    H, N = assign.shape
    device = assign.device

    child_order = torch.empty(H, N, dtype=torch.long, device=device)
    child_offsets = torch.empty(H, K + 1, dtype=torch.int32, device=device)
    child_counts = torch.empty(H, K, dtype=torch.int32, device=device)

    for h in range(H):
        counts = torch.bincount(assign[h], minlength=K).to(torch.int32)
        order = torch.argsort(assign[h], stable=True)
        offsets = torch.empty(K + 1, dtype=torch.int32, device=device)
        offsets[0] = 0
        offsets[1:] = counts.cumsum(dim=0)

        child_order[h] = order
        child_offsets[h] = offsets
        child_counts[h] = counts

    return child_order, child_offsets, child_counts


def build(keys: torch.Tensor, bf: int, n_subspaces: int, refine_iter: int = 5):
    """Build subspace k-center index.

    Returns dict with dim_slices, assigns, centers, radii, K, N.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    slices = _split_contiguous(D, n_subspaces)

    assigns, centers, radii = [], [], []
    child_order, child_offsets, child_counts = [], [], []
    for start, end in slices:
        keys_sub = keys[:, :, start:end].contiguous()
        a, c = _kcenter_subspace(keys_sub, K, refine_iter)
        r = _ball_centroid(keys_sub, a, c, K)
        order, offsets, counts = _parent_major_layout(a, K)
        assigns.append(a)
        centers.append(c)
        radii.append(r)
        child_order.append(order)
        child_offsets.append(offsets)
        child_counts.append(counts)

    return {
        "dim_slices": slices,
        "assigns": assigns,
        "centers": centers,
        "radii": radii,
        "child_order": child_order,
        "child_offsets": child_offsets,
        "child_counts": child_counts,
        "K": K,
        "N": N,
    }


KERNEL = build
