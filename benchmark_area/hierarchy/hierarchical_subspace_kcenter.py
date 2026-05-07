"""
Hierarchical subspace k-center index with ball_centroid enclosures.

Structure (per subspace, per head):

    Level 0 (leaf):  k-center on N points  → K_0 = ceil(N/bf) clusters
    Level 1:         k-center on K_0 centers → K_1 = ceil(K_0/bf) clusters
      radii via triangle inequality: r_1[k] = max_{c ∈ children(k)} dist(center_1[k], center_0[c]) + r_0[c]
    Level 2:         k-center on K_1 centers → K_2 = ceil(K_1/bf) clusters
      ...
    Level L-1 (top): K_{L-1} = ceil(K_{L-2}/bf) clusters

Gate (per subspace):
    Upper bound on partial dot product q_s · k_s for any descendant point:
        ub_l[k] = dot(center_l[k], q_s) + radius_l[k] * ||q_s||

    Starting from the top, prune clusters whose ub < threshold, then
    descend into surviving clusters' children.

Across subspaces:
    A leaf cluster survives only if it survives in ALL subspaces (AND-filter
    with per-subspace thresholds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ── k-center (reused from quick_pruning) ──────────────────────────────


def _kcenter(points: torch.Tensor, K: int, refine_iter: int = 5):
    """Farthest-point k-center + Lloyd refinement.

    Args:
        points: (H, M, d) — M points of dimension d, per head.
        K: number of clusters.

    Returns:
        assign: (H, M) int64 cluster ids.
        centers: (H, K, d) centroids.
    """
    H, M, d = points.shape
    device = points.device

    if K >= M:
        # Every point is its own cluster.
        assign = torch.arange(M, device=device).unsqueeze(0).expand(H, -1)
        centers = points.clone()
        if K > M:
            pad = torch.zeros(H, K - M, d, device=device, dtype=points.dtype)
            centers = torch.cat([centers, pad], dim=1)
        return assign, centers

    center_idx = torch.zeros(H, K, dtype=torch.long, device=device)
    center_idx[:, 0] = torch.randint(0, M, (H,), device=device)

    first = points.gather(1, center_idx[:, :1].unsqueeze(-1).expand(-1, 1, d))
    min_dist = (points - first).norm(dim=-1)

    for i in range(1, K):
        farthest = min_dist.argmax(dim=1)
        center_idx[:, i] = farthest
        new_c = points.gather(1, farthest.view(H, 1, 1).expand(-1, 1, d))
        min_dist = torch.min(min_dist, (points - new_c).norm(dim=-1))

    centers = points.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, d))

    for _ in range(refine_iter):
        dists = torch.cdist(points, centers)
        assign = dists.argmin(dim=2)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, d), points)
        counts = torch.zeros(H, K, device=device)
        counts.scatter_add_(1, assign, torch.ones(H, M, device=device))

        empty = counts == 0
        counts = counts.clamp_min(1)
        new_centers = new_centers / counts.unsqueeze(-1)

        if empty.any():
            cur_d = torch.cdist(points, new_centers).min(dim=2).values
            for h in range(H):
                for k_idx in empty[h].nonzero(as_tuple=True)[0]:
                    far = cur_d[h].argmax()
                    new_centers[h, k_idx] = points[h, far]
                    cur_d[h, far] = 0.0

        centers = new_centers

    assign = torch.cdist(points, centers).argmin(dim=2)
    return assign, centers


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class LevelData:
    """One level of the hierarchy for a single subspace."""
    centers: torch.Tensor    # (H, K_l, d_s)
    radii: torch.Tensor      # (H, K_l)
    assign: torch.Tensor     # (H, M) where M = N at level 0, K_{l-1} at level l
    K: int                   # number of clusters at this level


@dataclass
class HierarchicalSubspaceIndex:
    """Multi-level subspace k-center index."""
    n_subspaces: int
    num_levels: int
    bf: int
    N: int                   # number of original points
    dim_slices: list[tuple[int, int]] = field(default_factory=list)
    # levels[s] = list of LevelData, from level 0 (leaf) to level num_levels-1 (top)
    levels: list[list[LevelData]] = field(default_factory=list)


# ── Builder ────────────────────────────────────────────────────────────


def _ball_centroid_radii(points, assign, centers, K):
    """Max distance from centroid to any assigned point."""
    H, _, d = points.shape
    device = points.device
    parent = centers.gather(1, assign.unsqueeze(-1).expand(-1, -1, d))
    dists = (points - parent).norm(dim=-1)
    radii = torch.full((H, K), 0.0, device=device)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)
    return radii


def _triangle_radii(parent_centers, child_centers, child_radii, assign, K):
    """Enclosing radius via triangle inequality.

    For parent cluster k:
        r_parent[k] = max_{c in children(k)} ||parent_center[k] - child_center[c]|| + child_r[c]

    This guarantees the parent ball encloses all child balls.
    """
    H, _, d = child_centers.shape
    device = child_centers.device

    # Distance from each child center to its assigned parent center
    parent_for_child = parent_centers.gather(
        1, assign.unsqueeze(-1).expand(-1, -1, d)
    )  # (H, M, d)
    dist_to_parent = (child_centers - parent_for_child).norm(dim=-1)  # (H, M)

    # Enclosing radius = dist + child_radius
    enclosing = dist_to_parent + child_radii  # (H, M)

    radii = torch.full((H, K), 0.0, device=device)
    radii.scatter_reduce_(1, assign, enclosing, reduce="amax", include_self=True)
    return radii


def build_hierarchical_index(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int,
    num_levels: int,
    refine_iter: int = 5,
) -> HierarchicalSubspaceIndex:
    """Build a multi-level subspace k-center hierarchy.

    Args:
        keys: (H, N, D) full-dimensional key vectors.
        bf: branching factor.
        n_subspaces: number of contiguous dimension slices.
        num_levels: depth of the hierarchy (1 = flat, 2 = two levels, etc.).
        refine_iter: Lloyd refinement iterations per k-center call.

    Returns:
        HierarchicalSubspaceIndex.
    """
    H, N, D = keys.shape
    device = keys.device

    sub_dim = D // n_subspaces
    remainder = D % n_subspaces
    dim_slices = []
    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        dim_slices.append((offset, offset + sd))
        offset += sd

    idx = HierarchicalSubspaceIndex(
        n_subspaces=n_subspaces, num_levels=num_levels, bf=bf, N=N,
        dim_slices=dim_slices,
    )

    # Build all leaf partitions first so the L=0 clustering for each subspace
    # matches the flat single-level build under the same RNG seed. Otherwise the
    # extra random initialization draws for upper levels perturb the leaf
    # partitions of later subspaces, which makes L>1 results incomparable to L=1.
    prev_centers_per_subspace: list[torch.Tensor] = []
    prev_radii_per_subspace: list[torch.Tensor] = []
    prev_k_per_subspace: list[int] = []

    for s in range(n_subspaces):
        start, end = dim_slices[s]
        keys_sub = keys[:, :, start:end].contiguous()

        subspace_levels: list[LevelData] = []

        # Level 0: cluster the N original points
        K_0 = max(1, math.ceil(N / bf))
        assign_0, centers_0 = _kcenter(keys_sub, K_0, refine_iter=refine_iter)
        radii_0 = _ball_centroid_radii(keys_sub, assign_0, centers_0, K_0)
        subspace_levels.append(LevelData(
            centers=centers_0, radii=radii_0, assign=assign_0, K=K_0,
        ))
        idx.levels.append(subspace_levels)
        prev_centers_per_subspace.append(centers_0)
        prev_radii_per_subspace.append(radii_0)
        prev_k_per_subspace.append(K_0)

    # Higher levels: cluster the previous level's centers for every subspace.
    for lvl in range(1, num_levels):
        for s in range(n_subspaces):
            prev_centers = prev_centers_per_subspace[s]
            prev_radii = prev_radii_per_subspace[s]
            prev_K = prev_k_per_subspace[s]

            K_l = max(1, math.ceil(prev_K / bf))
            assign_l, centers_l = _kcenter(prev_centers, K_l, refine_iter=refine_iter)
            radii_l = _triangle_radii(
                centers_l, prev_centers, prev_radii, assign_l, K_l,
            )
            idx.levels[s].append(LevelData(
                centers=centers_l, radii=radii_l, assign=assign_l, K=K_l,
            ))
            prev_centers_per_subspace[s] = centers_l
            prev_radii_per_subspace[s] = radii_l
            prev_k_per_subspace[s] = K_l

    return idx


# ── Hierarchical gate ──────────────────────────────────────────────────


def _expand_to_q(tensor: torch.Tensor, q_head_to_kv: torch.Tensor | None):
    """Expand an (H_kv, ...) tensor to (H_q, ...) using the GQA mapping."""
    if q_head_to_kv is None:
        return tensor
    return tensor[q_head_to_kv]


def hierarchical_gate_per_subspace(
    levels: list[LevelData],
    q_sub: torch.Tensor,
    th_sub: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Top-down pruning for one subspace.

    Args:
        levels: list of LevelData, index 0 = leaf, index L-1 = top.
            Built on H_kv heads.
        q_sub: (H_q, d_s) subspace query.
        th_sub: (H_q,) per-head threshold for this subspace.
        q_head_to_kv: (H_q,) GQA mapping or None.

    Returns:
        survive_leaf: (H_q, K_0) bool — which leaf clusters survive.
        level_stats: list of dicts with per-level pruning info.
    """
    H_q = q_sub.shape[0]
    device = q_sub.device
    q_sub_norm = q_sub.norm(dim=-1)  # (H_q,)
    num_levels = len(levels)

    level_stats = []

    # Start from top level
    top = num_levels - 1
    top_level = levels[top]

    # Expand index tensors from H_kv to H_q
    centers = _expand_to_q(top_level.centers, q_head_to_kv)
    radii = _expand_to_q(top_level.radii, q_head_to_kv)

    scores = torch.einsum("hkd,hd->hk", centers, q_sub)
    ub = scores + radii * q_sub_norm.unsqueeze(-1)
    survive = ub >= th_sub.unsqueeze(-1)  # (H_q, K_top)

    level_stats.append({
        "level": top,
        "K": top_level.K,
        "checked": top_level.K,
        "survived": survive.float().sum(dim=1).mean().item(),
        "pass_rate": survive.float().mean().item(),
    })

    # Descend through levels
    for lvl in range(top - 1, -1, -1):
        level_data = levels[lvl]
        assign_to_parent = _expand_to_q(levels[lvl + 1].assign, q_head_to_kv)

        parent_pass = survive.gather(1, assign_to_parent)  # (H_q, K_lvl)

        centers = _expand_to_q(level_data.centers, q_head_to_kv)
        radii = _expand_to_q(level_data.radii, q_head_to_kv)

        scores = torch.einsum("hkd,hd->hk", centers, q_sub)
        ub = scores + radii * q_sub_norm.unsqueeze(-1)
        self_pass = ub >= th_sub.unsqueeze(-1)

        survive = parent_pass & self_pass

        checked_per_head = parent_pass.float().sum(dim=1)
        passed_per_head = survive.float().sum(dim=1)
        mean_checked = checked_per_head.mean().item()
        mean_passed = passed_per_head.mean().item()

        level_stats.append({
            "level": lvl,
            "K": level_data.K,
            "checked": mean_checked,
            "survived": mean_passed,
            "pass_rate": (mean_passed / max(1e-12, mean_checked)),
        })

    return survive, level_stats


def hierarchical_gate(
    idx: HierarchicalSubspaceIndex,
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[list[dict]]]:
    """Full hierarchical AND-gate across all subspaces.

    Args:
        q: (H_q, D) unit-norm query vectors.
        th_per_subspace: (S, H_q) per-subspace thresholds.
        q_head_to_kv: (H_q,) GQA mapping or None. Index is built on H_kv.

    Returns:
        survive_points: (H_q, N) bool — which points survive.
        all_stats: list (per subspace) of list (per level) of stat dicts.
    """
    H_q = q.shape[0]
    device = q.device
    N = idx.N

    survive_points = torch.ones(H_q, N, dtype=torch.bool, device=device)
    all_stats = []

    for s in range(idx.n_subspaces):
        start, end = idx.dim_slices[s]
        q_sub = q[:, start:end]
        th_sub = th_per_subspace[s]

        survive_leaf, level_stats = hierarchical_gate_per_subspace(
            idx.levels[s], q_sub, th_sub, q_head_to_kv,
        )

        # Map leaf cluster survival to point survival
        leaf_assign = _expand_to_q(idx.levels[s][0].assign, q_head_to_kv)
        point_pass = survive_leaf.gather(1, leaf_assign)
        survive_points &= point_pass

        all_stats.append(level_stats)

    return survive_points, all_stats
