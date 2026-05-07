"""
Subspace k-center pruning: product-quantization-style dimension splitting
with independent k-center clustering and ball_centroid enclosures per subspace.

Pruning rule: derive one threshold per subspace from the true full-space top-k
set, then keep a point only if it survives the ball gate in **every** subspace.

This module does not conform to the standard (assign, centers) clustering
interface because it maintains S independent indexes. Instead it exposes
a build/gate API consumed by comparison_subspace_kcenter.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ── k-center clustering (single subspace) ──────────────────────────────


def _kcenter(keys_sub: torch.Tensor, K: int, refine_iter: int = 5):
    """Farthest-point k-center + Lloyd refinement on a subspace slice.

    Args:
        keys_sub: (H, N, d) projected keys for one subspace.
        K: number of clusters.
        refine_iter: Lloyd refinement iterations.

    Returns:
        assign: (H, N) int64 cluster ids.
        centers: (H, K, d) centroids.
    """
    H, N, d = keys_sub.shape
    device = keys_sub.device

    center_idx = torch.zeros(H, K, dtype=torch.long, device=device)
    center_idx[:, 0] = torch.randint(0, N, (H,), device=device)

    first = keys_sub.gather(1, center_idx[:, :1].unsqueeze(-1).expand(-1, 1, d))
    min_dist = (keys_sub - first).norm(dim=-1)  # (H, N)

    for i in range(1, K):
        farthest = min_dist.argmax(dim=1)
        center_idx[:, i] = farthest
        new_c = keys_sub.gather(1, farthest.view(H, 1, 1).expand(-1, 1, d))
        min_dist = torch.min(min_dist, (keys_sub - new_c).norm(dim=-1))

    centers = keys_sub.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, d))

    for _ in range(refine_iter):
        dists = torch.cdist(keys_sub, centers)
        assign = dists.argmin(dim=2)

        new_centers = torch.zeros_like(centers)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, d), keys_sub)
        counts = torch.zeros(H, K, device=device)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        empty = counts == 0
        counts = counts.clamp_min(1)
        new_centers = new_centers / counts.unsqueeze(-1)

        if empty.any():
            cur_d = torch.cdist(keys_sub, new_centers).min(dim=2).values
            for h in range(H):
                for k_idx in empty[h].nonzero(as_tuple=True)[0]:
                    far = cur_d[h].argmax()
                    new_centers[h, k_idx] = keys_sub[h, far]
                    cur_d[h, far] = 0.0

        centers = new_centers

    assign = torch.cdist(keys_sub, centers).argmin(dim=2)
    return assign, centers


# ── Ball-centroid enclosure (single subspace) ──────────────────────────


def _ball_centroid(keys_sub, assign, centers, K):
    """Compute per-cluster ball radius for one subspace.

    Returns:
        radii: (H, K) max distance from centroid to any member.
    """
    H, N, d = keys_sub.shape
    device = keys_sub.device

    parent = centers.gather(1, assign.unsqueeze(-1).expand(-1, -1, d))
    dists = (keys_sub - parent).norm(dim=-1)  # (H, N)

    radii = torch.full((H, K), 0.0, device=device)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)
    return radii


# ── Multi-subspace builder ─────────────────────────────────────────────


@dataclass
class SubspaceKCenterIndex:
    """Holds S independent k-center ball indexes, one per subspace."""

    n_subspaces: int
    K: int
    strategy: str = "contiguous"
    # Per-subspace tensors (length = n_subspaces)
    assigns: list[torch.Tensor] = field(default_factory=list)   # each (H, N)
    centers: list[torch.Tensor] = field(default_factory=list)   # each (H, K, d_s)
    radii: list[torch.Tensor] = field(default_factory=list)     # each (H, K)
    dim_slices: list[tuple[int, int]] = field(default_factory=list)  # (start, end)
    # For projection-based strategies: (H, D, D) or None
    projection: torch.Tensor | None = None
    # For permutation-based strategies (interleaved): (D,) int64 or None
    permutation: torch.Tensor | None = None


# ── Subspace splitting strategies ──────────────────────────────────────


def _split_contiguous(D: int, S: int) -> list[tuple[int, int]]:
    """Contiguous slices: [0:d, d:2d, ...]."""
    sub_dim = D // S
    remainder = D % S
    slices = []
    offset = 0
    for s in range(S):
        sd = sub_dim + (1 if s < remainder else 0)
        slices.append((offset, offset + sd))
        offset += sd
    return slices


def _split_interleaved(D: int, S: int) -> list[list[int]]:
    """Interleaved/strided: subspace s gets dims {s, s+S, s+2S, ...}."""
    groups: list[list[int]] = [[] for _ in range(S)]
    for d in range(D):
        groups[d % S].append(d)
    return groups


def _make_random_orthogonal(D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Random orthogonal D×D matrix via QR decomposition."""
    M = torch.randn(D, D, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(M)
    return Q


def _compute_pca_rotation(keys: torch.Tensor) -> torch.Tensor:
    """Per-head PCA rotation matrix from key covariance.

    Args:
        keys: (H, N, D)

    Returns:
        V: (H, D, D) — columns are principal components (sorted by variance).
    """
    H, N, D = keys.shape
    # Center
    mean = keys.mean(dim=1, keepdim=True)
    centered = keys - mean
    # Covariance: (H, D, D)
    cov = torch.bmm(centered.transpose(1, 2), centered) / max(1, N - 1)
    # Eigendecomposition (ascending eigenvalues)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # Flip to descending variance
    V = eigenvectors.flip(-1)
    return V  # (H, D, D)


SUBSPACE_STRATEGIES = ("contiguous", "interleaved", "random", "pca")


def _project_keys(keys: torch.Tensor, strategy: str, S: int):
    """Project keys and return (projected_keys, dim_slices, projection, permutation).

    Returns:
        keys_proj: (H, N, D) — rotated/permuted keys.
        dim_slices: list of (start, end) tuples for contiguous subspace slicing
                    on keys_proj.
        proj: (H, D, D) or None — rotation matrix (random/pca only).
        perm: (D,) int64 or None — dimension permutation (interleaved only).
    """
    H, N, D = keys.shape
    device = keys.device
    dtype = keys.dtype
    slices = _split_contiguous(D, S)

    if strategy == "contiguous":
        return keys, slices, None, None

    if strategy == "interleaved":
        groups = _split_interleaved(D, S)
        perm = []
        new_slices = []
        offset = 0
        for g in groups:
            perm.extend(g)
            new_slices.append((offset, offset + len(g)))
            offset += len(g)
        perm_t = torch.tensor(perm, device=device, dtype=torch.long)
        keys_proj = keys[:, :, perm_t].contiguous()
        return keys_proj, new_slices, None, perm_t

    if strategy == "random":
        Q = _make_random_orthogonal(D, device, dtype)  # (D, D)
        Q_batch = Q.unsqueeze(0).expand(H, -1, -1)     # (H, D, D)
        keys_proj = torch.bmm(keys, Q_batch)            # (H, N, D)
        return keys_proj, slices, Q_batch, None

    if strategy == "pca":
        V = _compute_pca_rotation(keys)                 # (H, D, D)
        keys_proj = torch.bmm(keys, V)                  # (H, N, D)
        return keys_proj, slices, V, None

    raise ValueError(f"Unknown subspace strategy: {strategy!r}. "
                     f"Choose from {SUBSPACE_STRATEGIES}")


def build_subspace_kcenter(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int = 4,
    refine_iter: int = 5,
    strategy: str = "contiguous",
) -> SubspaceKCenterIndex:
    """Build independent k-center + ball_centroid indexes per subspace.

    Args:
        keys: (H, N, D) full-dimensional key vectors.
        bf: branching factor (cluster size target).
        n_subspaces: number of disjoint dimension slices.
        refine_iter: Lloyd refinement iterations per subspace.
        strategy: how to split dimensions — one of
                  "contiguous", "interleaved", "random", "pca".

    Returns:
        SubspaceKCenterIndex with S independent indexes.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))

    keys_proj, dim_slices, proj, perm = _project_keys(keys, strategy, n_subspaces)

    idx = SubspaceKCenterIndex(
        n_subspaces=n_subspaces, K=K, strategy=strategy,
        projection=proj, permutation=perm,
    )
    idx.dim_slices = dim_slices

    for s in range(n_subspaces):
        start, end = dim_slices[s]
        keys_sub = keys_proj[:, :, start:end].contiguous()
        assign_s, centers_s = _kcenter(keys_sub, K, refine_iter=refine_iter)
        radii_s = _ball_centroid(keys_sub, assign_s, centers_s, K)

        idx.assigns.append(assign_s)
        idx.centers.append(centers_s)
        idx.radii.append(radii_s)

    return idx


# ── Gate function ──────────────────────────────────────────────────────


def _expand_to_q(tensor: torch.Tensor, q_head_to_kv: torch.Tensor | None):
    """Expand an (H_kv, ...) tensor to (H_q, ...) using the GQA mapping."""
    if q_head_to_kv is None:
        return tensor
    return tensor[q_head_to_kv]


def _project_query(q: torch.Tensor, idx: SubspaceKCenterIndex, q_head_to_kv: torch.Tensor | None):
    """Apply the same projection used at build time to a query vector.

    Args:
        q: (H_q, D) query vectors.
        idx: the subspace index (carries projection/permutation if needed).
        q_head_to_kv: GQA mapping or None.

    Returns:
        q_proj: (H_q, D) projected query.
    """
    if idx.permutation is not None:
        return q[:, idx.permutation]
    if idx.projection is not None:
        # projection is (H_kv, D, D); expand to H_q if GQA
        proj = _expand_to_q(idx.projection, q_head_to_kv)  # (H_q, D, D)
        return torch.bmm(q.unsqueeze(1), proj).squeeze(1)   # (H_q, D)
    return q


def project_query_for_index(
    q: torch.Tensor,
    idx: SubspaceKCenterIndex,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Public wrapper to project queries into the index coordinate system."""
    return _project_query(q, idx, q_head_to_kv)


def project_keys_for_index(
    keys: torch.Tensor,
    idx: SubspaceKCenterIndex,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the same index-space transform used at build time to key vectors.

    Args:
        keys: (H_kv, N, D) key vectors in original space.
        idx: the subspace index (carries projection/permutation if needed).
        q_head_to_kv: optional GQA mapping to expand KV heads to query heads.

    Returns:
        keys_proj: (H_eval, N, D) keys in the same coordinate system used by
            the subspace index and query projection.
    """
    keys_eval = _expand_to_q(keys, q_head_to_kv)
    if idx.permutation is not None:
        return keys_eval[:, :, idx.permutation]
    if idx.projection is not None:
        proj = _expand_to_q(idx.projection, q_head_to_kv)
        return torch.bmm(keys_eval, proj)
    return keys_eval


def subspace_ball_gate(
    idx: SubspaceKCenterIndex,
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
):
    """AND-filter gate across subspaces using per-subspace thresholds.

    Args:
        q: (H_q, D) unit-norm query vectors (original space).
        th_per_subspace: (S, H_q) thresholds, one row per subspace.
        q_head_to_kv: (H_q,) int64 mapping from query heads to KV heads,
                      or None if H_q == H_kv.

    Returns:
        survive: (H_q, N) bool — which points survive (need scanning).
    """
    H_q = q.shape[0]
    device = q.device
    N = idx.assigns[0].shape[1]

    q_proj = _project_query(q, idx, q_head_to_kv)

    survive = torch.ones(H_q, N, dtype=torch.bool, device=device)

    for s in range(idx.n_subspaces):
        start, end = idx.dim_slices[s]
        q_sub = q_proj[:, start:end]                                   # (H_q, d_s)
        q_sub_norm = q_sub.norm(dim=-1)                                # (H_q,)
        centers_s = _expand_to_q(idx.centers[s], q_head_to_kv)        # (H_q, K, d_s)
        radii_s = _expand_to_q(idx.radii[s], q_head_to_kv)            # (H_q, K)
        assign_s = _expand_to_q(idx.assigns[s], q_head_to_kv)         # (H_q, N)

        # Per-cluster: center_dot + radius * ||q_s||
        center_dots = torch.einsum("hkd,hd->hk", centers_s, q_sub)    # (H_q, K)
        cluster_ub = center_dots + radii_s * q_sub_norm.unsqueeze(-1)  # (H_q, K)

        cluster_pass = cluster_ub >= th_per_subspace[s].unsqueeze(-1)  # (H_q, K)
        point_pass = cluster_pass.gather(1, assign_s)                  # (H_q, N)
        survive &= point_pass

    return survive


def subspace_cluster_gate(
    idx: SubspaceKCenterIndex,
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
):
    """Per-subspace cluster pass masks (for diagnostics)."""
    q_proj = _project_query(q, idx, q_head_to_kv)
    masks = []
    for s in range(idx.n_subspaces):
        start, end = idx.dim_slices[s]
        q_sub = q_proj[:, start:end]
        q_sub_norm = q_sub.norm(dim=-1)
        centers_s = _expand_to_q(idx.centers[s], q_head_to_kv)
        radii_s = _expand_to_q(idx.radii[s], q_head_to_kv)
        center_dots = torch.einsum("hkd,hd->hk", centers_s, q_sub)
        cluster_ub = center_dots + radii_s * q_sub_norm.unsqueeze(-1)
        masks.append(cluster_ub >= th_per_subspace[s].unsqueeze(-1))
    return masks
