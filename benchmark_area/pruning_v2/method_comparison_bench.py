#!/usr/bin/env python3
"""
Compare clustering + enclosing methods for halfspace pruning.

For each (clustering_method, enclosing_method) pair, measures what fraction
of children must be scanned when using the parent-level gate to prune.

Usage:
    python method_comparison_bench.py [--bf 16] [--n-tokens 2000] [--n-queries 50]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import _capture_qkv, _q_to_kv_map

# ── Model / capture settings ──
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 15
DEVICE = "cuda"
DTYPE = torch.float32

PROMPT = (
    "Solve the following problem step by step, showing all intermediate "
    "reasoning, calculations, and verification.\n\n"
    "A research lab is designing a distributed computing cluster. They have "
    "a budget for 120 machines. Each machine can be configured as a CPU node "
    "(32 cores, 128 GB RAM, $5000), a GPU node (8 cores, 64 GB RAM, 4×A100 "
    "GPUs, $35000), or a storage node (8 cores, 256 GB RAM, 100 TB disk, "
    "$12000). The workload consists of three phases that repeat in a cycle:\n\n"
    "Phase 1 (Training): Requires at least 200 A100 GPUs running in parallel. "
    "Each training job needs 4 GPUs and 48 GB RAM. Communication overhead "
    "between nodes adds 12% latency per additional node beyond the first. "
    "Calculate the optimal GPU node count to minimize total training time for "
    "a 500-epoch run where each epoch takes 45 minutes on a single 4-GPU node.\n\n"
    "Phase 2 (Data Processing): Must process 50 PB of raw data. Each CPU core "
    "can process 2 TB/hour. Storage nodes can serve data at 20 GB/s each but "
    "need 3 replicas for fault tolerance. Calculate the minimum storage and "
    "CPU nodes needed to finish processing within 72 hours.\n\n"
    "Phase 3 (Inference): Must serve 10,000 requests/second with p99 latency "
    "under 100ms. Each GPU can handle 150 requests/second. Each CPU core can "
    "handle 8 requests/second as fallback. The system must maintain 99.99% "
    "uptime, requiring N+2 redundancy.\n\n"
    "Determine the optimal allocation of the 120 machines across all three "
    "node types. Then analyze: What happens if the budget increases by 20%? "
    "What if training data doubles? What if inference load triples? For each "
    "scenario, re-derive the full allocation from scratch, show the math, "
    "compare trade-offs, and explain your reasoning at every step. Finally, "
    "prove mathematically that your allocation is Pareto-optimal across the "
    "three phases, or explain why no single allocation can be."
)


# =====================================================================
#  CLUSTERING METHODS
#  Each returns: assign (H, N) long tensor mapping child -> parent index
#                centers (H, K, D) parent centers
# =====================================================================


def cluster_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """Standard Lloyd's k-means on GPU."""
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Random init
    perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
    centers = keys.gather(1, perm[:, :K].unsqueeze(-1).expand(-1, -1, D)).clone()

    for _ in range(max_iter):
        # Assign: squared distance via expand trick
        # (H,N,1,D) - (H,1,K,D) -> (H,N,K)
        dists = torch.cdist(keys, centers)  # (H, N, K)
        assign = dists.argmin(dim=2)  # (H, N)

        # Recompute centers
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, K, device=device)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        # Handle empty clusters
        empty = counts == 0
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    new_centers[h, ek] = keys[h, far_idx]
                    counts[h, ek] = 1

        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        centers = new_centers

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def cluster_spherical_kmeans(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """K-means on L2-normalized keys (spherical k-means)."""
    norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_normed = keys / norms
    assign, centers = cluster_kmeans(keys_normed, bf, max_iter)
    # Re-normalize centers to unit sphere
    centers = centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return assign, centers


def cluster_random_projection(keys: torch.Tensor, bf: int, n_projections: int = 8):
    """
    Cluster by hashing random projections.
    Project keys onto random directions, quantize, hash into buckets.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Use a few random projections to create hash buckets
    n_proj = min(n_projections, max(1, int(math.log2(K)) + 1))
    proj = torch.randn(D, n_proj, device=device)
    proj = proj / proj.norm(dim=0, keepdim=True)

    # Project: (H, N, n_proj)
    projected = keys @ proj  # (H, N, n_proj)

    # Quantize each projection by median -> binary hash
    medians = projected.median(dim=1, keepdim=True).values
    bits = (projected > medians).long()  # (H, N, n_proj)

    # Hash bits to bucket ids
    powers = (2 ** torch.arange(n_proj, device=device)).long()
    bucket_ids = (bits * powers.view(1, 1, -1)).sum(dim=-1)  # (H, N)

    # Map bucket ids to [0, K) via modulo, then run one round of k-means to refine
    assign = bucket_ids % K

    # Compute centers from assignment
    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    # Reassign to nearest center
    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def cluster_random_partition(keys: torch.Tensor, bf: int):
    """Random assignment baseline — no clustering at all."""
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    assign = torch.randint(0, K, (H, N), device=device)

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    return assign, centers


def cluster_pq_subspace(keys: torch.Tensor, bf: int, n_subspaces: int = 4, max_iter: int = 10):
    """
    Product-quantization-inspired: split D dims into subspaces,
    cluster each subspace independently, combine assignments.
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    sub_dim = D // n_subspaces
    remainder = D % n_subspaces

    # Cluster each subspace with small k, combine via composite key
    sub_k = max(2, int(round(K ** (1.0 / n_subspaces))))
    sub_assigns = []

    offset = 0
    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = keys[:, :, offset : offset + sd].contiguous()
        offset += sd

        # Mini k-means on subspace
        perm = torch.argsort(torch.rand(H, N, device=device), dim=1)
        sc = sub_keys.gather(1, perm[:, :sub_k].unsqueeze(-1).expand(-1, -1, sd)).clone()

        for _ in range(max_iter):
            dists = torch.cdist(sub_keys, sc)
            sa = dists.argmin(dim=2)
            new_sc = torch.zeros_like(sc)
            cnt = torch.zeros(H, sub_k, device=device)
            new_sc.scatter_add_(1, sa.unsqueeze(-1).expand(-1, -1, sd), sub_keys)
            cnt.scatter_add_(1, sa, torch.ones(H, N, device=device))
            m = cnt > 0
            new_sc[m] /= cnt[m].unsqueeze(-1)
            new_sc[~m] = sc[~m]
            sc = new_sc

        sub_assigns.append(torch.cdist(sub_keys, sc).argmin(dim=2))

    # Composite hash
    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k

    # Map to [0, K) and compute centers
    assign = composite % K

    centers = torch.zeros(H, K, D, device=device, dtype=keys.dtype)
    counts = torch.zeros(H, K, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
    counts.scatter_add_(1, assign, torch.ones(H, N, device=device))
    empty = counts == 0
    if empty.any():
        for h in range(H):
            ek = empty[h].nonzero(as_tuple=False).view(-1)
            if ek.numel() > 0:
                centers[h, ek] = keys[h, torch.randperm(N, device=device)[: ek.numel()]]
                counts[h, ek] = 1
    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def cluster_gmm_diag(keys: torch.Tensor, bf: int, max_iter: int = 20):
    """
    Diagonal-covariance GMM clustering.
    Uses EM with diagonal covariances (axis-aligned ellipsoids).
    Assignment is hard (argmax of responsibilities).
    """
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    device = keys.device

    # Initialize from k-means
    assign_init, means = cluster_kmeans(keys, bf, max_iter=5)

    # Per-cluster diagonal variance: (H, K, D)
    variances = torch.ones(H, K, D, device=device)
    # Per-cluster mixing weights: (H, K)
    weights = torch.ones(H, K, device=device) / K

    for _ in range(max_iter):
        # E-step: compute log-responsibilities
        # log p(x|k) = -0.5 * sum_d [ (x_d - mu_d)^2 / var_d + log(var_d) ]
        # keys: (H, N, D), means: (H, K, D), variances: (H, K, D)
        diff = keys.unsqueeze(2) - means.unsqueeze(1)  # (H, N, K, D)
        var_exp = variances.unsqueeze(1).clamp_min(1e-8)  # (H, 1, K, D)
        log_prob = -0.5 * ((diff * diff / var_exp) + var_exp.log()).sum(dim=-1)  # (H, N, K)
        log_resp = log_prob + weights.log().unsqueeze(1)  # (H, N, K)

        # Hard assignment (like k-means but with Mahalanobis distance)
        assign = log_resp.argmax(dim=2)  # (H, N)

        # M-step: recompute means, variances, weights
        idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)
        new_means = torch.zeros(H, K, D, device=device)
        new_var_sum = torch.zeros(H, K, D, device=device)
        counts = torch.zeros(H, K, device=device)

        new_means.scatter_add_(1, idx_exp, keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        # Handle empty clusters
        empty = counts == 0
        if empty.any():
            for h in range(H):
                ek = empty[h].nonzero(as_tuple=False).view(-1)
                if ek.numel() > 0:
                    far_idx = torch.randperm(N, device=device)[: ek.numel()]
                    new_means[h, ek] = keys[h, far_idx]
                    counts[h, ek] = 1

        mask = counts > 0
        new_means[mask] /= counts[mask].unsqueeze(-1)
        means = new_means

        # Variance: E[(x - mu)^2] per cluster per dimension
        centered = keys - means.gather(1, idx_exp)  # (H, N, D)
        sq = centered * centered
        new_var_sum.scatter_add_(1, idx_exp, sq)
        variances = torch.ones(H, K, D, device=device)
        variances[mask] = (new_var_sum[mask] / counts[mask].unsqueeze(-1)).clamp_min(1e-8)

        # Weights
        weights = (counts / N).clamp_min(1e-8)

    # Final hard assignment
    diff = keys.unsqueeze(2) - means.unsqueeze(1)
    var_exp = variances.unsqueeze(1).clamp_min(1e-8)
    log_prob = -0.5 * ((diff * diff / var_exp) + var_exp.log()).sum(dim=-1)
    assign = log_prob.argmax(dim=2)

    return assign, means


# =====================================================================
#  ENCLOSING METHODS
#  Each returns a "gate" callable: gate(query, threshold) -> (H,K) bool
#  query is (H, D) unit-normalized, threshold is (H,) scalar
# =====================================================================


def enclose_ball_centroid(keys, assign, centers, K, bf):
    """
    Current method: ball centered at k-means centroid,
    radius = max dist from centroid to any child.
    """
    H, N, D = keys.shape
    device = keys.device

    # Dist from each child to its assigned center
    parent_for_child = centers.gather(1, assign.unsqueeze(-1).expand(-1, -1, D))
    dists = (keys - parent_for_child).norm(dim=-1)  # (H, N)

    radii = torch.full((H, K), 0.0, device=device)
    radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", centers, q)  # (H, K)
        return (scores + radii) > th.unsqueeze(-1)

    return gate, {"radii_mean": float(radii.mean()), "radii_max": float(radii.max())}


def enclose_min_ball(keys, assign, centers, K, bf):
    """
    Minimum enclosing ball via iterative farthest-point centering.
    Fully vectorized — no Python loops over H×K.
    """
    H, N, D = keys.shape
    device = keys.device
    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)

    meb_centers = centers.clone()

    for _ in range(10):
        # Distance from each child to its cluster's MEB center
        parent_c = meb_centers.gather(1, idx_exp)  # (H, N, D)
        dists = (keys - parent_c).norm(dim=-1)  # (H, N)

        # Find farthest point per cluster: set non-max to -inf, then scatter argmax
        # Trick: add large offset per cluster so argmax across all N gives per-cluster max
        offset = assign.float() * (dists.max() + 1)  # separate clusters
        keyed = dists + offset
        # For each cluster, the point with max keyed value within that cluster
        # is the farthest point. Use scatter_reduce to get max dist per cluster.
        max_dists = torch.full((H, K), 0.0, device=device)
        max_dists.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

        # To find farthest point index: mark points that achieve the max
        cluster_max_for_child = max_dists.gather(1, assign)  # (H, N)
        is_farthest = (dists >= cluster_max_for_child - 1e-6) & (dists > 0)

        # For each cluster, pick one farthest point (first one via argmax on mask)
        # Compute direction: farthest_point - center, averaged if multiple
        direction = torch.zeros(H, K, D, device=device)
        weight = torch.zeros(H, K, device=device)
        diff = keys - parent_c  # (H, N, D)
        masked_diff = diff * is_farthest.unsqueeze(-1).float()
        direction.scatter_add_(1, idx_exp, masked_diff)
        weight.scatter_add_(1, assign, is_farthest.float())
        weight = weight.clamp_min(1)
        direction = direction / weight.unsqueeze(-1)

        # Move center toward farthest point
        meb_centers = meb_centers + 0.5 * direction

    # Final radii
    parent_c = meb_centers.gather(1, idx_exp)
    dists = (keys - parent_c).norm(dim=-1)
    meb_radii = torch.full((H, K), 0.0, device=device)
    meb_radii.scatter_reduce_(1, assign, dists, reduce="amax", include_self=True)

    def gate(q, th):
        scores = torch.einsum("hkd,hd->hk", meb_centers, q)
        return (scores + meb_radii) > th.unsqueeze(-1)

    return gate, {"radii_mean": float(meb_radii.mean()), "radii_max": float(meb_radii.max())}


def enclose_aabb(keys, assign, centers, K, bf):
    """
    Axis-aligned bounding box.
    Gate: max_{k in box} q·k = sum_d max(q_d * lo_d, q_d * hi_d)
    """
    H, N, D = keys.shape
    device = keys.device

    lo = torch.full((H, K, D), float("inf"), device=device)
    hi = torch.full((H, K, D), float("-inf"), device=device)

    idx_exp = assign.unsqueeze(-1).expand(-1, -1, D)  # (H, N, D)
    lo.scatter_reduce_(1, idx_exp, keys, reduce="amin", include_self=False)
    hi.scatter_reduce_(1, idx_exp, keys, reduce="amax", include_self=False)

    # Fix clusters with no children
    empty = lo[:, :, 0].isinf()
    if empty.any():
        lo[empty] = 0.0
        hi[empty] = 0.0

    def gate(q, th):
        # q: (H, D), lo/hi: (H, K, D)
        q_exp = q.unsqueeze(1)  # (H, 1, D)
        max_dot = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H, K)
        return max_dot > th.unsqueeze(-1)

    vol = (hi - lo).clamp_min(0).prod(dim=-1)
    return gate, {"vol_mean": float(vol.mean()), "vol_max": float(vol.max())}


def enclose_cone(keys, assign, centers, K, bf):
    """
    Cone enclosure for normalized keys.
    Direction = mean direction of children (normalized).
    Half-angle = max angle between direction and any child.
    Gate: cos(max(0, angle(q, dir) - alpha)) > threshold
    For unit q and unit dir: q·dir >= cos(alpha) implies all children could pass.
    More precisely: max_{k in cone} q·k >= cos(max(0, arccos(q·dir) - alpha))
    """
    H, N, D = keys.shape
    device = keys.device

    # Normalize keys for cone computation
    key_norms = keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_normed = keys / key_norms

    # Cone direction = normalized mean of children directions
    cone_dir = torch.zeros(H, K, D, device=device)
    cone_dir.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys_normed)
    cone_dir = cone_dir / cone_dir.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Half-angle = max angle from cone_dir to any child
    # cos_angle(child, dir) = child_normed · dir
    dir_for_child = cone_dir.gather(1, assign.unsqueeze(-1).expand(-1, -1, D))
    cos_angles = (keys_normed * dir_for_child).sum(dim=-1)  # (H, N)
    cos_angles = cos_angles.clamp(-1, 1)

    # Min cosine per cluster = max angle
    min_cos = torch.full((H, K), 1.0, device=device)
    min_cos.scatter_reduce_(1, assign, cos_angles, reduce="amin", include_self=True)
    half_angles = torch.acos(min_cos.clamp(-1, 1))  # (H, K)

    # Also need max key norm per cluster for the bound
    max_norm = torch.full((H, K), 0.0, device=device)
    key_norms_sq = key_norms.squeeze(-1)
    max_norm.scatter_reduce_(1, assign, key_norms_sq, reduce="amax", include_self=True)

    def gate(q, th):
        # q is unit-normalized (H, D)
        # q · dir: (H, K)
        q_dot_dir = torch.einsum("hkd,hd->hk", cone_dir, q)
        q_dot_dir = q_dot_dir.clamp(-1, 1)
        angle_q_dir = torch.acos(q_dot_dir)  # (H, K)

        # Effective angle: max(0, angle - half_angle)
        effective_angle = (angle_q_dir - half_angles).clamp_min(0)

        # Upper bound on q·k for any k in cone with norm <= max_norm:
        # max_dot = max_norm * cos(effective_angle)
        upper_bound = max_norm * torch.cos(effective_angle)
        return upper_bound > th.unsqueeze(-1)

    return gate, {
        "half_angle_mean_deg": float(half_angles.mean() * 180 / math.pi),
        "half_angle_max_deg": float(half_angles.max() * 180 / math.pi),
    }


# =====================================================================
#  BENCHMARK CORE
# =====================================================================


def topk_threshold(q_normal, keys, k=20):
    """Ground-truth top-k threshold over all keys."""
    H_kv, N, D = keys.shape
    qg = q_normal.view(H_kv, -1, D)
    w = qg @ keys.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(k, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def measure_scanned_fraction(gate_fn, queries, keys, q_indices, q_head_to_kv, K, bf, topk):
    """Run queries through the gate and measure scanned fraction + search time."""
    H_kv, N, D = keys.shape
    fracs = []
    search_times = []

    for qi in q_indices:
        q = queries[:, qi, :]
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm

        # Expand to query heads via q_head_to_kv
        q_kv = q_normal[q_head_to_kv] if q_head_to_kv is not None else q_normal

        th = topk_threshold(q_kv, keys, k=topk)

        # Gate: (H_q, K) bool — timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        parent_pass = gate_fn(q_kv, th)
        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)

        scanned = parent_pass.sum(dim=1).float() * bf
        frac = scanned / max(1, N)
        fracs.append(frac.mean().item())

    mean_frac = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000 if search_times else 0.0
    return mean_frac, mean_search_ms


# =====================================================================
#  MAIN
# =====================================================================

CLUSTERING_METHODS = {
    "kmeans": cluster_kmeans,
    "spherical_kmeans": cluster_spherical_kmeans,
    "random_proj": cluster_random_projection,
    "random_partition": cluster_random_partition,
    "pq_subspace": cluster_pq_subspace,
    "gmm_diag": cluster_gmm_diag,
}

ENCLOSING_METHODS = {
    "ball_centroid": enclose_ball_centroid,
    "min_enclosing_ball": enclose_min_ball,
    "aabb": enclose_aabb,
    "cone": enclose_cone,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-tokens", type=int, default=2000, help="Tokens to capture")
    parser.add_argument("--n-queries", type=int, default=30, help="Number of queries to evaluate")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for threshold")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
    t0 = time.perf_counter()
    capture = _capture_qkv(
        model_name=args.model,
        prompt_text=PROMPT,
        n=args.n_tokens,
        device=DEVICE,
        torch_dtype=DTYPE,
        show_progress=True,
    )
    print(f"Capture done in {time.perf_counter() - t0:.1f}s\n")

    layer_ids = capture.layer_ids()
    layer = LAYER_IDX if LAYER_IDX in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)
    H_kv, N, D = keys.shape
    H_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(H_q, H_kv, DEVICE) if H_q != H_kv else None
    K = max(1, math.ceil(N / args.bf))

    # Query indices: sample from end of sequence
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))
    q_indices = q_indices[: args.n_queries]

    print(f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}")
    print(f"K={K} parents (bf={args.bf}), {len(q_indices)} queries, topk={args.topk}")
    print("=" * 90)

    results = []

    for clust_name, clust_fn in CLUSTERING_METHODS.items():
        print(f"\nClustering: {clust_name} ...")
        t0 = time.perf_counter()
        if clust_name == "random_partition":
            assign, centers = clust_fn(keys, args.bf)
        elif clust_name == "pq_subspace":
            assign, centers = clust_fn(keys, args.bf)
        else:
            assign, centers = clust_fn(keys, args.bf)
        clust_time = time.perf_counter() - t0

        # Expand centers/assign to query heads if needed
        if q_head_to_kv is not None:
            assign_q = assign[q_head_to_kv]
            centers_q = centers[q_head_to_kv]
            keys_q = keys[q_head_to_kv]
        else:
            assign_q = assign
            centers_q = centers
            keys_q = keys

        for enc_name, enc_fn in ENCLOSING_METHODS.items():
            t1 = time.perf_counter()
            gate_fn, enc_info = enc_fn(keys_q, assign_q, centers_q, K, args.bf)
            enc_time = time.perf_counter() - t1

            frac, search_ms = measure_scanned_fraction(
                gate_fn, queries, keys_q, q_indices, None, K, args.bf, args.topk
            )

            results.append({
                "clustering": clust_name,
                "enclosing": enc_name,
                "scanned_frac": frac,
                "clust_ms": clust_time * 1000,
                "enc_ms": enc_time * 1000,
                "search_ms": search_ms,
                **{f"enc_{k}": v for k, v in enc_info.items()},
            })

            pruning = 1.0 - frac
            print(
                f"  {enc_name:<20s}  scanned={frac:.4f}  pruned={pruning:.4f}  "
                f"search={search_ms:.3f}ms  "
                f"clust={clust_time*1000:.1f}ms  enc={enc_time*1000:.1f}ms  "
                + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in enc_info.items())
            )

    # ── Summary table ──
    print("\n" + "=" * 105)
    print(
        f"{'CLUSTERING':<22s} {'ENCLOSING':<22s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s}"
    )
    print("-" * 105)

    results.sort(key=lambda r: r["scanned_frac"])
    for r in results:
        build_ms = r["clust_ms"] + r["enc_ms"]
        pruned = 1.0 - r["scanned_frac"]
        print(
            f"{r['clustering']:<22s} {r['enclosing']:<22s} "
            f"{r['scanned_frac']:>8.4f} {pruned:>8.4f} "
            f"{r['search_ms']:>10.3f} {build_ms:>9.1f}"
        )

    print("=" * 105)
    best = results[0]
    print(
        f"\nBest: {best['clustering']} + {best['enclosing']} "
        f"-> scanned={best['scanned_frac']:.4f} (pruned {1-best['scanned_frac']:.4f})"
    )


if __name__ == "__main__":
    main()