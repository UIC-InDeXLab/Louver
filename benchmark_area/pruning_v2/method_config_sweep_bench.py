#!/usr/bin/env python3
"""
Sweep clustering configs for halfspace pruning gates.

This benchmark is similar to method_comparison_bench.py, but focuses on
configuration sweeps (especially pq_subspace) and adds a hybrid method:
    pq_subspace -> k-means refinement

Usage:
    python method_config_sweep_bench.py
    python method_config_sweep_bench.py --extended --enclosing full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    import faiss
except Exception:
    faiss = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import _capture_qkv, _q_to_kv_map

from method_comparison_bench import (
    DEVICE,
    DTYPE,
    LAYER_IDX,
    MODEL_NAME,
    PROMPT,
    cluster_gmm_diag,
    cluster_kmeans,
    cluster_pq_subspace,
    cluster_random_projection,
    cluster_spherical_kmeans,
    enclose_aabb,
    enclose_ball_centroid,
    enclose_cone,
    enclose_min_ball,
    measure_scanned_fraction,
)


@dataclass(frozen=True)
class ClusteringVariant:
    family: str
    label: str
    fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    params: dict[str, int | float]


def _kmeans_refine_from_init(
    keys: torch.Tensor,
    init_centers: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Lloyd refinement starting from caller-provided centers."""
    H, N, D = keys.shape
    centers = init_centers.clone()
    device = keys.device

    if max_iter <= 0:
        assign = torch.cdist(keys, centers).argmin(dim=2)
        return assign, centers

    for _ in range(max_iter):
        assign = torch.cdist(keys, centers).argmin(dim=2)

        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(H, centers.shape[1], device=device)
        new_centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), keys)
        counts.scatter_add_(1, assign, torch.ones(H, N, device=device))

        empty = counts == 0
        if empty.any():
            for h in range(H):
                empty_idx = empty[h].nonzero(as_tuple=False).view(-1)
                if empty_idx.numel() > 0:
                    repl_idx = torch.randperm(N, device=device)[: empty_idx.numel()]
                    new_centers[h, empty_idx] = keys[h, repl_idx]
                    counts[h, empty_idx] = 1

        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        centers = new_centers

    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def cluster_pq_then_kmeans_refine(
    keys: torch.Tensor,
    bf: int,
    n_subspaces: int = 4,
    pq_max_iter: int = 10,
    refine_iters: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hybrid: initialize with pq_subspace, then run extra k-means refinement."""
    _, centers = cluster_pq_subspace(
        keys,
        bf,
        n_subspaces=n_subspaces,
        max_iter=pq_max_iter,
    )
    return _kmeans_refine_from_init(keys, centers, max_iter=refine_iters)


def cluster_pq_subspace_faiss(
    keys: torch.Tensor, bf: int, n_subspaces: int = 4, max_iter: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FAISS-backed variant of pq_subspace:
    split dimensions into subspaces, run k-means per subspace with FAISS,
    combine subspace assignments, then compute full-space centers.
    """
    if faiss is None:
        raise RuntimeError(
            "pq_subspace_faiss requested but faiss is not installed in this environment."
        )

    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    device = keys.device

    sub_dim = d // n_subspaces
    remainder = d % n_subspaces
    sub_k = max(2, int(round(k ** (1.0 / n_subspaces))))
    sub_k = min(sub_k, n)

    sub_assigns: list[torch.Tensor] = []
    offset = 0
    base_seed = int(torch.initial_seed() % (2**31 - 1))

    for s in range(n_subspaces):
        sd = sub_dim + (1 if s < remainder else 0)
        sub_keys = keys[:, :, offset : offset + sd].contiguous()
        offset += sd

        sub_assign = torch.empty(h, n, dtype=torch.long, device=device)
        for hi in range(h):
            x_np = (
                sub_keys[hi].detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            if not x_np.flags["C_CONTIGUOUS"]:
                x_np = np.ascontiguousarray(x_np)

            km_seed = int((base_seed + 1009 * s + 7919 * hi) % (2**31 - 1))
            kmeans = faiss.Kmeans(
                d=sd,
                k=sub_k,
                niter=max_iter,
                nredo=1,
                seed=km_seed,
                verbose=False,
                min_points_per_centroid=1,
            )
            kmeans.train(x_np)
            _, idx = kmeans.index.search(x_np, 1)
            sub_assign[hi] = torch.from_numpy(idx[:, 0]).to(
                device=device, dtype=torch.long
            )

        sub_assigns.append(sub_assign)

    composite = sub_assigns[0]
    multiplier = sub_k
    for sa in sub_assigns[1:]:
        composite = composite * multiplier + sa
        multiplier *= sub_k

    assign = composite % k
    centers = _centers_from_assign_original(keys, assign, k=k)
    assign = torch.cdist(keys, centers).argmin(dim=2)
    return assign, centers


def _kmeans_on_features(
    features: torch.Tensor,
    k: int,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    K-means on arbitrary features.
    features: (H, N, F) -> assign (H, N), centers (H, K, F)
    """
    h, n, f = features.shape
    device = features.device

    perm = torch.argsort(torch.rand(h, n, device=device), dim=1)
    centers = features.gather(
        1, perm[:, :k].unsqueeze(-1).expand(-1, -1, f)
    ).clone()

    for _ in range(max_iter):
        assign = torch.cdist(features, centers).argmin(dim=2)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(h, k, device=device)
        new_centers.scatter_add_(
            1, assign.unsqueeze(-1).expand(-1, -1, f), features
        )
        counts.scatter_add_(1, assign, torch.ones(h, n, device=device))

        empty = counts == 0
        if empty.any():
            for hi in range(h):
                empty_idx = empty[hi].nonzero(as_tuple=False).view(-1)
                if empty_idx.numel() > 0:
                    repl = torch.randperm(n, device=device)[: empty_idx.numel()]
                    new_centers[hi, empty_idx] = features[hi, repl]
                    counts[hi, empty_idx] = 1

        mask = counts > 0
        new_centers[mask] /= counts[mask].unsqueeze(-1)
        centers = new_centers

    assign = torch.cdist(features, centers).argmin(dim=2)
    return assign, centers


def _centers_from_assign_original(
    keys: torch.Tensor, assign: torch.Tensor, k: int
) -> torch.Tensor:
    """Compute centers in original key space from assignments."""
    h, n, d = keys.shape
    device = keys.device
    centers = torch.zeros(h, k, d, device=device, dtype=keys.dtype)
    counts = torch.zeros(h, k, device=device)
    centers.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, d), keys)
    counts.scatter_add_(1, assign, torch.ones(h, n, device=device))

    empty = counts == 0
    if empty.any():
        for hi in range(h):
            empty_idx = empty[hi].nonzero(as_tuple=False).view(-1)
            if empty_idx.numel() > 0:
                repl = torch.randperm(n, device=device)[: empty_idx.numel()]
                centers[hi, empty_idx] = keys[hi, repl]
                counts[hi, empty_idx] = 1

    mask = counts > 0
    centers[mask] /= counts[mask].unsqueeze(-1)
    return centers


def _pca_project_topm(keys: torch.Tensor, m_dims: int) -> torch.Tensor:
    """
    Per-head PCA projection to top-m principal dimensions.
    Returns projected features: (H, N, m_eff).
    """
    h, n, d = keys.shape
    m_eff = max(1, min(int(m_dims), d))
    out = torch.empty(h, n, m_eff, device=keys.device, dtype=keys.dtype)

    for hi in range(h):
        x = keys[hi]
        x_centered = x - x.mean(dim=0, keepdim=True)
        # Vh: (D, D), principal directions are first rows of Vh.
        _, _, vh = torch.linalg.svd(x_centered, full_matrices=False)
        proj = x_centered @ vh[:m_eff].transpose(0, 1)
        out[hi] = proj

    return out


def _pca_ranked_dim_indices(keys: torch.Tensor) -> torch.Tensor:
    """
    Per-head PCA-based dimension importance ranking.
    Importance_j = sum_k lambda_k * v_{j,k}^2 (weighted squared loadings).
    Returns sorted indices: (H, D), descending importance per head.
    """
    h, n, d = keys.shape
    ranks = torch.empty(h, d, device=keys.device, dtype=torch.long)

    for hi in range(h):
        x = keys[hi]
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
        eig = (s * s) / max(1, (n - 1))
        v = vh.transpose(0, 1)  # (D, D)
        importance = (v * v * eig.unsqueeze(0)).sum(dim=1)  # (D,)
        ranks[hi] = torch.argsort(importance, descending=True)

    return ranks


def cluster_pca_kmeans(
    keys: torch.Tensor,
    bf: int,
    m_dims: int = 32,
    max_iter: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PCA + k-means:
      1) Per-head PCA projection to top-m_dims
      2) K-means in projected space
      3) Return assignments + original-space centers
    """
    _, n, _ = keys.shape
    k = max(1, math.ceil(n / bf))
    projected = _pca_project_topm(keys, m_dims=m_dims)
    assign, _ = _kmeans_on_features(projected, k=k, max_iter=max_iter)
    centers = _centers_from_assign_original(keys, assign, k=k)
    return assign, centers


def cluster_pca_ranked_pq_subspace(
    keys: torch.Tensor,
    bf: int,
    m_fraction: float = 0.5,
    n_subspaces: int = 4,
    max_iter: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PCA-ranked-dimension pq_subspace:
      1) Rank original dimensions per head by PCA-based importance.
      2) Keep top ceil(m_fraction * D) dimensions.
      3) Run pq_subspace on reduced keys.
      4) Return assignments + original-space centers.
    """
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    keep = max(1, min(d, int(math.ceil(float(m_fraction) * d))))

    ranked = _pca_ranked_dim_indices(keys)  # (H, D)
    top_idx = ranked[:, :keep]  # (H, keep)
    gather_idx = top_idx.unsqueeze(1).expand(-1, n, -1)
    reduced = keys.gather(2, gather_idx)

    assign, _ = cluster_pq_subspace(
        reduced, bf, n_subspaces=n_subspaces, max_iter=max_iter
    )
    centers = _centers_from_assign_original(keys, assign, k=k)
    return assign, centers


def _variant_seed(base_seed: int, label: str) -> int:
    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=4).digest()
    mix = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(base_seed) + mix) % (2**31 - 1)


def _make_variants(extended: bool) -> list[ClusteringVariant]:
    variants: list[ClusteringVariant] = []

    # Primary focus: pq_subspace grid.
    pq_subspaces = [2, 4, 8]
    pq_iters = [5, 10]
    if extended:
        pq_subspaces = [1, 2, 4, 6, 8, 12]
        pq_iters = [2, 5, 10, 15, 20]

    for n_subspaces in pq_subspaces:
        for pq_iter in pq_iters:
            params = {"n_subspaces": n_subspaces, "max_iter": pq_iter}
            label = f"pq_subspace(ns={n_subspaces},it={pq_iter})"
            variants.append(
                ClusteringVariant(
                    family="pq_subspace",
                    label=label,
                    fn=cluster_pq_subspace,
                    params=params,
                )
            )
            if faiss is not None:
                variants.append(
                    ClusteringVariant(
                        family="pq_subspace_faiss",
                        label=f"pq_subspace_faiss(ns={n_subspaces},it={pq_iter})",
                        fn=cluster_pq_subspace_faiss,
                        params=params,
                    )
                )

    # Hybrid: pq_subspace then k-means refinement.
    hybrid_subspaces = [2, 4, 8]
    hybrid_pq_iters = [5, 10]
    refine_iters = [5, 10]
    if extended:
        hybrid_subspaces = [1, 2, 4, 6, 8, 12]
        hybrid_pq_iters = [2, 5, 10]
        refine_iters = [1, 2, 4, 8, 12, 20]

    for n_subspaces in hybrid_subspaces:
        for pq_iter in hybrid_pq_iters:
            for refine in refine_iters:
                params = {
                    "n_subspaces": n_subspaces,
                    "pq_max_iter": pq_iter,
                    "refine_iters": refine,
                }
                label = (
                    "pq_kmeans_hybrid"
                    f"(ns={n_subspaces},pq_it={pq_iter},km_refine={refine})"
                )
                variants.append(
                    ClusteringVariant(
                        family="pq_kmeans_hybrid",
                        label=label,
                        fn=cluster_pq_then_kmeans_refine,
                        params=params,
                    )
                )

    # Subset sweeps for other configurable clustering methods.
    kmeans_iters = [10, 20] + ([5, 40, 60] if extended else [])
    spherical_iters = [10, 20] + ([5, 40] if extended else [])
    random_proj_counts = [4, 8] + ([2, 12, 16] if extended else [])
    gmm_iters = [10] + ([5, 20, 30] if extended else [])

    for it in kmeans_iters:
        variants.append(
            ClusteringVariant(
                family="kmeans",
                label=f"kmeans(it={it})",
                fn=cluster_kmeans,
                params={"max_iter": it},
            )
        )

    for it in spherical_iters:
        variants.append(
            ClusteringVariant(
                family="spherical_kmeans",
                label=f"spherical_kmeans(it={it})",
                fn=cluster_spherical_kmeans,
                params={"max_iter": it},
            )
        )

    for n_proj in random_proj_counts:
        variants.append(
            ClusteringVariant(
                family="random_proj",
                label=f"random_proj(n_proj={n_proj})",
                fn=cluster_random_projection,
                params={"n_projections": n_proj},
            )
        )

    for it in gmm_iters:
        variants.append(
            ClusteringVariant(
                family="gmm_diag",
                label=f"gmm_diag(it={it})",
                fn=cluster_gmm_diag,
                params={"max_iter": it},
            )
        )

    # PCA + kmeans on top-m dimensions.
    pca_m_dims = [16, 32, 64, 96]
    pca_kmeans_iters = [10, 20]
    if extended:
        pca_m_dims = [8, 16, 24, 32, 48, 64, 96, 120]
        pca_kmeans_iters = [10, 20, 40]

    for m_dims in pca_m_dims:
        for it in pca_kmeans_iters:
            variants.append(
                ClusteringVariant(
                    family="pca_kmeans",
                    label=f"pca_kmeans(m={m_dims},it={it})",
                    fn=cluster_pca_kmeans,
                    params={"m_dims": m_dims, "max_iter": it},
                )
            )

    # PCA-ranked dimensions + pq_subspace on top-m fraction.
    pca_frac = [0.25, 0.5, 0.75]
    pca_rank_ns = [2, 4]
    pca_rank_iters = [5, 10]
    if extended:
        pca_frac = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
        pca_rank_ns = [2, 4, 8]
        pca_rank_iters = [5, 10, 15]

    for frac in pca_frac:
        for ns in pca_rank_ns:
            for it in pca_rank_iters:
                variants.append(
                    ClusteringVariant(
                        family="pca_ranked_pq_subspace",
                        label=f"pca_ranked_pq(mfrac={frac:.3f},ns={ns},it={it})",
                        fn=cluster_pca_ranked_pq_subspace,
                        params={
                            "m_fraction": frac,
                            "n_subspaces": ns,
                            "max_iter": it,
                        },
                    )
                )

    return variants


def _make_enclosing_methods(kind: str):
    quick = {
        "ball_centroid": enclose_ball_centroid,
        "aabb": enclose_aabb,
    }
    full = {
        "ball_centroid": enclose_ball_centroid,
        "min_enclosing_ball": enclose_min_ball,
        "aabb": enclose_aabb,
        "cone": enclose_cone,
    }
    return quick if kind == "quick" else full


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return

    cols = sorted({k for row in rows for k in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _print_ranked_summary(
    rows: list[dict[str, object]], enclosing_name: str, top_n: int | None = None
) -> None:
    target = [r for r in rows if r["enclosing"] == enclosing_name]
    target.sort(key=lambda r: float(r["scanned_frac"]))
    if top_n is not None and top_n > 0:
        target = target[:top_n]

    print("\n" + "=" * 120)
    print(f"Enclosing: {enclosing_name}")
    print(
        f"{'RANK':<6s} {'CLUSTERING':<58s} {'SCANNED':>8s} {'PRUNED':>8s} "
        f"{'SEARCH_ms':>10s} {'BUILD_ms':>9s} {'FAMILY':<18s}"
    )
    print("-" * 120)

    for i, row in enumerate(target[:20], start=1):
        scanned = float(row["scanned_frac"])
        pruned = 1.0 - scanned
        build_ms = float(row["clust_ms"]) + float(row["enc_ms"])
        print(
            f"{i:<6d} {str(row['clustering']):<58s} {scanned:>8.4f} {pruned:>8.4f} "
            f"{float(row['search_ms']):>10.3f} {build_ms:>9.1f} {str(row['family']):<18s}"
        )

    best = min(
        (r for r in rows if r["enclosing"] == enclosing_name),
        key=lambda r: float(r["scanned_frac"]),
    )
    print("=" * 120)
    print(
        f"Best ({enclosing_name}): {best['clustering']} "
        f"-> scanned={float(best['scanned_frac']):.4f}, "
        f"pruned={1 - float(best['scanned_frac']):.4f}"
    )


def _print_all_enclosing_summaries(
    rows: list[dict[str, object]],
    enclosing_names: list[str],
    top_n: int | None = None,
) -> None:
    for enc_name in enclosing_names:
        _print_ranked_summary(rows, enclosing_name=enc_name, top_n=top_n)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-tokens", type=int, default=2000, help="Tokens to capture")
    parser.add_argument("--n-queries", type=int, default=30, help="Number of queries to evaluate")
    parser.add_argument("--topk", type=int, default=20, help="Top-k for threshold")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--layer", type=int, default=LAYER_IDX, help="Layer index to evaluate")
    parser.add_argument(
        "--enclosing",
        choices=["quick", "full"],
        default="full",
        help="Enclosing method set to evaluate.",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Use a larger config grid for deeper sweeps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("method_config_sweep_results.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--top-per-enclosing",
        type=int,
        default=0,
        help="Rows to print per enclosing summary (0 = all clusterings).",
    )
    parser.add_argument(
        "--families",
        type=str,
        default="all",
        help='Comma-separated clustering families to keep (e.g. "pq_subspace,pq_kmeans_hybrid"), or "all".',
    )
    parser.add_argument(
        "--enclosing-only",
        type=str,
        default="all",
        help='Comma-separated enclosing methods to keep (e.g. "aabb"), or "all".',
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base random seed")
    return parser.parse_args()


def main():
    args = parse_args()

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
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    keys = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)
    H_kv, N, D = keys.shape
    H_q = queries.shape[0]

    q_head_to_kv = _q_to_kv_map(H_q, H_kv, DEVICE) if H_q != H_kv else None
    K = max(1, math.ceil(N / args.bf))

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )
    q_indices = q_indices[: args.n_queries]

    variants = _make_variants(extended=args.extended)
    if args.families != "all":
        allowed_families = {x.strip() for x in args.families.split(",") if x.strip()}
        variants = [v for v in variants if v.family in allowed_families]
        if not variants:
            raise ValueError(f"No variants matched --families={args.families!r}")

    enclosing_methods = _make_enclosing_methods(args.enclosing)
    if args.enclosing_only != "all":
        allowed_enclosing = {
            x.strip() for x in args.enclosing_only.split(",") if x.strip()
        }
        enclosing_methods = {
            k: v for k, v in enclosing_methods.items() if k in allowed_enclosing
        }
        if not enclosing_methods:
            raise ValueError(
                f"No enclosing methods matched --enclosing-only={args.enclosing_only!r}"
            )

    print(
        f"Layer {layer}: H_kv={H_kv}, H_q={H_q}, N={N}, D={D}, "
        f"K={K}, queries={len(q_indices)}, topk={args.topk}"
    )
    print(
        f"Variants={len(variants)} ({'extended' if args.extended else 'base'}), "
        f"enclosing={list(enclosing_methods.keys())}"
    )
    print("=" * 120)

    rows: list[dict[str, object]] = []

    for variant in tqdm(variants, desc="Config sweep"):
        seed = _variant_seed(args.seed, variant.label)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t_cluster = time.perf_counter()
        assign, centers = variant.fn(keys, args.bf, **variant.params)
        clust_ms = (time.perf_counter() - t_cluster) * 1000.0

        if q_head_to_kv is not None:
            assign_q = assign[q_head_to_kv]
            centers_q = centers[q_head_to_kv]
            keys_q = keys[q_head_to_kv]
        else:
            assign_q = assign
            centers_q = centers
            keys_q = keys

        variant_best = float("inf")
        variant_parts = []

        for enc_name, enc_fn in enclosing_methods.items():
            t_enclose = time.perf_counter()
            gate_fn, enc_info = enc_fn(keys_q, assign_q, centers_q, K, args.bf)
            enc_ms = (time.perf_counter() - t_enclose) * 1000.0

            scanned_frac, search_ms = measure_scanned_fraction(
                gate_fn=gate_fn,
                queries=queries,
                keys=keys_q,
                q_indices=q_indices,
                q_head_to_kv=None,
                K=K,
                bf=args.bf,
                topk=args.topk,
            )

            row = {
                "layer": layer,
                "bf": args.bf,
                "n_tokens": args.n_tokens,
                "n_queries": len(q_indices),
                "topk": args.topk,
                "family": variant.family,
                "clustering": variant.label,
                "params": str(variant.params),
                "enclosing": enc_name,
                "scanned_frac": float(scanned_frac),
                "pruned_frac": float(1.0 - scanned_frac),
                "clust_ms": float(clust_ms),
                "enc_ms": float(enc_ms),
                "search_ms": float(search_ms),
                "seed": int(seed),
            }
            row.update({f"enc_{k}": v for k, v in enc_info.items()})
            rows.append(row)
            variant_best = min(variant_best, float(scanned_frac))
            variant_parts.append(f"{enc_name}:{float(scanned_frac):.4f}")

        print(
            f"{variant.label:<58s} clust={clust_ms:>8.1f}ms "
            f"best_scanned={variant_best:.4f}  "
            + "  ".join(variant_parts)
        )

    _write_csv(rows, args.output)

    top_n = args.top_per_enclosing if args.top_per_enclosing > 0 else None
    _print_all_enclosing_summaries(
        rows=rows,
        enclosing_names=list(enclosing_methods.keys()),
        top_n=top_n,
    )
    print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()
