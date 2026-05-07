#!/usr/bin/env python3
"""
Adaptive flat-index BF refinement benchmark for AABB pruning.

Structure (always two-level):
  parents: flat list of K AABB clusters  →  children: the raw keys inside each cluster

Method:
1) Build root: cluster all N keys with BF=bf_start → K parent AABBs.
2) Search: for each query, classify every parent AABB:
   - outside:      prune (skip)
   - inside:       count all keys in cluster
   - intersecting: scan all keys + queue cluster for refinement
3) Every refine_window_size queries, replace each queued cluster with its own
   tighter sub-clusters at BF=BF//2 (no cross-cluster pooling, no tree).
4) Clusters at bf_min are leaves — never split further.

Search timing excludes refinement.
As intersecting clusters are progressively replaced by finer ones, pruning power
approaches that of a global BF=bf_min flat index.
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

import torch
from tqdm.auto import tqdm

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
    cluster_pq_subspace,
)


@dataclass(frozen=True)
class AdaptiveConfig:
    label: str
    n_subspaces: int
    pq_iter: int
    bf_start: int
    bf_min: int
    refine_window_size: int
    query_log_every: int


@dataclass
class FlatIndex:
    """
    Flat two-level index for one attention head.
    lo/hi/sizes are kept as tensors for batched AABB queries.
    key_idx and cluster_bf are Python lists (one entry per cluster).
    """
    lo: torch.Tensor             # (K, d)
    hi: torch.Tensor             # (K, d)
    sizes: torch.Tensor          # (K,)  number of keys per cluster
    key_idx: list[torch.Tensor]  # K tensors of global key indices
    cluster_bf: list[int]        # bf used when each cluster was created


def _parse_int_list(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def _topk_threshold(q_normal: torch.Tensor, keys: torch.Tensor, topk: int) -> torch.Tensor:
    """Ground-truth top-k threshold over all keys per head."""
    h, _, d = keys.shape
    qg = q_normal.view(h, -1, d)
    w = qg @ keys.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(topk, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def _cluster(
    keys_h: torch.Tensor,   # (N, d)
    key_idx: torch.Tensor,  # (n_local,)
    bf: int,
    n_subspaces: int,
    pq_iter: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Cluster key_idx into up to bf groups.
    Returns (lo_list, hi_list, key_idx_list) — one entry per non-empty cluster.
    """
    n_local = int(key_idx.numel())

    if n_local == 1:
        pt = keys_h.index_select(0, key_idx).squeeze(0)   # (d,)
        return [pt], [pt], [key_idx]

    local_keys = keys_h.index_select(0, key_idx)           # (n_local, d)
    assign, _ = cluster_pq_subspace(
        local_keys.unsqueeze(0), bf=bf, n_subspaces=n_subspaces, max_iter=pq_iter
    )
    assign = assign[0]
    k_nominal = max(1, math.ceil(n_local / max(1, bf)))

    lo_list, hi_list, kidx_list = [], [], []
    for cid in range(k_nominal):
        mask = assign == cid
        if not bool(mask.any()):
            continue
        c_idx = key_idx[mask]
        c_keys = local_keys[mask]
        kidx_list.append(c_idx)
        lo_list.append(c_keys.min(0).values)
        hi_list.append(c_keys.max(0).values)

    if not kidx_list:                           # fallback
        kidx_list = [key_idx]
        lo_list   = [local_keys.min(0).values]
        hi_list   = [local_keys.max(0).values]

    return lo_list, hi_list, kidx_list


def _build_flat(
    keys_h: torch.Tensor,
    key_idx: torch.Tensor,
    bf: int,
    n_subspaces: int,
    pq_iter: int,
) -> FlatIndex:
    """Build initial flat index by clustering all keys with BF=bf."""
    lo_list, hi_list, kidx_list = _cluster(keys_h, key_idx, bf, n_subspaces, pq_iter)
    device = keys_h.device
    return FlatIndex(
        lo=torch.stack(lo_list),                                              # (K, d)
        hi=torch.stack(hi_list),                                              # (K, d)
        sizes=torch.tensor([k.numel() for k in kidx_list],
                           device=device, dtype=torch.long),                  # (K,)
        key_idx=kidx_list,
        cluster_bf=[bf] * len(kidx_list),
    )


def _search_flat(
    index: FlatIndex,
    q_h: torch.Tensor,      # (d,)
    th_h: float,
    bf_min: int,
    seen_refine: set[int],  # id(key_idx tensor) already queued — dedup across window
    refine_queue: set[int], # ids to refine at end of window
) -> int:
    """
    Batched AABB query over all K clusters.
    Intersecting clusters are scanned (counted) and queued for refinement.
    No k-means here — pure tensor ops.
    """
    q_exp = q_h.unsqueeze(0)                                        # (1, d)
    max_dot = torch.maximum(q_exp * index.lo, q_exp * index.hi).sum(-1)  # (K,)
    min_dot = torch.minimum(q_exp * index.lo, q_exp * index.hi).sum(-1)  # (K,)

    outside = max_dot <= th_h
    inside  = min_dot > th_h
    inter   = ~outside & ~inside

    scanned  = int(index.sizes[inside].sum().item())
    scanned += int(index.sizes[inter].sum().item())

    for cid in inter.nonzero(as_tuple=False).view(-1).tolist():
        cbf      = index.cluster_bf[cid]
        child_bf = max(bf_min, cbf // 2)
        if cbf > bf_min and int(index.sizes[cid].item()) > child_bf:
            uid = id(index.key_idx[cid])
            if uid not in seen_refine:
                seen_refine.add(uid)
                refine_queue.add(uid)

    return scanned


def _apply_refinement(
    index: FlatIndex,
    queued_ids: set[int],   # id(key_idx) of clusters to split
    keys_h: torch.Tensor,
    n_subspaces: int,
    pq_iter: int,
    bf_min: int,
) -> int:
    """
    Replace each queued cluster with its tighter sub-clusters (in-place on index).
    Each cluster is split independently — no cross-cluster pooling.
    Returns number of new clusters created.
    """
    new_lo:         list[torch.Tensor] = []
    new_hi:         list[torch.Tensor] = []
    new_kidx:       list[torch.Tensor] = []
    new_cluster_bf: list[int]          = []
    built = 0

    for lo_i, hi_i, kidx, cbf in zip(index.lo, index.hi, index.key_idx, index.cluster_bf):
        if id(kidx) in queued_ids:
            child_bf = max(bf_min, cbf // 2)
            if cbf > bf_min and int(kidx.numel()) > child_bf:
                sub_lo, sub_hi, sub_kidx = _cluster(
                    keys_h, kidx, child_bf, n_subspaces, pq_iter
                )
                new_lo.extend(sub_lo)
                new_hi.extend(sub_hi)
                new_kidx.extend(sub_kidx)
                new_cluster_bf.extend([child_bf] * len(sub_kidx))
                built += len(sub_kidx)
                continue        # old cluster replaced — do not keep it
        # keep cluster unchanged
        new_lo.append(lo_i)
        new_hi.append(hi_i)
        new_kidx.append(kidx)
        new_cluster_bf.append(cbf)

    device = keys_h.device
    index.lo         = torch.stack(new_lo)
    index.hi         = torch.stack(new_hi)
    index.sizes      = torch.tensor([k.numel() for k in new_kidx],
                                    device=device, dtype=torch.long)
    index.key_idx    = new_kidx
    index.cluster_bf = new_cluster_bf
    return built


def _index_stats(index: FlatIndex) -> dict[int, int]:
    """Count clusters per bf level."""
    counts: dict[int, int] = {}
    for cbf in index.cluster_bf:
        counts[cbf] = counts.get(cbf, 0) + 1
    return counts


def _run_adaptive(
    keys: torch.Tensor,
    queries: torch.Tensor,
    q_indices: list[int],
    topk: int,
    cfg: AdaptiveConfig,
    show_progress: bool,
) -> dict[str, float | int | str]:
    h, n, _ = keys.shape

    # ------------------------------------------------------------------
    # Build flat index for each head — one k-means pass, children=None
    # ------------------------------------------------------------------
    build_t0 = time.perf_counter()
    root_idx = torch.arange(n, device=keys.device, dtype=torch.long)

    head_iter: range | tqdm = range(h)
    if show_progress:
        head_iter = tqdm(head_iter, desc=f"Build [{cfg.label}]", leave=False)

    indices: list[FlatIndex] = []
    for hi in head_iter:
        indices.append(_build_flat(
            keys_h=keys[hi],
            key_idx=root_idx,
            bf=cfg.bf_start,
            n_subspaces=cfg.n_subspaces,
            pq_iter=cfg.pq_iter,
        ))

    torch.cuda.synchronize()
    root_build_ms = (time.perf_counter() - build_t0) * 1000.0

    # ------------------------------------------------------------------
    # Search + adaptive per-cluster refinement between windows
    # ------------------------------------------------------------------
    fracs: list[float]  = []
    window_fracs: list[float] = []
    search_times: list[float] = []
    refine_times: list[float] = []
    clusters_built_total = 0

    # Per-head refinement queues (sets of id(key_idx)).
    window_queues: list[set[int]] = [set() for _ in range(h)]
    # Per-head dedup: never re-queue a cluster already in the current window queue.
    seen_refine: list[set[int]] = [set() for _ in range(h)]

    q_iter: list[int] | tqdm = q_indices
    if show_progress:
        q_iter = tqdm(q_indices, desc=f"Search [{cfg.label}]", leave=False)

    for step_i, qi in enumerate(q_iter, start=1):
        q = queries[:, qi, :]
        q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_normal = q / q_norm                               # (h, d)
        th = _topk_threshold(q_normal, keys, topk=topk)    # (h,)

        # ---- search (timed — pure tensor ops, no k-means) ------------
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        scanned_total = 0
        for hi in range(h):
            scanned_total += _search_flat(
                index=indices[hi],
                q_h=q_normal[hi],
                th_h=float(th[hi].item()),
                bf_min=cfg.bf_min,
                seen_refine=seen_refine[hi],
                refine_queue=window_queues[hi],
            )

        torch.cuda.synchronize()
        search_times.append(time.perf_counter() - t0)
        # ---- end timed search ----------------------------------------

        frac = float(scanned_total) / float(max(1, h * n))
        fracs.append(frac)
        window_fracs.append(frac)

        # ---- refinement (not counted in search_ms) -------------------
        end_of_window = (
            step_i % cfg.refine_window_size == 0
            or step_i == len(q_indices)
        )
        if end_of_window:
            r0 = time.perf_counter()
            for hi in range(h):
                if window_queues[hi]:
                    clusters_built_total += _apply_refinement(
                        index=indices[hi],
                        queued_ids=window_queues[hi],
                        keys_h=keys[hi],
                        n_subspaces=cfg.n_subspaces,
                        pq_iter=cfg.pq_iter,
                        bf_min=cfg.bf_min,
                    )
                    window_queues[hi] = set()
                    seen_refine[hi]   = set()
            torch.cuda.synchronize()
            refine_times.append(time.perf_counter() - r0)
            window_fracs = []
        # ---- end refinement ------------------------------------------

        if show_progress and hasattr(q_iter, "set_postfix"):
            if step_i % max(1, len(q_indices) // 50) == 0 or step_i == len(q_indices):
                mean_frac = sum(fracs) / len(fracs)
                mean_ms = (sum(search_times) / len(search_times)) * 1000.0
                q_iter.set_postfix(
                    scanned=f"{mean_frac:.4f}",
                    search_ms=f"{mean_ms:.3f}",
                    built=clusters_built_total,
                )

        if cfg.query_log_every > 0 and (
            step_i % cfg.query_log_every == 0 or step_i == 1 or step_i == len(q_indices)
        ):
            mean_frac   = sum(fracs) / len(fracs)
            win_frac    = sum(window_fracs) / len(window_fracs) if window_fracs else frac
            mean_ms     = (sum(search_times) / len(search_times)) * 1000.0
            last_ms     = search_times[-1] * 1000.0
            refine_ms   = (sum(refine_times) / len(refine_times)) * 1000.0 if refine_times else 0.0
            n_clusters  = sum(len(idx.key_idx) for idx in indices)
            print(
                f"  query {step_i:>5d}/{len(q_indices):<5d} "
                f"q_ms={last_ms:>8.2f} mean_ms={mean_ms:>8.2f} "
                f"refine_ms={refine_ms:>7.2f} "
                f"mean_scanned={mean_frac:>7.4f} win_scanned={win_frac:>7.4f} "
                f"clusters={n_clusters} built={clusters_built_total}"
            )

    mean_frac      = sum(fracs) / len(fracs) if fracs else 1.0
    mean_search_ms = (sum(search_times) / len(search_times)) * 1000.0 if search_times else 0.0
    mean_refine_ms = (sum(refine_times) / len(refine_times)) * 1000.0 if refine_times else 0.0

    # Final index statistics.
    aabb_by_bf: dict[int, int] = {}
    for idx in indices:
        for bf, cnt in _index_stats(idx).items():
            aabb_by_bf[bf] = aabb_by_bf.get(bf, 0) + cnt
    total_aabbs     = sum(aabb_by_bf.values())
    aabbs_per_head  = total_aabbs // max(1, h)
    bf_hist_str = ",".join(
        f"bf{bf}:{cnt // max(1, h)}" for bf, cnt in sorted(aabb_by_bf.items(), reverse=True)
    )

    return {
        "label":          cfg.label,
        "n_subspaces":    cfg.n_subspaces,
        "pq_iter":        cfg.pq_iter,
        "bf_start":       cfg.bf_start,
        "bf_min":         cfg.bf_min,
        "refine_window":  cfg.refine_window_size,
        "queries":        len(q_indices),
        "scanned_frac":   float(mean_frac),
        "pruned_frac":    float(1.0 - mean_frac),
        "mean_search_ms": float(mean_search_ms),
        "mean_refine_ms": float(mean_refine_ms),
        "root_build_ms":  float(root_build_ms),
        "clusters_built":  int(clusters_built_total),
        "aabbs_per_head":  int(aabbs_per_head),   # comparable to N // bf_min
        "aabbs_by_bf":     bf_hist_str,            # per-head counts per bf level
    }


def _variant_seed(base_seed: int, label: str) -> int:
    digest = hashlib.blake2s(label.encode("utf-8"), digest_size=4).digest()
    mix = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(base_seed) + mix) % (2**31 - 1)


def _build_configs(args: argparse.Namespace) -> list[AdaptiveConfig]:
    if not args.sweep:
        label = (
            f"adaptive(ns={args.n_subspaces},it={args.pq_iter},"
            f"bf={args.bf_start}->{args.bf_min},win={args.refine_window_size})"
        )
        return [AdaptiveConfig(
            label=label,
            n_subspaces=args.n_subspaces,
            pq_iter=args.pq_iter,
            bf_start=args.bf_start,
            bf_min=args.bf_min,
            refine_window_size=args.refine_window_size,
            query_log_every=args.query_log_every,
        )]

    ns_list  = _parse_int_list(args.sweep_n_subspaces)
    it_list  = _parse_int_list(args.sweep_pq_iter)
    win_list = _parse_int_list(args.sweep_refine_window_size)

    cfgs: list[AdaptiveConfig] = []
    for ns in ns_list:
        for it in it_list:
            for win in win_list:
                label = (
                    f"adaptive(ns={ns},it={it},"
                    f"bf={args.bf_start}->{args.bf_min},win={win})"
                )
                cfgs.append(AdaptiveConfig(
                    label=label,
                    n_subspaces=ns,
                    pq_iter=it,
                    bf_start=args.bf_start,
                    bf_min=args.bf_min,
                    refine_window_size=win,
                    query_log_every=args.query_log_every,
                ))
    return cfgs


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    cols = sorted({k for row in rows for k in row.keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model",        type=str, default=MODEL_NAME)
    p.add_argument("--layer",        type=int, default=LAYER_IDX)
    p.add_argument("--n-tokens",     type=int, default=2000)

    p.add_argument("--start-query",  type=int, default=0)
    p.add_argument("--max-queries",  type=int, default=2000)
    p.add_argument("--query-stride", type=int, default=1)
    p.add_argument("--topk",         type=int, default=20)

    p.add_argument("--bf-start",     type=int, default=16)
    p.add_argument("--bf-min",       type=int, default=2)
    p.add_argument("--n-subspaces",  type=int, default=2)
    p.add_argument("--pq-iter",      type=int, default=10)

    p.add_argument(
        "--refine-window-size", type=int, default=50,
        help="Replace intersecting clusters with sub-clusters every N queries.",
    )

    p.add_argument("--sweep",                    action="store_true")
    p.add_argument("--sweep-n-subspaces",        type=str, default="2,4")
    p.add_argument("--sweep-pq-iter",            type=str, default="5,10")
    p.add_argument("--sweep-refine-window-size", type=str, default="1,50,200")

    p.add_argument("--seed",   type=int,  default=1234)
    p.add_argument("--output", type=Path, default=Path("adaptive_refine_bf_results.csv"))
    p.add_argument(
        "--query-log-every", type=int, default=100,
        help="Print a query-progress log every N queries (0 disables).",
    )
    p.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction, default=True,
        help="Enable tqdm progress bars.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    if args.bf_start < 2:
        raise ValueError("--bf-start must be >= 2")
    if args.bf_min < 2 or args.bf_min > args.bf_start:
        raise ValueError("--bf-min must satisfy 2 <= bf_min <= bf_start")
    if args.refine_window_size < 1:
        raise ValueError("--refine-window-size must be >= 1")

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
    print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    layer_ids = capture.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]

    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)
    keys    = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    queries = queries_cpu.to(device=DEVICE, dtype=torch.float32)

    h_kv, n, d = keys.shape
    h_q, t_q, _ = queries.shape

    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
    keys_eval = keys[q_head_to_kv] if q_head_to_kv is not None else keys

    q_start   = max(0, int(args.start_query))
    q_end     = min(t_q, q_start + max(1, int(args.max_queries)) * max(1, int(args.query_stride)))
    q_indices = list(range(q_start, q_end, max(1, int(args.query_stride))))
    if not q_indices:
        raise ValueError("No query indices selected.")

    print(
        f"Layer {layer}: H_q={h_q}, H_eval={keys_eval.shape[0]}, N={n}, D={d}, "
        f"queries={len(q_indices)}, topk={args.topk}"
    )

    cfgs = _build_configs(args)
    print(f"Running {len(cfgs)} config(s)")

    rows: list[dict[str, object]] = []
    run_iter = cfgs
    if args.show_progress and len(cfgs) > 1:
        run_iter = tqdm(cfgs, desc="Adaptive sweep")

    for cfg in run_iter:
        seed = _variant_seed(args.seed, cfg.label)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n[{cfg.label}] seed={seed}")
        result = _run_adaptive(
            keys=keys_eval,
            queries=queries,
            q_indices=q_indices,
            topk=args.topk,
            cfg=cfg,
            show_progress=args.show_progress,
        )
        result.update({
            "seed":      seed,
            "layer":     layer,
            "n_tokens":  args.n_tokens,
            "n_queries": len(q_indices),
            "topk":      args.topk,
        })
        rows.append(result)

        print(
            f"scanned={float(result['scanned_frac']):.4f} "
            f"pruned={float(result['pruned_frac']):.4f} "
            f"search_ms={float(result['mean_search_ms']):.3f} "
            f"refine_ms={float(result['mean_refine_ms']):.3f} "
            f"root_build_ms={float(result['root_build_ms']):.1f} "
            f"clusters_built={int(result['clusters_built'])} "
            f"aabbs/head={int(result['aabbs_per_head'])} (max={n}//{args.bf_min}={n // args.bf_min}) "
            f"[{str(result['aabbs_by_bf'])}]"
        )

    rows.sort(key=lambda r: (float(r["scanned_frac"]), float(r["mean_search_ms"])))

    print("\n" + "=" * 120)
    print(f"{'RANK':<6s} {'LABEL':<60s} {'SCANNED':>8s} {'PRUNED':>8s} {'SEARCH_ms':>10s} {'REFINE_ms':>10s} {'BUILT':>8s}")
    print("-" * 120)
    for i, row in enumerate(rows[:20], start=1):
        print(
            f"{i:<6d} {str(row['label']):<60s} {float(row['scanned_frac']):>8.4f} "
            f"{float(row['pruned_frac']):>8.4f} {float(row['mean_search_ms']):>10.3f} "
            f"{float(row['mean_refine_ms']):>10.3f} {int(row['clusters_built']):>8d}"
        )

    _write_csv(rows, args.output)
    print(f"\nSaved CSV: {args.output}")


if __name__ == "__main__":
    main()