#!/usr/bin/env python3
"""
Sweep comparison.py over multiple generated-token counts and write CSV results.

The script captures Q/K/V once at the largest requested token count, then
reuses prefixes of that capture for smaller token counts. This matches the
meaning of comparison.py's --n-tokens argument while avoiding repeated model
loads and captures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "result" / "comparison_n_tokens_sweep"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clusterings import CLUSTERING_METHODS
from comparison import (
    DEVICE,
    DTYPE,
    GATE_COST_DP,
    LAYER_IDX,
    MODEL_NAME,
    PROMPT,
    format_speedup,
    measure_scanned_fraction,
)
from enclosings import ENCLOSING_METHODS
from pruning_bench_utils import _capture_qkv, _q_to_kv_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep the quick-pruning comparison benchmark over n-tokens values."
    )
    parser.add_argument(
        "--n-token-values",
        type=str,
        default="250,500,1000,2000,4000",
        help="Comma-separated generated-token counts to evaluate.",
    )
    parser.add_argument("--bf", type=int, default=4, help="Branching factor.")
    parser.add_argument(
        "--n-queries", type=int, default=30, help="Queries to evaluate."
    )
    parser.add_argument("--topk", type=int, default=20, help="Top-k for threshold.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HF model id.")
    parser.add_argument(
        "--layer",
        type=int,
        default=LAYER_IDX,
        help="Layer to analyze. Falls back to the middle captured layer if absent.",
    )
    parser.add_argument(
        "--clusterings",
        type=str,
        default="all",
        help='Comma-separated clustering names or "all".',
    )
    parser.add_argument(
        "--enclosings",
        type=str,
        default="all",
        help='Comma-separated enclosing names or "all".',
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV files.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--fp16-keys",
        action="store_true",
        help="Store keys on GPU as float16 instead of float32 (~2x memory reduction).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable capture progress display.",
    )
    return parser.parse_args()


def parse_n_token_values(raw: str) -> list[int]:
    vals = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        vals.append(int(piece))
    if not vals:
        raise ValueError("No n-token values provided.")
    if any(v < 1 for v in vals):
        raise ValueError("All n-token values must be at least 1.")
    return sorted(set(vals))


def select_query_indices(total_queries: int, n_queries: int) -> list[int]:
    if total_queries <= 0:
        return []
    stride = max(1, total_queries // n_queries)
    q_indices = list(
        range(
            total_queries - 1,
            max(0, total_queries - n_queries * stride) - 1,
            -stride,
        )
    )
    return q_indices[:n_queries]


def select_methods(
    methods: dict[str, object], wanted: str, label: str
) -> dict[str, object]:
    if wanted == "all":
        return methods

    names = [name.strip() for name in wanted.split(",") if name.strip()]
    selected = {}
    for name in names:
        if name not in methods:
            available = ", ".join(sorted(methods))
            raise ValueError(f"Unknown {label} method '{name}'. Available: {available}")
        selected[name] = methods[name]
    return selected


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    if args.bf < 1:
        raise ValueError("--bf must be at least 1.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_token_values = parse_n_token_values(args.n_token_values)
    max_n_tokens = max(n_token_values)
    clustering_methods = select_methods(
        CLUSTERING_METHODS, args.clusterings, "clustering"
    )
    enclosing_methods = select_methods(ENCLOSING_METHODS, args.enclosings, "enclosing")

    print(f"Capturing {max_n_tokens} tokens from {args.model} ...")
    t0 = time.perf_counter()
    capture = _capture_qkv(
        model_name=args.model,
        prompt_text=PROMPT,
        n=max_n_tokens,
        device=DEVICE,
        torch_dtype=DTYPE,
        show_progress=not args.no_progress,
    )
    print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU memory after model free: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    layer_ids = capture.layer_ids()
    if not layer_ids:
        raise RuntimeError("No layers were captured.")

    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu_full, keys_cpu_full, _ = capture.to_layer_tensors(layer)
    prompt_len = int(capture.prompt_length or 0)
    max_generated = int(queries_cpu_full.shape[1])
    if max_generated < max_n_tokens:
        raise RuntimeError(
            f"Capture produced only {max_generated} generated queries, expected {max_n_tokens}."
        )

    keys_dtype = torch.float16 if args.fp16_keys else torch.float32
    keys_full = keys_cpu_full.to(device=DEVICE, dtype=keys_dtype, non_blocking=True)
    queries_full = queries_cpu_full

    h_kv, n_full, d = keys_full.shape
    h_q = queries_full.shape[0]
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None

    print(f"Layer {layer}: H_kv={h_kv}, H_q={h_q}, prompt_len={prompt_len}, D={d}")
    print(f"bf={args.bf}, topk={args.topk}, n-token sweep={n_token_values}")
    print(f"Max captured keys={n_full}, generated queries={queries_full.shape[1]}")
    print("=" * 140)

    all_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for n_tokens in n_token_values:
        total_keys = prompt_len + n_tokens
        keys = keys_full[:, :total_keys, :]
        queries = queries_full[:, :n_tokens, :]
        k_parents = max(1, math.ceil(total_keys / args.bf))
        q_indices = select_query_indices(queries.shape[1], args.n_queries)
        if not q_indices:
            raise RuntimeError(f"No queries selected for n_tokens={n_tokens}.")

        keys_mb = keys.numel() * keys.element_size() / 1e6
        queries_mb = queries.numel() * queries.element_size() / 1e6
        print(f"\nEvaluating n_tokens={n_tokens}")
        print("-" * 140)
        print(
            f"N={total_keys} keys ({prompt_len} prompt + {n_tokens} generated), "
            f"K={k_parents}, queries={len(q_indices)}"
        )
        print(
            f"Memory: keys={keys_mb:.0f} MB on GPU ({keys.dtype}), "
            f"queries={queries_mb:.0f} MB on CPU"
        )

        rows_for_n: list[dict[str, object]] = []
        keys_f32 = keys.float() if keys.dtype != torch.float32 else keys

        cdist_gb = total_keys * total_keys * 4 / 1e9
        if cdist_gb > 1.0 and any(
            name in clustering_methods
            for name in ("nn_greedy", "fast_balanced_nn", "block_nn")
        ):
            print(
                f"WARNING: N={total_keys} -> cdist-based clustering will allocate "
                f"~{cdist_gb:.1f} GB. Use --clusterings kcenter,kmeans to avoid OOM."
            )

        for clust_name, clust_fn in clustering_methods.items():
            print(f"Clustering: {clust_name} ...")
            torch.cuda.synchronize()
            clust_t0 = time.perf_counter()
            assign, centers = clust_fn(keys_f32, args.bf)
            torch.cuda.synchronize()
            clust_ms = (time.perf_counter() - clust_t0) * 1000

            if q_head_to_kv is not None:
                assign_q = assign[q_head_to_kv]
                centers_q = centers[q_head_to_kv]
                keys_q = keys_f32[q_head_to_kv]
            else:
                assign_q = assign
                centers_q = centers
                keys_q = keys_f32

            for enc_name, enc_fn in enclosing_methods.items():
                torch.cuda.synchronize()
                enc_t0 = time.perf_counter()
                gate_fn, enc_info = enc_fn(
                    keys_q, assign_q, centers_q, k_parents, args.bf
                )
                torch.cuda.synchronize()
                enc_ms = (time.perf_counter() - enc_t0) * 1000

                frac, search_ms = measure_scanned_fraction(
                    gate_fn=gate_fn,
                    queries=queries,
                    keys=keys_q,
                    q_indices=q_indices,
                    q_head_to_kv=None,
                    K=k_parents,
                    bf=args.bf,
                    topk=args.topk,
                    assign=assign_q,
                )

                g = GATE_COST_DP.get(str(enc_name), 2.0)
                ratio = g / args.bf + frac
                row = {
                    "n_tokens": n_tokens,
                    "prompt_len": prompt_len,
                    "total_keys": total_keys,
                    "layer": layer,
                    "bf": args.bf,
                    "topk": args.topk,
                    "n_queries": len(q_indices),
                    "clustering": str(clust_name),
                    "enclosing": str(enc_name),
                    "scanned_frac": frac,
                    "pruned_frac": 1.0 - frac,
                    "search_ms": search_ms,
                    "build_ms": clust_ms + enc_ms,
                    "clust_ms": clust_ms,
                    "enc_ms": enc_ms,
                    "gate_cost_dp": g,
                    "ratio": ratio,
                    "speedup": 1.0 / ratio,
                    **{f"enc_{key}": value for key, value in dict(enc_info).items()},
                }
                rows_for_n.append(row)
                all_rows.append(row)

                print(
                    f"  {enc_name:<20s}  scanned={frac:.4f}  pruned={1.0-frac:.4f}  "
                    f"search={search_ms:.3f}ms  build={clust_ms + enc_ms:.1f}ms  "
                    f"ratio={ratio:.3f}  speedup={format_speedup(ratio)}"
                )

        rows_for_n.sort(key=lambda row: float(row["ratio"]))
        best = rows_for_n[0]
        best_rows.append(best)
        print(
            f"Best at n_tokens={n_tokens}: {best['clustering']} + {best['enclosing']}  "
            f"scanned={float(best['scanned_frac']):.4f}  "
            f"ratio={float(best['ratio']):.3f}  "
            f"speedup={format_speedup(float(best['ratio']))}"
        )

    all_rows.sort(key=lambda row: (int(row["n_tokens"]), float(row["ratio"])))
    best_rows.sort(key=lambda row: int(row["n_tokens"]))

    all_csv = args.output_dir / "all_results.csv"
    best_csv = args.output_dir / "best_per_n_tokens.csv"
    write_csv(all_rows, all_csv)
    write_csv(best_rows, best_csv)

    print("\n" + "=" * 140)
    print("Best Per n_tokens")
    print("-" * 140)
    print(
        f"{'n_tokens':>10s} {'N':>8s} {'CLUSTERING':<22s} {'ENCLOSING':<22s} "
        f"{'SCANNED':>8s} {'PRUNED':>8s} {'SEARCH_ms':>10s} {'BUILD_ms':>9s} "
        f"{'g':>5s} {'RATIO':>7s} {'SPEEDUP':>8s}"
    )
    print("-" * 140)
    for row in best_rows:
        print(
            f"{int(row['n_tokens']):>10d} {int(row['total_keys']):>8d} "
            f"{str(row['clustering']):<22s} {str(row['enclosing']):<22s} "
            f"{float(row['scanned_frac']):>8.4f} {float(row['pruned_frac']):>8.4f} "
            f"{float(row['search_ms']):>10.3f} {float(row['build_ms']):>9.1f} "
            f"{float(row['gate_cost_dp']):>5.1f} {float(row['ratio']):>7.3f} "
            f"{format_speedup(float(row['ratio'])):>8s}"
        )
    print("=" * 140)

    overall_best = min(all_rows, key=lambda row: float(row["ratio"]))
    print(
        f"\nOverall best asymptotic speedup: n_tokens={int(overall_best['n_tokens'])}  "
        f"{overall_best['clustering']} + {overall_best['enclosing']}  "
        f"ratio={float(overall_best['ratio']):.3f}  "
        f"speedup={format_speedup(float(overall_best['ratio']))}"
    )
    print(f"All results CSV: {all_csv}")
    print(f"Best-per-n_tokens CSV: {best_csv}")


if __name__ == "__main__":
    main()
