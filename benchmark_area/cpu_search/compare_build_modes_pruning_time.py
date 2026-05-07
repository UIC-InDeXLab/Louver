import argparse
import csv
import itertools
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
HIRA_ROOT = REPO_ROOT / "hira"
for p in (REPO_ROOT, HIRA_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from hira.indexer import CPUIndexer, CPUIndexerV2
from hira.searcher import CPUSearcher


class _NullProgressBar:
    def update(self, n: int = 1) -> None:
        _ = n

    def set_postfix_str(self, s: str) -> None:
        _ = s

    def close(self) -> None:
        pass


def make_progress_bar(total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(total=total, desc=desc, dynamic_ncols=True)
    return _NullProgressBar()


def progress_iter(iterable: Iterable, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True)
    return iterable


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [batch, n_kv_heads, seq_len, head_dim]
    b, n_kv, s, d = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(b, n_kv, n_rep, s, d)
    return x.reshape(b, n_kv * n_rep, s, d)


def compute_threshold(
    query: torch.Tensor, keys_prefix: torch.Tensor, topk: int
) -> torch.Tensor:
    # query: (1, H, 1, D), keys_prefix: (1, H, N, D)
    q = query / query.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
    scores = (q * keys_prefix).sum(dim=-1)  # (1, H, N)
    k = max(1, min(topk, scores.size(-1)))
    threshold = -(-scores).kthvalue(k=k, dim=-1).values  # (1, H)
    return threshold.squeeze(0).contiguous()


def measure_pruning_ratio(
    searcher: CPUSearcher,
    indexer: CPUIndexer,
    query: torch.Tensor,
    keys_prefix: torch.Tensor,
    topk: int,
) -> float:
    threshold = compute_threshold(query, keys_prefix, topk)
    searcher.search(query, threshold, indexer)
    exact_checks = int(searcher.stats["exact_checks"])
    h = int(query.size(1))
    n = int(indexer.num_keys)
    denom = max(1, h * n)
    return 1.0 - (exact_checks / denom)


def make_eval_positions(
    prefill: int, total_tokens: int, update_every: int, max_steps: Optional[int]
) -> list[int]:
    positions = []
    cursor = prefill
    while cursor < total_tokens:
        next_cursor = min(cursor + update_every, total_tokens)
        positions.append(next_cursor - 1)
        cursor = next_cursor
        if max_steps is not None and len(positions) >= max_steps:
            break
    if not positions:
        positions = [total_tokens - 1]
    return positions


def run_incremental_case(
    mode_name: str,
    indexer_factory: Callable[[], CPUIndexer],
    keys: torch.Tensor,
    queries: torch.Tensor,
    prefill: int,
    update_every: int,
    topk: int,
    max_steps: Optional[int],
    show_progress: bool,
) -> dict:
    indexer = indexer_factory().build(keys[:, :, :prefill, :])
    searcher = CPUSearcher(profiling=True, kernel="torch")

    update_times = []
    pruning_ratios = []
    eval_positions = make_eval_positions(prefill, keys.size(2), update_every, max_steps)

    cursor = prefill
    step_iter = progress_iter(
        eval_positions, desc=f"{mode_name} updates", enabled=show_progress
    )
    for pos in step_iter:
        next_cursor = pos + 1
        new_chunk = keys[:, :, cursor:next_cursor, :]

        t0 = time.perf_counter()
        indexer.update(new_chunk)
        update_times.append(time.perf_counter() - t0)

        query = queries[:, :, pos : pos + 1, :]
        keys_prefix = keys[:, :, :next_cursor, :]
        pruning = measure_pruning_ratio(searcher, indexer, query, keys_prefix, topk)
        pruning_ratios.append(pruning)

        cursor = next_cursor

    avg_update = float(sum(update_times) / len(update_times)) if update_times else 0.0
    avg_pruning = (
        float(sum(pruning_ratios) / len(pruning_ratios)) if pruning_ratios else 0.0
    )
    return {
        "mode": mode_name,
        "updates": len(update_times),
        "avg_update_time_s": avg_update,
        "avg_pruning_ratio": avg_pruning,
    }


def run_full_build_case(
    keys: torch.Tensor,
    queries: torch.Tensor,
    prefill: int,
    update_every: int,
    topk: int,
    max_steps: Optional[int],
    num_levels: int,
    branching_factor: int,
    max_iterations: int,
    balance_every: int,
    centroid_refine_iters: int,
    show_progress: bool,
) -> dict:
    t0 = time.perf_counter()
    indexer = CPUIndexerV2(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=max_iterations,
        balance_every=balance_every,
        centroid_refine_iters=centroid_refine_iters,
    ).build(keys)
    full_build_time = time.perf_counter() - t0

    searcher = CPUSearcher(profiling=True, kernel="torch")
    pruning_ratios = []
    eval_positions = make_eval_positions(prefill, keys.size(2), update_every, max_steps)
    step_iter = progress_iter(
        eval_positions, desc="cpu_v1_full_build eval", enabled=show_progress
    )
    for pos in step_iter:
        query = queries[:, :, pos : pos + 1, :]
        pruning = measure_pruning_ratio(searcher, indexer, query, keys, topk)
        pruning_ratios.append(pruning)

    avg_pruning = (
        float(sum(pruning_ratios) / len(pruning_ratios)) if pruning_ratios else 0.0
    )
    return {
        "mode": "cpu_v1_full_build_whole_keys",
        "updates": 1,
        "avg_update_time_s": float(full_build_time),
        "avg_pruning_ratio": avg_pruning,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare update-time and pruning ratio across CPUIndexer modes."
    )
    parser.add_argument("--keys", type=Path, default=SCRIPT_DIR / "keys.pt")
    parser.add_argument("--queries", type=Path, default=SCRIPT_DIR / "queries.pt")
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--repeat-kv", type=int, default=3)
    parser.add_argument("--heads", type=int, default=0, help="0 means use all heads")

    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--update-every", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit")
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--num-levels", type=int, default=5)
    parser.add_argument("--branching-factor", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--centroid-refine-iters", type=int, default=0)
    parser.add_argument(
        "--balance-every",
        type=int,
        default=-1,
        help="-1 means use update-every, 0 disables balancing",
    )
    parser.add_argument("--kmeans-split-iters", type=int, default=3)

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=SCRIPT_DIR / "compare_build_modes_pruning_time.csv",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def main():
    size = 10000  # size of keys

    args = parse_args()
    max_steps = None if args.max_steps <= 0 else args.max_steps
    balance_every = args.update_every if args.balance_every < 0 else args.balance_every
    show_progress = not args.no_progress

    if show_progress and tqdm is None:
        print("tqdm not found; running without progress bars.")

    keys = torch.load(args.keys, map_location="cpu")[args.layer_idx, :].unsqueeze(0)
    queries = torch.load(args.queries, map_location="cpu")[args.layer_idx, :].unsqueeze(
        0
    )

    if args.repeat_kv > 1:
        keys = repeat_kv(keys, args.repeat_kv)

    keys = keys.contiguous()
    queries = queries.contiguous()

    num_heads = min(keys.size(1), queries.size(1))
    if args.heads > 0:
        num_heads = min(num_heads, args.heads)
    keys = keys[:, :num_heads, :, :]
    queries = queries[:, :num_heads, :, :]

    total_tokens = min(keys.size(2), queries.size(2))
    keys = keys[:, :, :total_tokens, :]
    queries = queries[:, :, :total_tokens, :]

    keys = keys[:, :, :size, :]

    if total_tokens < 2:
        raise ValueError(f"Need at least 2 tokens, got {total_tokens}")
    if args.prefill <= 0 or args.prefill >= total_tokens:
        raise ValueError(
            f"prefill must be in [1, {total_tokens - 1}], got {args.prefill}"
        )

    results = []
    total_modes = 1 + (2**3) + 1
    mode_pbar = make_progress_bar(
        total=total_modes, desc="Modes", enabled=show_progress
    )

    def make_cpu_v1() -> CPUIndexer:
        return CPUIndexer(
            num_levels=args.num_levels,
            branching_factor=args.branching_factor,
            max_iterations=args.max_iterations,
            balance_every=balance_every,
            centroid_refine_iters=args.centroid_refine_iters,
        )

    print("Running cpu_v1_incremental...")
    results.append(
        run_incremental_case(
            mode_name="cpu_v1_incremental",
            indexer_factory=make_cpu_v1,
            keys=keys,
            queries=queries,
            prefill=args.prefill,
            update_every=args.update_every,
            topk=args.topk,
            max_steps=max_steps,
            show_progress=show_progress,
        )
    )
    mode_pbar.update(1)
    mode_pbar.set_postfix_str("cpu_v1_incremental")

    for (
        enable_radius_tightening,
        enable_smart_balance_split,
    ) in itertools.product([False, True], repeat=2):
        mode_name = (
            "cpu_v2_hr"
            f"{int(0)}"
            "_rt"
            f"{int(enable_radius_tightening)}"
            "_sb"
            f"{int(enable_smart_balance_split)}"
        )

        def make_cpu_v2(
            hr: bool = False,
            rt: bool = enable_radius_tightening,
            sb: bool = enable_smart_balance_split,
        ) -> CPUIndexerV2:
            return CPUIndexerV2(
                num_levels=args.num_levels,
                branching_factor=args.branching_factor,
                max_iterations=args.max_iterations,
                balance_every=balance_every,
                centroid_refine_iters=args.centroid_refine_iters,
                enable_hybrid_rebuild=hr,
                rebuild_every=args.update_every,
                enable_radius_tightening=rt,
                tighten_every=args.update_every,
                enable_smart_balance_split=sb,
                kmeans_split_iters=args.kmeans_split_iters,
                reassign_every=0,
            )

        print(f"Running {mode_name}...")
        results.append(
            run_incremental_case(
                mode_name=mode_name,
                indexer_factory=make_cpu_v2,
                keys=keys,
                queries=queries,
                prefill=args.prefill,
                update_every=args.update_every,
                topk=args.topk,
                max_steps=max_steps,
                show_progress=show_progress,
            )
        )
        mode_pbar.update(1)
        mode_pbar.set_postfix_str(mode_name)

    print("Running cpu_v1_full_build_whole_keys...")
    results.append(
        run_full_build_case(
            keys=keys,
            queries=queries,
            prefill=args.prefill,
            update_every=args.update_every,
            topk=args.topk,
            max_steps=max_steps,
            num_levels=args.num_levels,
            branching_factor=args.branching_factor,
            max_iterations=args.max_iterations,
            balance_every=balance_every,
            centroid_refine_iters=args.centroid_refine_iters,
            show_progress=show_progress,
        )
    )
    mode_pbar.update(1)
    mode_pbar.set_postfix_str("cpu_v1_full_build_whole_keys")
    mode_pbar.close()

    results = sorted(results, key=lambda r: r["mode"])

    fieldnames = ["mode", "updates", "avg_update_time_s", "avg_pruning_ratio"]
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nMinimal report")
    print(
        f"{'mode':<30} {'updates':>7} {'avg_update_time_s':>18} {'avg_pruning_ratio':>18}"
    )
    for row in results:
        print(
            f"{row['mode']:<30} "
            f"{row['updates']:>7d} "
            f"{row['avg_update_time_s']:>18.6f} "
            f"{row['avg_pruning_ratio']:>18.6f}"
        )
    print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
