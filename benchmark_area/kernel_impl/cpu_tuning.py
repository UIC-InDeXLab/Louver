"""Tune CPU attention_v4.3 hyperparameters.

The current v4.3 kernel has only a few real knobs:

  - parents_per_tile: compile-time HIRA_V4_PARENTS_PER_TILE. This controls
    both the scoring tile size and, when enabled, the parent-block granularity
    of the parallel mask build.
  - parallel_pass_blocks: compile-time HIRA_V4_PARALLEL_PASS_BLOCKS. This
    splits the per-subspace parent-pass mask build over parent blocks instead
    of only over (subspace, kv_head).
  - threads: runtime torch/OpenMP thread count for the benchmark process.

The script JIT-builds temporary v4 variants and compares them to the currently
checked-in attention_v4_3 configuration: parents_per_tile=128,
parallel_pass_blocks=1.

Example:
    python benchmark_area/kernel_impl/cpu_tuning.py \
        --input benchmark_area/quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \
        --bf 4 --S 8 --buffer 256
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels._cpu_ext_loader import (
        _BASE_CFLAGS,
        _BASE_LDFLAGS,
    )
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.attention_v4_3 import (
        attend as attend_v4_3,
    )
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import (
        real_attention_case,
    )
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_attention import (
        dense_attention,
        sdpa_attention,
        subspace_topk_thresholds,
    )
except ModuleNotFoundError:
    from benchmark_area.kernel_impl.kernels.cpu_kernels._cpu_ext_loader import (
        _BASE_CFLAGS,
        _BASE_LDFLAGS,
    )
    from benchmark_area.kernel_impl.kernels.cpu_kernels.attention_v4_3 import (
        attend as attend_v4_3,
    )
    from benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build
    from benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import (
        real_attention_case,
    )
    from benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_attention import (
        dense_attention,
        sdpa_attention,
        subspace_topk_thresholds,
    )


CURRENT_PARENTS_PER_TILE = 128
CURRENT_PARALLEL_PASS_BLOCKS = True
_GREEN = "\033[32m"
_RESET = "\033[0m"


@dataclass(frozen=True, order=True)
class TrialConfig:
    parents_per_tile: int
    parallel_pass_blocks: bool
    threads: int


def parse_int_list(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise argparse.ArgumentTypeError("empty integer list")
    if any(v < 1 for v in vals):
        raise argparse.ArgumentTypeError("all values must be positive")
    return vals


def parse_bool_list(raw: str) -> list[bool]:
    vals: set[bool] = set()
    for item in raw.split(","):
        s = item.strip().lower()
        if not s:
            continue
        if s in {"1", "true", "yes", "on"}:
            vals.add(True)
        elif s in {"0", "false", "no", "off"}:
            vals.add(False)
        else:
            raise argparse.ArgumentTypeError(
                f"invalid boolean value {item!r}; use 0/1"
            )
    if not vals:
        raise argparse.ArgumentTypeError("empty boolean list")
    return sorted(vals)


def format_ms(ms: float, best_ms: float | None = None) -> str:
    text = f"{ms:.4f}"
    if best_ms is not None and ms == best_ms:
        return f"{_GREEN}{text}{_RESET}"
    return text


def time_call(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def generated_source_path(cfg: TrialConfig) -> Path:
    src_dir = Path(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions"))
    src_dir = src_dir / "hira_cpu_v43_tuning_sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    mode = "pb1" if cfg.parallel_pass_blocks else "pb0"
    return src_dir / f"attention_v43_tile{cfg.parents_per_tile}_{mode}.cpp"


def build_variant(cfg: TrialConfig):
    cpu_dir = (
        Path(__file__).resolve().parent / "kernels" / "cpu_kernels"
    )
    common_header = cpu_dir / "_attention_v4_bitmask_common.h"
    src = generated_source_path(cfg)
    fn_name = f"attend_v43_t{cfg.parents_per_tile}_{int(cfg.parallel_pass_blocks)}"
    lines = [
        f"#define HIRA_V4_ATTEND_FN {fn_name}",
        f"#define HIRA_V4_PARENTS_PER_TILE {cfg.parents_per_tile}",
    ]
    if cfg.parallel_pass_blocks:
        lines.append("#define HIRA_V4_PARALLEL_PASS_BLOCKS")
    lines.append(f'#include "{common_header}"')
    src.write_text("\n".join(lines) + "\n")
    return load(
        name=f"hira_cpu_attention_v43_tune_t{cfg.parents_per_tile}_pb{int(cfg.parallel_pass_blocks)}",
        sources=[str(src)],
        extra_cflags=list(_BASE_CFLAGS),
        extra_ldflags=list(_BASE_LDFLAGS),
        verbose=False,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Tune CPU attention_v4.3 knobs.")
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, default=None)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--n-queries", type=int, default=16)
    p.add_argument("--buffer-len", "--buffer", dest="buffer_len", type=int, default=256)
    p.add_argument("--refine-iter", type=int, default=2)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--parents-per-tile-sweep",
        type=parse_int_list,
        default=parse_int_list("32,64,96,128"),
    )
    p.add_argument(
        "--parallel-pass-blocks-sweep",
        type=parse_bool_list,
        default=parse_bool_list("0,1"),
        help="Comma-separated 0/1 values for HIRA_V4_PARALLEL_PASS_BLOCKS.",
    )
    p.add_argument(
        "--threads-sweep",
        type=parse_int_list,
        default=parse_int_list(str(os.cpu_count() or 1)),
    )
    p.add_argument("--top-results", type=int, default=10)
    p.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip dense/SDPA timing and only tune v4.3 variants.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_qkv is None:
        raise ValueError("--input/--input-qkv is required for CPU tuning")

    print(f"Loading capture from {args.input_qkv} ...")
    real = real_attention_case(
        args.input_qkv, args.layer, args.N, args.n_queries, args.buffer_len,
    )
    keys = real["keys"]
    values = real["values"]
    keys_eval = real["keys_eval"]
    values_eval = real["values_eval"]
    buffer_keys = real["buffer_keys"]
    buffer_values = real["buffer_values"]
    q_batch = real["q_batch"]
    q_head_to_kv = real["q_head_to_kv"]

    scale = 1.0 / math.sqrt(keys.shape[-1])
    print(
        f"CPU v4.3 tuning: layer={real['layer']} H_q={q_batch.shape[1]} "
        f"H_kv={keys.shape[0]} N_idx={keys.shape[1]} "
        f"N_buf={buffer_keys.shape[1]} D={keys.shape[-1]} "
        f"bf={args.bf} S={args.S} queries={q_batch.shape[0]}"
    )
    print(
        "Tuned knobs: parents_per_tile (compile-time), "
        "parallel_pass_blocks (compile-time), threads (runtime)."
    )
    print("Building CPU index state ...")
    state = build(keys, args.bf, args.S, args.refine_iter, values=values)

    pairs = [
        (
            q.contiguous(),
            subspace_topk_thresholds(
                q, keys_eval, args.topk, state["dim_slices"],
                q_head_to_kv=q_head_to_kv,
            ),
        )
        for q in q_batch
    ]

    requested = {
        TrialConfig(tile, parallel, threads)
        for tile, parallel, threads in itertools.product(
            args.parents_per_tile_sweep,
            args.parallel_pass_blocks_sweep,
            args.threads_sweep,
        )
    }
    default_threads = max(args.threads_sweep)
    requested.add(
        TrialConfig(
            CURRENT_PARENTS_PER_TILE,
            CURRENT_PARALLEL_PASS_BLOCKS,
            int(default_threads),
        )
    )
    configs = sorted(requested)

    modules: dict[tuple[int, bool], object] = {}
    results: list[tuple[float, TrialConfig, str]] = []

    print(
        f"Trials: {len(configs)} configs x {len(pairs)} queries x "
        f"{args.iters} timed iters"
    )
    for idx, cfg in enumerate(configs, start=1):
        torch.set_num_threads(cfg.threads)
        key = (cfg.parents_per_tile, cfg.parallel_pass_blocks)
        label = ""
        try:
            if (
                cfg.parents_per_tile == CURRENT_PARENTS_PER_TILE
                and cfg.parallel_pass_blocks == CURRENT_PARALLEL_PASS_BLOCKS
            ):
                attend_fn = attend_v4_3
                label = "current"
            else:
                mod = modules.get(key)
                if mod is None:
                    mod = build_variant(cfg)
                    modules[key] = mod
                attend_fn = mod.attend

            def run_trial() -> None:
                for q, th in pairs:
                    attend_fn(
                        q, th, state,
                        buffer_keys=buffer_keys,
                        buffer_values=buffer_values,
                        q_head_to_kv=q_head_to_kv,
                        scale=scale,
                    )

            ms = time_call(run_trial, iters=args.iters, warmup=args.warmup)
            per_q = ms / len(pairs)
            results.append((per_q, cfg, label))
            suffix = f"  {label}" if label else ""
            print(
                f"[{idx:02d}/{len(configs):02d}] "
                f"tile={cfg.parents_per_tile:3d} "
                f"parallel_pass={int(cfg.parallel_pass_blocks)} "
                f"threads={cfg.threads:2d}: {per_q:.4f} ms/query{suffix}"
            )
        except Exception as exc:
            msg = str(exc).splitlines()[0]
            if len(msg) > 220:
                msg = msg[:217] + "..."
            print(
                f"[{idx:02d}/{len(configs):02d}] "
                f"tile={cfg.parents_per_tile:3d} "
                f"parallel_pass={int(cfg.parallel_pass_blocks)} "
                f"threads={cfg.threads:2d}: FAILED "
                f"({type(exc).__name__}: {msg})"
            )

    if not results:
        raise RuntimeError("all CPU tuning trials failed")

    results.sort(key=lambda x: x[0])
    best_ms, best_cfg, _ = results[0]
    current_matches = [
        (ms, cfg) for ms, cfg, _ in results
        if cfg.parents_per_tile == CURRENT_PARENTS_PER_TILE
        and cfg.parallel_pass_blocks == CURRENT_PARALLEL_PASS_BLOCKS
    ]
    current_best_ms = min((ms for ms, _ in current_matches), default=None)

    print("\nTop configs:")
    for rank, (ms, cfg, label) in enumerate(results[: args.top_results], start=1):
        delta = ""
        if current_best_ms is not None:
            delta = f"  delta_vs_current_best={ms - current_best_ms:+.4f}"
        suffix = f"  {label}" if label else ""
        print(
            f"{rank:2d}. {format_ms(ms, best_ms)} ms/query  "
            f"tile={cfg.parents_per_tile} "
            f"parallel_pass={int(cfg.parallel_pass_blocks)} "
            f"threads={cfg.threads}{delta}{suffix}"
        )

    print("\nBest CPU v4.3 hyperparameters:")
    print(f"  parents_per_tile:     {best_cfg.parents_per_tile}")
    print(f"  parallel_pass_blocks: {int(best_cfg.parallel_pass_blocks)}")
    print(f"  threads:              {best_cfg.threads}")
    print(f"  best_time:            {best_ms:.4f} ms/query")
    if current_best_ms is not None:
        print(f"  current_best_time:    {current_best_ms:.4f} ms/query")
        print(f"  improvement:          {current_best_ms - best_ms:+.4f} ms/query")
        if (
            best_cfg.parents_per_tile == CURRENT_PARENTS_PER_TILE
            and best_cfg.parallel_pass_blocks == CURRENT_PARALLEL_PASS_BLOCKS
        ):
            print("  current_kernel:       uses the best compile-time hyperparameters tested")
        else:
            print("  current_kernel:       does NOT use the best compile-time hyperparameters tested")

    if not args.skip_baselines:
        torch.set_num_threads(best_cfg.threads)
        keys_bf16 = keys_eval.to(torch.bfloat16).contiguous()
        values_bf16 = values_eval.to(torch.bfloat16).contiguous()
        q_bf16 = [q.to(torch.bfloat16).contiguous() for q, _ in pairs]

        def run_dense() -> None:
            for q, _ in pairs:
                dense_attention(q, keys_eval, values_eval, scale, q_head_to_kv)

        def run_sdpa_fp32() -> None:
            for q, _ in pairs:
                sdpa_attention(q, keys_eval, values_eval, scale, q_head_to_kv)

        def run_sdpa_bf16() -> None:
            for q in q_bf16:
                sdpa_attention(q, keys_bf16, values_bf16, scale, q_head_to_kv)

        dense_ms = time_call(run_dense, iters=args.iters, warmup=args.warmup) / len(pairs)
        sdpa_fp32_ms = time_call(run_sdpa_fp32, iters=args.iters, warmup=args.warmup) / len(pairs)
        sdpa_bf16_ms = time_call(run_sdpa_bf16, iters=args.iters, warmup=args.warmup) / len(pairs)
        print("\nBaselines at best thread count:")
        print(f"  dense_fp32: {dense_ms:.4f} ms/query")
        print(f"  sdpa_fp32:  {sdpa_fp32_ms:.4f} ms/query")
        print(f"  sdpa_bf16:  {sdpa_bf16_ms:.4f} ms/query")


if __name__ == "__main__":
    main()
