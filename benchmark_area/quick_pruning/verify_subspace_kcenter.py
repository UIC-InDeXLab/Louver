#!/usr/bin/env python3
"""
Verify that subspace_kcenter never prunes true top-k points.

For each selected layer and sampled query position, this script builds the
subspace_kcenter index and checks per head that every exact top-k key survives
the halfspace gate. Any miss is reported as a correctness violation.

Usage:
    python verify_subspace_kcenter.py --input-qkv capture_qkv_2000_model.pt
    python verify_subspace_kcenter.py --n-tokens 2000 --topk 20 --n-subspaces 4
    python verify_subspace_kcenter.py --layers all --queries all
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import CaptureState, _capture_qkv, _q_to_kv_map
from clusterings.subspace_kcenter_ball import (
    SUBSPACE_STRATEGIES,
    build_subspace_kcenter,
    project_keys_for_index,
    project_query_for_index,
    subspace_ball_gate,
)


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
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


def _parse_layers(spec: str, available_layers: list[int]) -> list[int]:
    if spec == "all":
        return available_layers
    wanted = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        wanted.append(int(part))
    missing = sorted(set(wanted) - set(available_layers))
    if missing:
        raise ValueError(f"Requested layers not present in capture: {missing}")
    return sorted(wanted)


def _select_query_indices(total_q: int, spec: str, n_queries: int) -> list[int]:
    if spec == "all":
        return list(range(total_q))
    if spec == "tail":
        stride = max(1, total_q // max(1, n_queries))
        q_indices = list(
            range(total_q - 1, max(0, total_q - n_queries * stride) - 1, -stride)
        )
        return q_indices[:n_queries]
    raise ValueError(f"Unknown query selection mode: {spec}")


def topk_threshold(q_normal: torch.Tensor, keys: torch.Tensor, k: int) -> torch.Tensor:
    """Ground-truth top-k threshold over all keys."""
    h_kv, _, d = keys.shape
    qg = q_normal.view(h_kv, -1, d)
    scores = qg @ keys.transpose(-2, -1)
    scores = scores.reshape(q_normal.shape[0], -1)
    k = min(k, scores.shape[-1])
    th, _ = scores.topk(k, dim=-1)
    return th[:, -1]


def subspace_topk_thresholds(
    q_proj: torch.Tensor,
    keys_proj: torch.Tensor,
    topk: int,
    dim_slices: list[tuple[int, int]],
) -> torch.Tensor:
    """Per-subspace thresholds from the true full-space top-k set in index space."""
    scores = torch.einsum("hd,hnd->hn", q_proj, keys_proj)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices

    thresholds = []
    for start, end in dim_slices:
        q_sub = q_proj[:, start:end]
        keys_sub = keys_proj[:, :, start:end]
        sub_scores = torch.einsum("hd,hnd->hn", q_sub, keys_sub)
        sub_topk_scores = sub_scores.gather(1, topk_idx)
        thresholds.append(sub_topk_scores.min(dim=1).values)
    return torch.stack(thresholds, dim=0)


def exact_topk_mask(
    q_normal: torch.Tensor, keys: torch.Tensor, topk: int,
    q_head_to_kv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a per-head mask marking the exact top-k keys.

    Handles GQA: q_normal is (H_q, D), keys is (H_kv, N, D).
    Returns (H_q, N) bool mask.
    """
    if q_head_to_kv is not None:
        keys = keys[q_head_to_kv]  # (H_q, N, D)
    scores = torch.einsum("hd,hnd->hn", q_normal, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, topk_idx, True)
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf", type=int, default=4, help="Branching factor")
    parser.add_argument("--n-subspaces", type=int, default=4, help="Number of subspaces")
    parser.add_argument(
        "--subspace-strategy",
        type=str,
        default="contiguous",
        help='How to split or rotate dimensions before building subspace indexes, '
             'or "all" to verify every strategy',
    )
    parser.add_argument("--refine-iter", type=int, default=5)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--layers", type=str, default="all", help='"all" or comma-separated layer ids')
    parser.add_argument(
        "--queries",
        type=str,
        default="tail",
        choices=["tail", "all"],
        help='Sample from the tail like comparison_subspace_kcenter, or use "all"',
    )
    parser.add_argument("--n-queries", type=int, default=30, help="Queries to sample when --queries=tail")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--input-qkv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16-keys", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.input_qkv is not None:
        print(f"Loading captured QKV from {args.input_qkv} ...")
        capture = CaptureState.load(args.input_qkv)
    else:
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
    selected_layers = _parse_layers(args.layers, layer_ids)
    total_failures = 0
    total_checks = 0
    strategy_summaries: list[dict[str, str | int | float]] = []

    if args.subspace_strategy == "all":
        strategies = list(SUBSPACE_STRATEGIES)
    else:
        if args.subspace_strategy not in SUBSPACE_STRATEGIES:
            raise ValueError(
                f"Unknown subspace strategy: {args.subspace_strategy}. "
                f"Available: {', '.join(SUBSPACE_STRATEGIES)}, all"
            )
        strategies = [args.subspace_strategy]

    print(
        f"Verifying subspace_kcenter: layers={selected_layers}, "
        f"bf={args.bf}, n_subspaces={args.n_subspaces}, "
        f"strategies={strategies}, topk={args.topk}"
    )

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        strategy_failures = 0
        strategy_checks = 0

        for layer in selected_layers:
            queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)
            keys_dtype = torch.float16 if args.fp16_keys else torch.float32
            keys = keys_cpu.to(device=DEVICE, dtype=keys_dtype)
            keys_f32 = keys.float() if keys.dtype != torch.float32 else keys
            queries = queries_cpu

            h_kv, n, d = keys_f32.shape
            h_q = queries.shape[0]
            q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
            q_indices = _select_query_indices(queries.shape[1], args.queries, args.n_queries)

            print(
                f"\nLayer {layer}: H_kv={h_kv}, H_q={h_q}, N={n}, D={d}, "
                f"queries={len(q_indices)}"
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            idx = build_subspace_kcenter(
                keys_f32,
                args.bf,
                n_subspaces=args.n_subspaces,
                refine_iter=args.refine_iter,
                strategy=strategy,
            )
            torch.cuda.synchronize()
            build_ms = (time.perf_counter() - t0) * 1000
            print(f"  index build: {build_ms:.1f} ms")

            layer_failures = 0
            layer_checks = 0

            for qi in q_indices:
                q = queries[:, qi, :].to(device=DEVICE, dtype=torch.float32)
                q_norm = q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                q_normal = q / q_norm  # (H_q, D) — no remapping needed

                keys_proj = project_keys_for_index(keys_f32, idx, q_head_to_kv)
                q_proj = project_query_for_index(q_normal, idx, q_head_to_kv)

                th_per_subspace = subspace_topk_thresholds(
                    q_proj, keys_proj, args.topk, idx.dim_slices
                )
                survive = subspace_ball_gate(idx, q_normal, th_per_subspace, q_head_to_kv)
                topk_mask = exact_topk_mask(q_normal, keys_f32, args.topk, q_head_to_kv)

                missed = topk_mask & ~survive
                missed_per_head = missed.sum(dim=1)
                bad_heads = (missed_per_head > 0).nonzero(as_tuple=True)[0]

                layer_checks += q_normal.shape[0]
                if bad_heads.numel() == 0:
                    continue

                layer_failures += int(bad_heads.numel())
                for head in bad_heads.tolist():
                    missed_idx = missed[head].nonzero(as_tuple=True)[0].tolist()
                    print(
                        f"  FAIL strategy={strategy} layer={layer} query_idx={qi} head={head} "
                        f"missed_topk={len(missed_idx)} indices={missed_idx[:10]}"
                    )

            strategy_failures += layer_failures
            strategy_checks += layer_checks
            total_failures += layer_failures
            total_checks += layer_checks

            status = "PASS" if layer_failures == 0 else "FAIL"
            print(
                f"  {status}: checked {layer_checks} head/query pairs, "
                f"violations={layer_failures}"
            )

        strategy_status = "PASS" if strategy_failures == 0 else "FAIL"
        strategy_mean_violations = (
            strategy_failures / strategy_checks if strategy_checks else 0.0
        )
        strategy_summaries.append(
            {
                "strategy": strategy,
                "checks": strategy_checks,
                "violations": strategy_failures,
                "mean_violations": strategy_mean_violations,
                "status": strategy_status,
            }
        )
        print(
            f"\nStrategy {strategy}: {strategy_status} "
            f"(checked {strategy_checks}, violations={strategy_failures})"
        )

    print("\n" + "=" * 80)
    print(f"{'STRATEGY':<14s} {'CHECKS':>10s} {'VIOLATIONS':>12s} {'MEAN_VIOL':>12s} {'STATUS':>8s}")
    print("-" * 80)
    for row in strategy_summaries:
        print(
            f"{row['strategy']:<14s} {row['checks']:>10d} {row['violations']:>12d} "
            f"{row['mean_violations']:>12.6f} {row['status']:>8s}"
        )

    print("=" * 80)
    print("\n" + "=" * 80)
    if total_failures == 0:
        print(f"PASS: no missed top-k points across {total_checks} head/query checks.")
        return

    print(
        f"FAIL: found {total_failures} violating head/query checks "
        f"out of {total_checks} total."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
