#!/usr/bin/env python3
"""
Simple benchmark: full-build CPU/CUDA indexers on real keys (n=10000).

Usage:
    python fullbuild_bench.py [--output fullbuild_results.csv]
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from itertools import product
from pathlib import Path

import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.indexer.cpu import CPUIndexer
from hira.indexer.cuda import CUDAIndexer
from hira.searcher.cpu import CPUSearcher
from hira.searcher.cuda import CUDASearcher

from pruning_bench_utils import (
    _capture_qkv,
    _q_to_kv_map,
)

# ===================== PARAMETERS =====================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER_IDX = 15
MAX_ITERATIONS = 10
N_TOKENS = 2000
TOPK = 20

BRANCHING_FACTORS = [2, 4, 8, 16, 32]
CPU_LEVELS = [2]
CUDA_LEVELS = [2]

QUERY_SAMPLE = 50
QUERY_STRIDE = 50
SEARCH_REPEATS = 3

DEVICE = "cuda"
DTYPE = torch.float32


# ===================== HELPERS =====================


def _safe_cuda_empty():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _topk_threshold_cpu(q_normal, indexer_keys, k=TOPK):
    children = indexer_keys  # (H_kv, N, D)
    qg = q_normal.view(children.shape[0], -1, children.shape[-1])
    w = qg @ children.transpose(-2, -1)
    w = w.reshape(q_normal.shape[0], -1)
    k = min(k, w.shape[-1])
    th, _ = w.topk(k, dim=-1)
    return th[:, -1]


def cpu_scanned_fraction(query_cpu, threshold_cpu, indexer, q_head_to_kv_cpu):
    depth = len(indexer.levels)
    bf = indexer.branching_factor

    if depth <= 1:
        return 1.0

    top = depth - 1
    top_level = indexer.levels[top]
    centers = top_level.ball_centers[q_head_to_kv_cpu]
    radii = top_level.ball_radii[q_head_to_kv_cpu]
    scores = torch.einsum("hnd,hd->hn", centers, query_cpu)
    cascaded_pass = (scores + radii) > threshold_cpu.unsqueeze(-1)

    for lvl_idx in range(top - 1, 0, -1):
        lvl = indexer.levels[lvl_idx]
        c = lvl.ball_centers[q_head_to_kv_cpu]
        r = lvl.ball_radii[q_head_to_kv_cpu]

        # Use actual child2parent mapping from k-means assignment
        # lvl.child2parent: (H_kv, num_children_at_lvl) -> parent index at lvl+1
        c2p = lvl.child2parent[q_head_to_kv_cpu]  # (H_q, num_children)
        parent_pass = torch.gather(cascaded_pass, 1, c2p)

        s = torch.einsum("hnd,hd->hn", c, query_cpu)
        cascaded_pass = parent_pass & ((s + r) > threshold_cpu.unsqueeze(-1))

    total = indexer.levels[0].size
    scanned_per_head = cascaded_pass.sum(dim=1).long() * bf
    return float(scanned_per_head.float().mean().item()) / max(1, total)


# ===================== EVALUATION =====================


def evaluate_cpu_fullbuild(
    keys_cpu, queries_cuda, q_indices, q_head_to_kv_cpu, num_levels, bf
):
    H_kv, N, D = keys_cpu.shape
    if bf ** (num_levels - 1) > N:
        return None

    try:
        t0 = time.perf_counter()
        indexer = CPUIndexer(
            num_levels=num_levels,
            branching_factor=bf,
            max_iterations=MAX_ITERATIONS,
        ).build(keys_cpu)
        build_time = time.perf_counter() - t0

        searcher = CPUSearcher(search_strategy="fused_v3")
        fracs, search_times = [], []

        for qi in tqdm(q_indices, desc=f"  CPU L={num_levels} bf={bf}", leave=False):
            q = queries_cuda[:, qi, :].cpu().float()
            q_norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
            q_normal = q / q_norm

            th = _topk_threshold_cpu(q_normal, indexer.keys)
            fracs.append(cpu_scanned_fraction(q_normal, th, indexer, q_head_to_kv_cpu))

            q_4d = q_normal.unsqueeze(0).unsqueeze(2)
            for _ in range(SEARCH_REPEATS):
                t0 = time.perf_counter()
                searcher.search(query=q_4d, threshold=th, indexer=indexer)
                search_times.append(time.perf_counter() - t0)

        return {
            "mean_frac": sum(fracs) / len(fracs),
            "min_frac": min(fracs),
            "max_frac": max(fracs),
            "search_ms": (sum(search_times) / len(search_times)) * 1000,
            "build_ms": build_time * 1000,
            "actual_depth": len(indexer.levels),
            "level_sizes": str([lvl.size for lvl in indexer.levels]),
        }
    except Exception as e:
        return {"error": str(e)[:200]}


def evaluate_cuda_fullbuild(
    keys_cuda, queries_cuda, q_indices, q_head_to_kv_cuda, num_levels, bf
):
    H_kv, N, D = keys_cuda.shape
    if bf ** (num_levels - 1) > N:
        return None

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        indexer = CUDAIndexer(
            num_levels=num_levels,
            branching_factor=bf,
            max_iterations=MAX_ITERATIONS,
        ).build(keys_cuda)
        torch.cuda.synchronize()
        build_time = time.perf_counter() - t0

        searcher = CUDASearcher(block_c=bf)
        fracs, search_times = [], []

        for qi in tqdm(q_indices, desc=f"  CUDA L={num_levels} bf={bf}", leave=False):
            q = queries_cuda[:, qi, :]
            q_norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
            q_normal = q / q_norm

            th = _topk_threshold_cpu(q_normal, keys_cuda)
            stats = searcher.synthetic_scanned_fraction(
                query=q_normal,
                threshold=th,
                indexer=indexer,
                q_head_to_kv=q_head_to_kv_cuda,
            )
            fracs.append(stats["scanned_fraction_mean"])

            for _ in range(SEARCH_REPEATS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                searcher.search(
                    query=q_normal,
                    threshold=th,
                    indexer=indexer,
                    q_head_to_kv=q_head_to_kv_cuda,
                )
                torch.cuda.synchronize()
                search_times.append(time.perf_counter() - t0)

        n_children = indexer.children.shape[1] if indexer.children is not None else 0
        n_parents = indexer.parents.shape[1] if indexer.parents is not None else 0
        n_gp = (
            indexer.grand_parents.shape[1] if indexer.grand_parents is not None else 0
        )
        level_sizes = [n_children, n_parents] + ([n_gp] if num_levels == 3 else [])

        return {
            "mean_frac": sum(fracs) / len(fracs),
            "min_frac": min(fracs),
            "max_frac": max(fracs),
            "search_ms": (sum(search_times) / len(search_times)) * 1000,
            "build_ms": build_time * 1000,
            "actual_depth": num_levels,
            "level_sizes": str(level_sizes),
        }
    except Exception as e:
        _safe_cuda_empty()
        return {"error": str(e)[:200]}


# ===================== MAIN =====================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="fullbuild_results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for model inference.")

    # Capture QKV — reasoning prompt to force long generation
    prompt_text = (
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
    print(f"Capturing {N_TOKENS} tokens from {MODEL_NAME} …")
    t0 = time.perf_counter()
    capture = _capture_qkv(
        model_name=MODEL_NAME,
        prompt_text=prompt_text,
        n=N_TOKENS,
        device=DEVICE,
        torch_dtype=DTYPE,
        show_progress=True,
    )
    print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    layer_ids = capture.layer_ids()
    layer = LAYER_IDX if LAYER_IDX in layer_ids else layer_ids[len(layer_ids) // 2]
    print(f"Using layer {layer}")

    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)
    prompt_len = capture.prompt_length
    num_q_heads = queries_cpu.shape[0]
    num_kv_heads = keys_cpu.shape[0]

    keys_cuda = keys_cpu.to(device=DEVICE, dtype=torch.float32)
    keys_cpu_f32 = keys_cpu.to(dtype=torch.float32)
    queries_cuda = queries_cpu.to(device=DEVICE, dtype=torch.float32)

    total_keys = keys_cuda.shape[1]
    total_queries = queries_cuda.shape[1]
    print(
        f"prompt_len={prompt_len}  total_keys={total_keys}  "
        f"total_queries={total_queries}  H_q={num_q_heads}  H_kv={num_kv_heads}\n"
    )

    # Query indices near the end
    end = total_queries
    start = max(0, end - QUERY_SAMPLE * QUERY_STRIDE)
    q_indices = list(range(start, end, QUERY_STRIDE))[:QUERY_SAMPLE]

    q_head_to_kv_cuda = _q_to_kv_map(num_q_heads, num_kv_heads, DEVICE)
    q_head_to_kv_cpu = _q_to_kv_map(num_q_heads, num_kv_heads, "cpu")

    # Build config list: (indexer_type, num_levels, branching_factor)
    configs = []
    for bf in BRANCHING_FACTORS:
        for nl in CPU_LEVELS:
            configs.append(("cpu", nl, bf))
        for nl in CUDA_LEVELS:
            configs.append(("cuda", nl, bf))

    csv_fields = [
        "indexer",
        "num_levels",
        "branching_factor",
        "mean_frac",
        "min_frac",
        "max_frac",
        "search_ms",
        "build_ms",
        "actual_depth",
        "level_sizes",
        "error",
    ]

    out_path = Path(args.output)
    print(f"Writing results to {out_path}")
    print(f"Total configs: {len(configs)}\n")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for indexer_type, nl, bf in tqdm(configs, desc="Sweep"):
            row = {
                "indexer": indexer_type,
                "num_levels": nl,
                "branching_factor": bf,
                "error": "",
            }

            if indexer_type == "cpu":
                res = evaluate_cpu_fullbuild(
                    keys_cpu_f32,
                    queries_cuda,
                    q_indices,
                    q_head_to_kv_cpu,
                    nl,
                    bf,
                )
            else:
                res = evaluate_cuda_fullbuild(
                    keys_cuda,
                    queries_cuda,
                    q_indices,
                    q_head_to_kv_cuda,
                    nl,
                    bf,
                )

            if res is None:
                row["error"] = "infeasible"
            elif "error" in res:
                row["error"] = res["error"]
            else:
                row.update(res)

            writer.writerow(row)
            f.flush()

            label = f"{indexer_type:>4s}  lvl={nl}  bf={bf:>2d}"
            if res and "error" not in res:
                tqdm.write(
                    f"  {label}  frac={res['mean_frac']:.4f}  "
                    f"search={res['search_ms']:.2f}ms  build={res['build_ms']:.2f}ms"
                )
            else:
                err = res.get("error", "infeasible") if res else "infeasible"
                tqdm.write(f"  {label}  ERROR: {err}")

            _safe_cuda_empty()

    print(f"\nDone. Results saved to {out_path}")


if __name__ == "__main__":
    main()
