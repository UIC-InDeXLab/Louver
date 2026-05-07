#!/usr/bin/env python3
"""
Experiment: derive per-subspace top-k thresholds and test an OR gate.

For each selected layer / query / head:
- split dimensions into 8 subspaces (contiguous, PQ-style)
- find the exact top-k keys in full space
- for each subspace i, compute t_i as the minimum projected dot product
  among those top-k keys
- filter all keys with an OR gate: keep a key if it passes any subspace
  threshold q_i · x_i >= t_i

Set notation used in reports:
- N: all keys
- T: exact full-space top-k set
- H: full-space halfspace at the top-k threshold
- O: filter output set

Reports:
- |O \\ T|: kept points outside the top-k set
- |O \\ H|: kept points outside the full-space halfspace
- |T \\ O|: true top-k points missed by the OR gate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import CaptureState, _capture_qkv, _q_to_kv_map


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
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            selected.append(int(part))
    missing = sorted(set(selected) - set(available_layers))
    if missing:
        raise ValueError(f"Requested layers not present in capture: {missing}")
    return sorted(selected)


def _select_query_indices(total_q: int, mode: str, n_queries: int) -> list[int]:
    if mode == "all":
        return list(range(total_q))
    stride = max(1, total_q // max(1, n_queries))
    q_indices = list(
        range(total_q - 1, max(0, total_q - n_queries * stride) - 1, -stride)
    )
    return q_indices[:n_queries]


def _split_contiguous(d: int, n_subspaces: int) -> list[tuple[int, int]]:
    sub_dim = d // n_subspaces
    remainder = d % n_subspaces
    slices = []
    offset = 0
    for s in range(n_subspaces):
        width = sub_dim + (1 if s < remainder else 0)
        slices.append((offset, offset + width))
        offset += width
    return slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--n-subspaces", type=int, default=8)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--queries", choices=["tail", "all"], default="tail")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--n-tokens", type=int, default=2000)
    parser.add_argument("--input-qkv", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16-keys", action="store_true")
    parser.add_argument(
        "--max-report",
        type=int,
        default=10,
        help="How many worst offending head/query cases to print at the end",
    )
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

    global_cases = 0
    global_extras = 0
    global_extras_inside = 0
    global_extras_outside = 0
    global_missed_topk = 0
    global_with_any_extra = 0
    global_with_any_miss = 0
    worst_cases: list[dict[str, float | int]] = []

    for layer in selected_layers:
        queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)
        keys_dtype = torch.float16 if args.fp16_keys else torch.float32
        keys = keys_cpu.to(device=DEVICE, dtype=keys_dtype)
        keys_f32 = keys.float() if keys.dtype != torch.float32 else keys
        queries = queries_cpu

        h_kv, n, d = keys_f32.shape
        h_q = queries.shape[0]
        q_head_to_kv = _q_to_kv_map(h_q, h_kv, DEVICE) if h_q != h_kv else None
        h_eval = h_q if q_head_to_kv is not None else h_kv
        q_indices = _select_query_indices(queries.shape[1], args.queries, args.n_queries)
        dim_slices = _split_contiguous(d, args.n_subspaces)

        layer_cases = 0
        layer_extras = 0
        layer_extras_inside = 0
        layer_extras_outside = 0
        layer_missed_topk = 0
        layer_with_any_extra = 0
        layer_with_any_miss = 0

        print(
            f"\nLayer {layer}: H_kv={h_kv}, H_q={h_q}, H_eval={h_eval}, "
            f"N={n}, D={d}, queries={len(q_indices)}, topk={args.topk}, "
            f"subspaces={args.n_subspaces}"
        )

        for qi in q_indices:
            q = queries[:, qi, :].to(device=DEVICE, dtype=torch.float32)
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            if q_head_to_kv is not None:
                q_eval = q
                keys_eval = keys_f32[q_head_to_kv]
            else:
                q_eval = q
                keys_eval = keys_f32

            scores = torch.einsum("hd,hnd->hn", q_eval, keys_eval)
            k = min(args.topk, n)
            topk_scores, topk_idx = scores.topk(k, dim=1)
            thresholds = topk_scores[:, -1]

            topk_mask = torch.zeros_like(scores, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)
            halfspace_mask = scores >= thresholds.unsqueeze(-1)

            subspace_passes = []
            t_values = []
            for start, end in dim_slices:
                q_sub = q_eval[:, start:end]
                keys_sub = keys_eval[:, :, start:end]
                sub_scores = torch.einsum("hd,hnd->hn", q_sub, keys_sub)
                topk_sub_scores = sub_scores.gather(1, topk_idx)
                t_i = topk_sub_scores.min(dim=1).values
                t_values.append(t_i)
                subspace_passes.append(sub_scores >= t_i.unsqueeze(-1))

            keep_mask = torch.stack(subspace_passes, dim=0).any(dim=0)

            extra_mask = keep_mask & ~topk_mask
            extra_inside_mask = extra_mask & halfspace_mask
            extra_outside_mask = extra_mask & ~halfspace_mask
            missed_topk_mask = topk_mask & ~keep_mask

            extra_per_head = extra_mask.sum(dim=1)
            extra_inside_per_head = extra_inside_mask.sum(dim=1)
            extra_outside_per_head = extra_outside_mask.sum(dim=1)
            missed_topk_per_head = missed_topk_mask.sum(dim=1)

            for head in range(h_eval):
                et = int(extra_per_head[head].item())
                ei = int(extra_inside_per_head[head].item())
                eo = int(extra_outside_per_head[head].item())
                mt = int(missed_topk_per_head[head].item())

                layer_cases += 1
                layer_extras += et
                layer_extras_inside += ei
                layer_extras_outside += eo
                layer_missed_topk += mt

                if et > 0:
                    layer_with_any_extra += 1
                if mt > 0:
                    layer_with_any_miss += 1

                if et > 0 or mt > 0:
                    t_min = min(float(t[head].item()) for t in t_values)
                    t_max = max(float(t[head].item()) for t in t_values)
                    worst_cases.append(
                        {
                            "layer": layer,
                            "query_idx": qi,
                            "head": head,
                            "extra_total": et,
                            "extra_inside": ei,
                            "extra_outside": eo,
                            "missed_topk": mt,
                            "t_min": t_min,
                            "t_max": t_max,
                        }
                    )

        global_cases += layer_cases
        global_extras += layer_extras
        global_extras_inside += layer_extras_inside
        global_extras_outside += layer_extras_outside
        global_missed_topk += layer_missed_topk
        global_with_any_extra += layer_with_any_extra
        global_with_any_miss += layer_with_any_miss

        print(f"  cases:                       {layer_cases}")
        print(f"  cases with |O \\\\ T| > 0:      {layer_with_any_extra} ({layer_with_any_extra / max(1, layer_cases):.2%})")
        print(f"  cases with |T \\\\ O| > 0:      {layer_with_any_miss} ({layer_with_any_miss / max(1, layer_cases):.2%})")
        print(f"  avg |O \\\\ T| / case:          {layer_extras / max(1, layer_cases):.3f}")
        print(f"  avg |(O ∩ H) \\\\ T| / case:    {layer_extras_inside / max(1, layer_cases):.3f}")
        print(f"  avg |O \\\\ H| / case:          {layer_extras_outside / max(1, layer_cases):.3f}")
        print(f"  avg |T \\\\ O| / case:          {layer_missed_topk / max(1, layer_cases):.3f}")
        print(f"  total |O \\\\ T|:               {layer_extras}")
        print(f"  total |(O ∩ H) \\\\ T|:         {layer_extras_inside}")
        print(f"  total |O \\\\ H|:               {layer_extras_outside}")
        print(f"  total |T \\\\ O|:               {layer_missed_topk}")

    print("\n" + "=" * 90)
    print("Global Summary")
    print(f"cases:                         {global_cases}")
    print(f"cases with |O \\\\ T| > 0:        {global_with_any_extra} ({global_with_any_extra / max(1, global_cases):.2%})")
    print(f"cases with |T \\\\ O| > 0:        {global_with_any_miss} ({global_with_any_miss / max(1, global_cases):.2%})")
    print(f"avg |O \\\\ T| / case:            {global_extras / max(1, global_cases):.3f}")
    print(f"avg |(O ∩ H) \\\\ T| / case:      {global_extras_inside / max(1, global_cases):.3f}")
    print(f"avg |O \\\\ H| / case:            {global_extras_outside / max(1, global_cases):.3f}")
    print(f"avg |T \\\\ O| / case:            {global_missed_topk / max(1, global_cases):.3f}")
    print(f"total |O \\\\ T|:                 {global_extras}")
    print(f"total |(O ∩ H) \\\\ T|:           {global_extras_inside}")
    print(f"total |O \\\\ H|:                 {global_extras_outside}")
    print(f"total |T \\\\ O|:                 {global_missed_topk}")

    if worst_cases:
        worst_cases.sort(
            key=lambda row: (
                int(row["missed_topk"]),
                int(row["extra_outside"]),
                int(row["extra_total"]),
            ),
            reverse=True,
        )
        print("\nWorst Cases")
        for row in worst_cases[: args.max_report]:
            print(
                f"layer={row['layer']} query={row['query_idx']} head={row['head']} "
                f"|O\\T|={row['extra_total']} |(O∩H)\\T|={row['extra_inside']} "
                f"|O\\H|={row['extra_outside']} |T\\O|={row['missed_topk']} "
                f"t_min={row['t_min']:.5f} t_max={row['t_max']:.5f}"
            )


if __name__ == "__main__":
    main()
