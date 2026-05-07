#!/usr/bin/env python3
"""
Observe which orthants (sign-pattern sections of R^d) benchmark queries occupy.

Each query vector q in R^d is mapped to one of 2^d orthants based on per-axis
sign (+/-). We bucket all captured queries by orthant and report concentration
statistics to show whether queries cluster in a small subset of orthants.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import _capture_qkv, _q_to_kv_map
from method_comparison_bench import DEVICE, DTYPE, LAYER_IDX, MODEL_NAME, PROMPT


def _orthant_key(q: torch.Tensor) -> str:
    """
    Orthant key as a hex string from sign bits:
      bit=1 for q_i >= 0, bit=0 for q_i < 0
    """
    bits = (q >= 0).to(torch.int32).tolist()
    key_int = 0
    for b in bits:
        key_int = (key_int << 1) | int(b)
    hex_len = (len(bits) + 3) // 4
    return f"{key_int:0{hex_len}x}"


def _counter_stats(counter: Counter[str], total: int) -> dict[str, float]:
    if total <= 0:
        return {
            "unique": 0.0,
            "top1_share": 0.0,
            "top5_share": 0.0,
            "top10_share": 0.0,
            "entropy_bits": 0.0,
            "effective_orthants": 0.0,
            "hhi_inverse": 0.0,
        }

    counts = [v for _, v in counter.most_common()]
    probs = [c / total for c in counts]

    entropy_bits = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
    hhi = sum(p * p for p in probs)

    return {
        "unique": float(len(counter)),
        "top1_share": float(sum(counts[:1]) / total),
        "top5_share": float(sum(counts[:5]) / total),
        "top10_share": float(sum(counts[:10]) / total),
        "entropy_bits": float(entropy_bits),
        "effective_orthants": float(2.0**entropy_bits),
        "hhi_inverse": float(1.0 / max(hhi, 1e-12)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    p.add_argument("--n-tokens", type=int, default=600)
    p.add_argument("--top-k", type=int, default=20, help="How many top orthants to print.")
    p.add_argument("--device", type=str, default=DEVICE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError('This observer is CUDA-only. Use "--device cuda".')

    print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
    t0 = time.perf_counter()
    capture = _capture_qkv(
        model_name=args.model,
        prompt_text=PROMPT,
        n=args.n_tokens,
        device=args.device,
        torch_dtype=DTYPE,
        show_progress=True,
    )
    print(f"Capture done in {time.perf_counter() - t0:.1f}s")

    layer_ids = capture.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, _ = capture.to_layer_tensors(layer)

    # Analysis is CPU-side to simplify counting/aggregation.
    queries = queries_cpu.to(dtype=torch.float32, device="cpu")
    keys = keys_cpu.to(dtype=torch.float32, device="cpu")

    h_q, t_q, _ = queries.shape
    h_kv, t_k, _ = keys.shape
    prompt_len = int(capture.prompt_length or 0)
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cpu") if h_q != h_kv else None

    print(
        f"Layer {layer}: H_q={h_q}, H_kv={h_kv}, T_q={t_q}, T_k={t_k}, "
        f"prompt_len={prompt_len}, total_queries={h_q * t_q}, dim={queries.shape[-1]}"
    )
    if q_head_to_kv is not None:
        print(f"GQA mapping active: q_head_to_kv shape={tuple(q_head_to_kv.shape)}")
    print("-" * 100)

    global_counter: Counter[str] = Counter()
    per_head_counters = [Counter() for _ in range(h_q)]
    zero_coords = 0
    total_coords = 0

    for h in range(h_q):
        for t in range(t_q):
            q = queries[h, t, :]
            zero_coords += int((q == 0).sum().item())
            total_coords += int(q.numel())

            key = _orthant_key(q)
            global_counter[key] += 1
            per_head_counters[h][key] += 1

    stats = _counter_stats(global_counter, total=h_q * t_q)

    d = queries.shape[-1]
    print("Global Orthant Concentration")
    print("-" * 100)
    print(f"Possible orthants in R^{d}: 2^{d}")
    print(f"Observed unique orthants: {int(stats['unique'])}")
    print(f"Top-1 share:  {stats['top1_share']:.4f}")
    print(f"Top-5 share:  {stats['top5_share']:.4f}")
    print(f"Top-10 share: {stats['top10_share']:.4f}")
    print(f"Entropy (bits): {stats['entropy_bits']:.4f}")
    print(f"Effective orthants (2^entropy): {stats['effective_orthants']:.2f}")
    print(f"Inverse HHI (effective count): {stats['hhi_inverse']:.2f}")
    print(f"Zero-coordinate ratio: {zero_coords / max(1, total_coords):.8f}")

    top_k = max(1, args.top_k)
    print("\nTop Orthants (Global)")
    print("-" * 100)
    print(f"{'RANK':<6s} {'COUNT':>10s} {'FRACTION':>10s} {'ORTHANT_KEY_HEX':<40s}")
    for i, (k, c) in enumerate(global_counter.most_common(top_k), start=1):
        frac = c / (h_q * t_q)
        print(f"{i:<6d} {c:>10d} {frac:>10.4f} {k:<40s}")

    print("\nPer-Head Concentration")
    print("-" * 100)
    print(f"{'HEAD':<6s} {'UNIQUE':>8s} {'TOP1':>8s} {'TOP5':>8s} {'ENTROPY':>10s}")
    for h in range(h_q):
        hs = _counter_stats(per_head_counters[h], total=t_q)
        print(
            f"{h:<6d} {int(hs['unique']):>8d} {hs['top1_share']:>8.4f} "
            f"{hs['top5_share']:>8.4f} {hs['entropy_bits']:>10.4f}"
        )


if __name__ == "__main__":
    main()
