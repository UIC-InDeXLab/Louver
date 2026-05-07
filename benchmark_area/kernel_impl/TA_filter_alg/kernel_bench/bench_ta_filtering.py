"""Benchmark TA stage-1 filtering kernels (parent scoring + alive key mask).

Example:
~/venv/bin/python bench_ta_filtering.py \
  --input ../../../quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.filtering import (
    ta_filter_v8_0,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.TA_build import (
    build as build_v13_0,
    build_selected_clusters,
    compute_centroid_scores,
    per_key_candidate_mask,
    stop_depth_per_head,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map


def topk_full_dot_threshold(
    q: torch.Tensor,
    keys_full: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    topk: int,
) -> torch.Tensor:
    keys_eval = keys_full if q_head_to_kv is None else keys_full.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def _dense_alive_mask_for_bench(
    fn: Callable[..., object],
    q: torch.Tensor,
    th: torch.Tensor,
    state: dict,
    q_head_to_kv: torch.Tensor | None,
    n_ctx: int,
) -> torch.Tensor:
    """Dense [Hq, n_ctx] bool alive mask, materialised from whatever the
    kernel natively returns (dense or compact)."""
    out = fn(q, th, state, q_head_to_kv)
    if isinstance(out, tuple):
        # v4.0 compact: (live_idx[Hq, Npad], live_count[Hq])
        live_idx, live_count = out[0], out[1]
        h_q = int(live_idx.shape[0])
        mask = torch.zeros(h_q, n_ctx, dtype=torch.bool, device=live_idx.device)
        cnts = live_count.tolist()
        for h in range(h_q):
            c = int(cnts[h])
            if c == 0:
                continue
            idx = live_idx[h, :c].long()
            idx = idx[idx < n_ctx]
            mask[h, idx] = True
        return mask
    return out[:, :n_ctx].bool()


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=6)
    args = p.parse_args()

    cap = CaptureState.load(args.input_qkv)
    qcpu, kcpu, _vcpu = cap.to_layer_tensors(args.layer)
    keys = kcpu.to(device="cuda", dtype=torch.float32).contiguous()

    h_q = int(qcpu.shape[0])
    h_kv, n_ctx, d = keys.shape
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None

    # Build once with fixed specialization (bf=4, S=4).
    state = build_v13_0(
        keys=keys,
        values=None,
        bf=4,
        n_subspaces=4,
    )

    total_q = int(qcpu.shape[1])
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[
        : args.n_queries
    ]
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for qi in q_indices:
        q = qcpu[:, qi, :].to(device="cuda", dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = topk_full_dot_threshold(q, keys, q_head_to_kv, args.topk)
        pairs.append((q.half().contiguous(), th.float().contiguous()))

    impls = [
        ("ta_filter_v8.0", ta_filter_v8_0),
    ]

    # quick correctness check vs reference PyTorch stage-1 logic
    q_ref, th_ref = pairs[0]
    scores_h_s_k = compute_centroid_scores(
        q=q_ref.float(),
        centers_padded_f16=state["centers_padded_f16"],
        dim_slices=state["dim_slices"],
        q_head_to_kv=q_head_to_kv,
    )
    sorted_scores, order = torch.sort(scores_h_s_k, dim=-1, descending=True)
    depth = stop_depth_per_head(sorted_scores, th_ref.float())
    selected = build_selected_clusters(order, depth)
    ref_mask = per_key_candidate_mask(
        selected=selected,
        assigns_padded=state["assigns_padded"],
        q_head_to_kv=q_head_to_kv,
    )[:, :n_ctx]

    print(f"Hq={h_q} Hkv={h_kv} N={n_ctx} D={d} queries={len(pairs)} topk={args.topk}")
    print("-" * 72)
    results: dict[str, float] = {}
    keep_stats: dict[str, float] = {}
    for name, fn in impls:
        def run() -> None:
            for q, th in pairs:
                fn(q, th, state, q_head_to_kv)

        ms = time_call(run, args.iters, args.warmup) / len(pairs)
        results[name] = ms
        with torch.no_grad():
            keep = []
            for q, th in pairs[: min(8, len(pairs))]:
                mask = _dense_alive_mask_for_bench(
                    fn, q, th, state, q_head_to_kv, n_ctx
                )
                keep.append(float(mask.float().mean().item()))
            keep_stats[name] = sum(keep) / len(keep)
        eq = (
            _dense_alive_mask_for_bench(
                fn, q_ref, th_ref, state, q_head_to_kv, n_ctx
            )
            == ref_mask
        ).float().mean().item()
        print(f"{name:<16s} {ms:9.6f} ms/query  keep={keep_stats[name]:.4f}  ref_match={eq:.4f}")

    best_name = min(results, key=results.get)
    print("-" * 72)
    print(f"best={best_name} ({results[best_name]:.6f} ms/query)")
    for name in results:
        speedup = results[name] / results[best_name]
        print(f"{name:<16s} speedup_vs_best={1.0 / speedup:.3f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bench_ta_filtering.py")
    main()
