"""Benchmark the incremental update path against a freshly built index.

For each flush boundary (every BUFFER_SIZE=256 decode steps) we report:

    * scanned_frac_inc      — fraction of arena keys the filter says are
                              alive after the i-th incremental update.
    * scanned_frac_fresh    — same fraction for a fully rebuilt index over
                              the exact same key set (TA_build called
                              from scratch on the cumulative keys).
    * attend_inc_ms         — ``index_inc.attend`` GPU time, repeated.
    * attend_fresh_ms       — ``index_fresh.attend`` GPU time, repeated.
    * update_kernel_ms      — last incremental-update kernel time
                              (Phase-1 GPU work, .cu cluster kernel).
    * fresh_build_ms        — wall time to TA_build the fresh index.

This isolates the quality + speed of the incremental update kernel from
the rest of the decoding pipeline. Compared to ``bench.py`` it skips
threshold sweeps, dense baselines, CSV output, and the full per-step
timing loop — only the per-flush snapshot is reported.

Usage:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.kernel_bench.bench_update \\
        --input-qkv capture.pt --n-flushes 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.index import (
    BUFFER_SIZE,
    TAIndex,
    TAIndexConfig,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _q_to_kv_map,
)


def topk_full_dot_threshold(q, keys, q_head_to_kv, topk):
    keys_eval = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def time_repeated(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def time_gpu(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def scanned_fraction(index: TAIndex, q_norm, threshold, q_head_to_kv) -> float:
    """Fraction of currently-indexed keys (= arena slots in [0, N_used)) that
    survive the filter on this query. Calls ``attend`` once to populate
    ``live_count`` in the workspace.
    """
    index.attend(q_norm, threshold, q_head_to_kv=q_head_to_kv)
    torch.cuda.synchronize()
    ws = index._ws
    if ws is None:
        return float("nan")
    n_index = int(index.state["N_used"])
    if n_index == 0:
        return float("nan")
    # In Variant A live_count covers only the index portion (filter output).
    lc = ws["live_count"]
    return float(lc.float().mean().item()) / float(n_index)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--prefill-frac", type=float, default=0.5)
    p.add_argument("--n-flushes", type=int, default=4,
                   help="Number of update flushes to perform "
                        "(decoding length = n_flushes * BUFFER_SIZE).")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    cap = CaptureState.load(args.input_qkv)
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError("Captured values required.")

    keys = keys_cpu.to(device="cuda", dtype=torch.float16)
    values = values_cpu.to(device="cuda", dtype=torch.float16)
    queries = queries_cpu.to(device="cuda", dtype=torch.float16)
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    n_decode = args.n_flushes * BUFFER_SIZE
    if n_prefill + n_decode > min(N_total, queries.shape[1]):
        raise ValueError(
            f"Need {n_prefill + n_decode} keys/queries; "
            f"capture has {min(N_total, queries.shape[1])}."
        )

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill={n_prefill}  flushes={args.n_flushes}  BUFFER={BUFFER_SIZE}")

    cfg = TAIndexConfig(n_growth=n_decode + BUFFER_SIZE)
    index_inc = TAIndex(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    _, build_inc_ms = time_gpu(lambda: index_inc.build(prefill_keys, prefill_values))
    print(f"Initial build: {build_inc_ms:.1f} ms\n")

    print(
        f"{'flush':>5} {'n_keys':>7}   "
        f"{'scan_inc':>10} {'scan_fresh':>11} {'Δscan':>8}   "
        f"{'attend_inc':>11} {'attend_fr':>10}   "
        f"{'upd_ms':>9} {'fresh_build_ms':>15}"
    )

    for flush_i in range(args.n_flushes):
        # Append BUFFER_SIZE new tokens, then update.
        for s in range(BUFFER_SIZE):
            tok = n_prefill + flush_i * BUFFER_SIZE + s
            index_inc.append_decoding_kv(
                keys[:, tok:tok + 1, :], values[:, tok:tok + 1, :]
            )

        _, upd_ms = time_gpu(index_inc.update)

        # Cumulative key count covered by the index after this flush.
        n_keys_after = n_prefill + (flush_i + 1) * BUFFER_SIZE
        cum_keys = keys[:, :n_keys_after, :].contiguous()
        cum_values = values[:, :n_keys_after, :].contiguous()

        # Fresh-built reference index over the same key set.
        index_fresh = TAIndex(TAIndexConfig(n_growth=0))
        _, fresh_build_ms = time_gpu(
            lambda: index_fresh.build(cum_keys, cum_values)
        )

        # Probe query: the next decode token (matches what bench.py would use
        # immediately after this flush).
        q_idx = n_keys_after
        q = queries[:, q_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        # Threshold from the same key set (cumulative).
        th = topk_full_dot_threshold(qn, cum_keys, q_head_to_kv, args.topk).to(
            torch.float32
        )

        scan_inc = scanned_fraction(index_inc, qn, th, q_head_to_kv)
        scan_fresh = scanned_fraction(index_fresh, qn, th, q_head_to_kv)

        attend_inc_ms = time_repeated(
            lambda: index_inc.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )
        attend_fresh_ms = time_repeated(
            lambda: index_fresh.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )

        d_scan = scan_inc - scan_fresh

        print(
            f"{flush_i+1:>5} {n_keys_after:>7}   "
            f"{scan_inc:>10.4f} {scan_fresh:>11.4f} {d_scan:>+8.4f}   "
            f"{attend_inc_ms:>11.4f} {attend_fresh_ms:>10.4f}   "
            f"{upd_ms:>9.3f} {fresh_build_ms:>15.1f}"
        )


if __name__ == "__main__":
    main()
