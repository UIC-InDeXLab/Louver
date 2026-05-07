"""CPU micro-benchmark for gated attention kernels.

Mirrors the GPU bench layout. Reports per-query timings for every discovered
attention_v* kernel and several CPU baselines:

    - dense_attention            : native-GQA einsum + softmax + einsum (fp32)
    - matmul_baseline            : native-GQA dot product via torch.matmul
    - sdpa_fp32                  : torch.nn.functional.scaled_dot_product_attention
    - sdpa_bf16                  : SDPA with bf16 keys/values/q

The primary CPU baseline is the fastest full-attention baseline measured in
this run. Kernels printed in green beat it.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_attention
    python -m hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_attention \
        --input-qkv /path/to/capture.pt --layer 15 --bf 4 --S 8
    python bench_attention.py --input /path/to/capture.pt --bf 4 --S 8 --buffer 256
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels import attention_kernels
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import (
        real_attention_case,
    )
except ModuleNotFoundError:
    from benchmark_area.kernel_impl.kernels.cpu_kernels import attention_kernels
    from benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build
    from benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import (
        real_attention_case,
    )

_GREEN = "\033[32m"
_RESET = "\033[0m"

# Kernels that consume bf16 q / th / buffer tensors directly. v1.4+ go in here.
BF16_ONLY_KERNELS: set[str] = {"attention_v1_4", "attention_v1_6"}


def _gqa_groups(h_q: int, h_kv: int) -> int:
    if h_q % h_kv != 0:
        raise ValueError(f"H_q={h_q} must be divisible by H_kv={h_kv}")
    return h_q // h_kv


def subspace_topk_thresholds(q, keys, topk, dim_slices, q_head_to_kv=None):
    h_q, d = q.shape
    h_kv = keys.shape[0]
    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys)
    else:
        groups = _gqa_groups(h_q, h_kv)
        q_hg = q.view(h_kv, groups, d)
        scores = torch.einsum("hgd,hnd->hgn", q_hg, keys).reshape(h_q, keys.shape[1])
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    ths = []
    for start, end in dim_slices:
        if h_q == h_kv:
            ss = torch.einsum("hd,hnd->hn", q[:, start:end], keys[:, :, start:end])
        else:
            groups = _gqa_groups(h_q, h_kv)
            q_hg = q[:, start:end].view(h_kv, groups, end - start)
            ss = torch.einsum(
                "hgd,hnd->hgn", q_hg, keys[:, :, start:end]
            ).reshape(h_q, keys.shape[1])
        ths.append(ss.gather(1, topk_idx).min(dim=1).values)
    return torch.stack(ths, dim=0).contiguous()


def dense_attention(q, keys, values, scale, q_head_to_kv=None):
    h_q, d = q.shape
    h_kv = keys.shape[0]
    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("hn,hnd->hd", probs, values)

    groups = _gqa_groups(h_q, h_kv)
    q_hg = q.view(h_kv, groups, d)
    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hgn,hnd->hgd", probs, values)
    return out.reshape(h_q, values.shape[-1])


def matmul_attention(q, keys, values, scale, q_head_to_kv=None):
    """Dense reference using matmul without materializing repeated GQA K/V."""
    h_q, d = q.shape
    h_kv = keys.shape[0]
    if h_q == h_kv:
        scores = torch.matmul(keys, q.unsqueeze(-1)).squeeze(-1) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs.unsqueeze(1), values).squeeze(1)

    groups = _gqa_groups(h_q, h_kv)
    q_hg = q.view(h_kv, groups, d)
    scores = torch.matmul(keys.unsqueeze(1), q_hg.unsqueeze(-1)).squeeze(-1) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.unsqueeze(2), values.unsqueeze(1)).squeeze(2)
    return out.reshape(h_q, values.shape[-1])


def sdpa_attention(q, keys, values, scale, q_head_to_kv=None):
    """SDPA with single-query (1 token, decode-style) — closest to our kernel."""
    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    enable_gqa = (h_q != h_kv)
    q4 = q.view(1, h_q, 1, d)
    k4 = keys.view(1, h_kv, n, d)
    v4 = values.view(1, h_kv, n, d_v)
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, scale=scale, enable_gqa=enable_gqa,
    )
    return out.view(h_q, d_v)


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def format_ms(per_q: float, baseline: float | None) -> str:
    text = f"{per_q:9.3f}"
    if baseline is not None and per_q < baseline:
        return f"{_GREEN}{text}{_RESET}"
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, default=None)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--n-queries", type=int, default=16)
    p.add_argument(
        "--buffer-len", "--buffer", dest="buffer_len", type=int, default=0,
        help="Hold the final buffer-len rows out as the decoding buffer.",
    )
    p.add_argument(
        "--only", type=str, default=None,
        help="Comma-separated attention_v* module names to benchmark.",
    )
    p.add_argument("--refine-iter", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--threads", type=int, default=os.cpu_count(),
                   help="torch.set_num_threads override (default: all logical CPUs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    torch_threads = torch.get_num_threads()

    if args.input_qkv is not None:
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
        print(f"Loading capture from {args.input_qkv} ...")
        print(
            f"Using layer {real['layer']} with N_idx={keys.shape[1]}, "
            f"N_buf={buffer_keys.shape[1]}, queries={len(real['q_indices'])}"
        )
    else:
        n = 4096 if args.N is None else args.N
        torch.manual_seed(args.seed)
        keys = torch.randn(args.H, n, args.D)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        values = torch.randn(args.H, n, args.D)
        keys_eval = keys
        values_eval = values
        buffer_keys = None
        buffer_values = None
        q_batch = torch.randn(args.n_queries, args.H, args.D)
        q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        q_head_to_kv = None

    scale = 1.0 / math.sqrt(keys.shape[-1])
    state = build(keys, args.bf, args.S, args.refine_iter, values=values)

    pairs_fp32 = [
        (
            q.contiguous(),
            subspace_topk_thresholds(
                q, keys_eval, args.topk, state["dim_slices"],
                q_head_to_kv=q_head_to_kv,
            ),
        )
        for q in q_batch
    ]
    loose_pair_fp32 = (pairs_fp32[0][0], torch.full_like(pairs_fp32[0][1], -1.0e30))

    # bf16 inputs (q, threshold packed with q_norms — same packing as GPU v5.14)
    pairs_bf16: list[tuple[torch.Tensor, torch.Tensor]] = []
    for q, th in pairs_fp32:
        q_norms = torch.stack(
            [q[:, s:e].norm(dim=-1) for s, e in state["dim_slices"]], dim=0,
        )
        th_packed = torch.cat([th, q_norms], dim=0).to(torch.bfloat16)
        pairs_bf16.append((q.to(torch.bfloat16), th_packed))

    h_kv, n, d = keys.shape
    h_q = q_batch.shape[1]
    n_buf = 0 if buffer_keys is None else buffer_keys.shape[1]
    print(
        f"cpu attention micro-bench: H_q={h_q} H_kv={h_kv} N_idx={n} "
        f"N_buf={n_buf} D={d} bf={args.bf} S={args.S} threads={torch_threads} "
        f"queries={len(pairs_fp32)}"
    )
    print("-" * 78)

    # --- Baselines first so we know the green target before kernel rows print ---
    keys_bf16 = keys_eval.to(torch.bfloat16).contiguous()
    values_bf16 = values_eval.to(torch.bfloat16).contiguous()
    q_bf16_batch = [q.to(torch.bfloat16).contiguous() for q, _ in pairs_fp32]

    baseline_results: dict[str, float] = {}

    def bench_baseline(label: str, fn):
        ms = time_call(fn, args.iters, args.warmup) / len(pairs_fp32)
        baseline_results[label] = ms
        return ms

    def dense_run():
        for q, _ in pairs_fp32:
            dense_attention(q, keys_eval, values_eval, scale, q_head_to_kv)

    def matmul_run():
        for q, _ in pairs_fp32:
            matmul_attention(q, keys_eval, values_eval, scale, q_head_to_kv)

    def sdpa_fp32_run():
        for q, _ in pairs_fp32:
            sdpa_attention(q, keys_eval, values_eval, scale, q_head_to_kv)

    def sdpa_bf16_run():
        for q in q_bf16_batch:
            sdpa_attention(q, keys_bf16, values_bf16, scale, q_head_to_kv)

    bench_baseline("dense_attention", dense_run)
    bench_baseline("matmul_attention", matmul_run)
    bench_baseline("sdpa_fp32", sdpa_fp32_run)
    bench_baseline("sdpa_bf16", sdpa_bf16_run)

    primary_label, primary_baseline = min(baseline_results.items(), key=lambda kv: kv[1])

    # --- Our kernels ---
    results = []
    only = None
    if args.only:
        only = {item.strip() for item in args.only.split(",") if item.strip()}

    for name, info in sorted(attention_kernels().items()):
        if only is not None and name not in only:
            continue
        if name in BF16_ONLY_KERNELS:
            pairs_used = pairs_bf16
            keys_used = keys_eval.to(torch.bfloat16).contiguous()
            buf_keys_used = (
                buffer_keys.to(torch.bfloat16).contiguous() if buffer_keys is not None else None
            )
            buf_values_used = (
                buffer_values.to(torch.bfloat16).contiguous() if buffer_values is not None else None
            )
        else:
            pairs_used = pairs_fp32
            keys_used = keys
            buf_keys_used = buffer_keys
            buf_values_used = buffer_values

        def run_tight(_pairs=pairs_used, _info=info, _ku=keys_used,
                      _bk=buf_keys_used, _bv=buf_values_used):
            for q, th in _pairs:
                _info.fn(
                    q, th, state, _bk, _bv, _ku,
                    q_head_to_kv=q_head_to_kv, scale=scale,
                )

        ms = time_call(run_tight, args.iters, args.warmup) / len(pairs_used)
        results.append((name, ms))
        time_text = format_ms(ms, primary_baseline)
        print(f"  {name:<24s} {info.version:<10s} {time_text} ms/query")

        # Loose-gate correctness check vs dense fp32 reference.
        if name in BF16_ONLY_KERNELS:
            q0 = pairs_used[0][0]
            th0 = pairs_used[0][1]
            th_loose = th0.clone()
            n_subspaces = len(state["dim_slices"])
            th_loose[:n_subspaces] = float(torch.finfo(torch.bfloat16).min)
        else:
            q0, th0 = loose_pair_fp32
            th_loose = th0
        out = info.fn(
            q0, th_loose, state, buf_keys_used, buf_values_used, keys_used,
            q_head_to_kv=q_head_to_kv, scale=scale,
        )
        ref = dense_attention(
            pairs_fp32[0][0], keys_eval, values_eval, scale, q_head_to_kv,
        )
        diff = (out.float() - ref).abs().max().item()
        print(f"  {'  loose vs dense fp32':<24s} {'-':<10s} max_abs_diff={diff:.4e}")

    # --- Print baselines after kernels for easy comparison ---
    print("-" * 78)
    print(f"CPU baselines (PRIMARY = {primary_label}):")
    for label, ms in baseline_results.items():
        tag = " (PRIMARY)" if label == primary_label else ""
        print(f"  {label + tag:<24s} {'-':<10s} {ms:9.3f} ms/query")

    print("-" * 78)
    all_results = results + list(baseline_results.items())
    best = min(all_results, key=lambda x: x[1])
    print(f"Fastest overall: {best[0]} at {best[1]:.3f} ms/query")
    if results:
        best_kernel = min(results, key=lambda x: x[1])
        speedup = primary_baseline / best_kernel[1]
        print(
            f"Fastest HIRA kernel: {best_kernel[0]} at {best_kernel[1]:.3f} ms/query "
            f"({speedup:.2f}x vs {primary_label})"
        )


if __name__ == "__main__":
    main()
