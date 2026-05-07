"""CPU micro-benchmark for build kernels.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_build
    python benchmark_area/kernel_impl/kernels/cpu_kernels/kernel_bench/bench_build.py
"""

from __future__ import annotations

import argparse
import math
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
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels import build_kernels
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import real_build_case
except ModuleNotFoundError:
    from benchmark_area.kernel_impl.kernels.cpu_kernels import build_kernels
    from benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import real_build_case


def torch_naive_build(keys, bf, n_subspaces, refine_iter):
    h, n, d = keys.shape
    k = max(1, math.ceil(n / bf))
    sub = d // n_subspaces
    out = []
    for s in range(n_subspaces):
        start = s * sub
        end = start + sub if s < n_subspaces - 1 else d
        ks = keys[:, :, start:end].contiguous()
        width = end - start
        centers = ks[:, torch.linspace(0, n - 1, k).long()].contiguous()
        for _ in range(refine_iter):
            dists = torch.cdist(ks, centers)
            assign = dists.argmin(dim=2)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(1, assign[..., None].expand(-1, -1, width), ks)
            cnt = torch.zeros(h, k, dtype=keys.dtype)
            cnt.scatter_add_(1, assign, torch.ones(h, n, dtype=keys.dtype))
            centers = new_centers / cnt.clamp_min(1.0).unsqueeze(-1)
        assign = torch.cdist(ks, centers).argmin(dim=2)
        parent = centers.gather(1, assign[..., None].expand(-1, -1, width))
        dd = (ks - parent).norm(dim=-1)
        radii = torch.zeros(h, k, dtype=keys.dtype)
        radii.scatter_reduce_(1, assign, dd, reduce="amax", include_self=True)
        out.append((assign, centers, radii))
    return out


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=2)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.input_qkv is not None:
        real = real_build_case(args.input_qkv, args.layer, args.N)
        keys = real["keys"]
        print(f"Loading capture from {args.input_qkv} ...")
        print(
            f"Using layer {real['layer']} with N={keys.shape[1]} "
            f"of {real['total_keys']} captured keys"
        )
    else:
        n = 4096 if args.N is None else args.N
        torch.manual_seed(args.seed)
        keys = torch.randn(args.H, n, args.D, dtype=torch.float32)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    h, n, d = keys.shape
    print(f"cpu build micro-bench: H={h} N={n} D={d} bf={args.bf} S={args.S}")
    print("-" * 72)

    results = []
    for name, info in sorted(build_kernels().items()):
        ms = time_call(
            lambda: info.fn(keys, args.bf, args.S, args.refine_iter),
            args.iters,
            args.warmup,
        )
        results.append((name, ms))
        print(f"  {name:<22s} {info.version:<10s} {ms:9.2f} ms")

    ms = time_call(
        lambda: torch_naive_build(keys, args.bf, args.S, args.refine_iter),
        max(1, min(args.iters, 2)),
        args.warmup,
    )
    results.append(("torch_naive", ms))
    print(f"  {'torch_naive':<22s} {'-':<10s} {ms:9.2f} ms")
    print("-" * 72)
    best = min(results, key=lambda x: x[1])
    print(f"Fastest: {best[0]} at {best[1]:.2f} ms")


if __name__ == "__main__":
    main()
