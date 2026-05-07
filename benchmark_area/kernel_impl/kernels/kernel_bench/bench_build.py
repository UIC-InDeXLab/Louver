"""Micro-benchmark: compare all build_v* kernels on synthetic keys.

Also runs a naive torch baseline that loops in Python over subspaces for
reference (same algorithm, no auto-dispatch).

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_build
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels import build_kernels


def torch_naive_build(keys, bf, n_subspaces, refine_iter):
    """Unoptimized python-loop torch reference."""
    H, N, D = keys.shape
    K = max(1, math.ceil(N / bf))
    sub = D // n_subspaces
    out_assigns, out_centers, out_radii = [], [], []

    for s in range(n_subspaces):
        start = s * sub
        end = start + sub if s < n_subspaces - 1 else D
        ks = keys[:, :, start:end].contiguous()
        d = end - start

        # Seeding: pick random points
        idx = torch.randint(0, N, (H, K), device=keys.device)
        centers = ks.gather(1, idx[..., None].expand(-1, -1, d))

        # Lloyd
        for _ in range(refine_iter):
            dists = torch.cdist(ks, centers)
            assign = dists.argmin(dim=2)
            new_centers = torch.zeros_like(centers)
            new_centers.scatter_add_(1, assign[..., None].expand(-1, -1, d), ks)
            cnt = torch.zeros(H, K, device=keys.device, dtype=keys.dtype)
            cnt.scatter_add_(1, assign, torch.ones(H, N, device=keys.device, dtype=keys.dtype))
            cnt = cnt.clamp_min(1.0)
            centers = new_centers / cnt.unsqueeze(-1)

        assign = torch.cdist(ks, centers).argmin(dim=2)
        parent = centers.gather(1, assign[..., None].expand(-1, -1, d))
        dd = (ks - parent).norm(dim=-1)
        radii = torch.zeros(H, K, device=keys.device, dtype=keys.dtype)
        radii.scatter_reduce_(1, assign, dd, reduce="amax", include_self=True)

        out_assigns.append(assign)
        out_centers.append(centers)
        out_radii.append(radii)

    return out_assigns, out_centers, out_radii


def time_call(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=4)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    torch.manual_seed(args.seed)
    keys = torch.randn(args.H, args.N, args.D, device="cuda", dtype=torch.float32)

    print(f"build micro-bench: H={args.H} N={args.N} D={args.D} bf={args.bf} S={args.S}")
    print("-" * 70)

    results = []
    for name, info in sorted(build_kernels().items()):
        fn = info.fn
        ms = time_call(
            lambda: fn(keys, args.bf, args.S, args.refine_iter),
            iters=args.iters, warmup=1,
        )
        results.append((f"{name} ({info.version})", ms))
        print(f"  {name:<24s} {info.version:<6s}  {ms:8.2f} ms")

    ms = time_call(
        lambda: torch_naive_build(keys, args.bf, args.S, args.refine_iter),
        iters=args.iters, warmup=1,
    )
    results.append(("torch_naive (no fpc)", ms))
    print(f"  {'torch_naive':<24s} {'-':<6s}  {ms:8.2f} ms")

    print("-" * 70)
    best = min(results, key=lambda r: r[1])
    print(f"Fastest: {best[0]} at {best[1]:.2f} ms")


if __name__ == "__main__":
    main()
