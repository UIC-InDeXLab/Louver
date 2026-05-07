#!/usr/bin/env python3
"""
Run PCA diagnostics on captured attention keys to assess low-dimensional structure.

This script captures Q/K/V the same way as the pruning benchmarks, then runs PCA on:
1) pooled keys across KV heads
2) each KV head independently

It reports:
- components needed to explain 90/95/99% variance
- effective rank (entropy-based)
- participation ratio
- reconstruction error for several k values
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning_bench_utils import _capture_qkv
from method_comparison_bench import DEVICE, DTYPE, LAYER_IDX, MODEL_NAME, PROMPT


@dataclass
class PcaStats:
    tag: str
    n_samples: int
    d: int
    k90: int
    k95: int
    k99: int
    effective_rank: float
    participation_ratio: float
    recon_rel_k4: float
    recon_rel_k8: float
    recon_rel_k16: float
    recon_rel_k32: float
    recon_rel_k64: float


def _components_for_ratio(cum_ratio: torch.Tensor, target: float) -> int:
    idx = int(torch.searchsorted(cum_ratio, torch.tensor(target, dtype=cum_ratio.dtype)).item())
    return min(cum_ratio.numel(), idx + 1)


def _recon_rel_error(x_centered: torch.Tensor, v: torch.Tensor, k: int) -> float:
    if k <= 0:
        return 1.0
    k = min(k, v.shape[1])
    vk = v[:, :k]  # (D, k)
    x_hat = (x_centered @ vk) @ vk.transpose(0, 1)
    num = torch.linalg.norm(x_centered - x_hat)
    den = torch.linalg.norm(x_centered).clamp_min(1e-12)
    return float((num / den).item())


def _run_pca_stats(x: torch.Tensor, tag: str) -> PcaStats:
    """
    x: (N, D) on CPU.
    """
    n, d = x.shape
    if n < 2:
        raise ValueError(f"{tag}: need at least 2 samples, got {n}")

    x = x.to(dtype=torch.float64)
    x_centered = x - x.mean(dim=0, keepdim=True)

    # X = U S V^T
    _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
    eig = (s * s) / max(1, (n - 1))  # covariance eigenvalues
    total = eig.sum().clamp_min(1e-12)
    ratio = eig / total
    cum = torch.cumsum(ratio, dim=0)

    k90 = _components_for_ratio(cum, 0.90)
    k95 = _components_for_ratio(cum, 0.95)
    k99 = _components_for_ratio(cum, 0.99)

    # Effective rank via entropy of normalized eigenvalues.
    p = ratio.clamp_min(1e-20)
    entropy_bits = float((-p * torch.log2(p)).sum().item())
    effective_rank = 2.0**entropy_bits

    # Participation ratio.
    participation_ratio = float((eig.sum() ** 2 / (eig.square().sum().clamp_min(1e-20))).item())

    v = vh.transpose(0, 1).contiguous()  # (D, D')
    recon = {}
    for k in (4, 8, 16, 32, 64):
        recon[k] = _recon_rel_error(x_centered, v, k=k)

    return PcaStats(
        tag=tag,
        n_samples=n,
        d=d,
        k90=k90,
        k95=k95,
        k99=k99,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        recon_rel_k4=recon[4],
        recon_rel_k8=recon[8],
        recon_rel_k16=recon[16],
        recon_rel_k32=recon[32],
        recon_rel_k64=recon[64],
    )


def _print_table(stats: list[PcaStats]) -> None:
    print("\n" + "=" * 140)
    print(
        f"{'TAG':<16s} {'N':>6s} {'D':>4s} {'K90':>6s} {'K95':>6s} {'K99':>6s} "
        f"{'EFF_RANK':>10s} {'PART_RATIO':>11s} "
        f"{'RECON@4':>9s} {'RECON@8':>9s} {'RECON@16':>10s} {'RECON@32':>10s} {'RECON@64':>10s}"
    )
    print("-" * 140)
    for s in stats:
        print(
            f"{s.tag:<16s} {s.n_samples:>6d} {s.d:>4d} {s.k90:>6d} {s.k95:>6d} {s.k99:>6d} "
            f"{s.effective_rank:>10.2f} {s.participation_ratio:>11.2f} "
            f"{s.recon_rel_k4:>9.4f} {s.recon_rel_k8:>9.4f} {s.recon_rel_k16:>10.4f} "
            f"{s.recon_rel_k32:>10.4f} {s.recon_rel_k64:>10.4f}"
        )
    print("=" * 140)


def _write_csv(stats: list[PcaStats], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].__dict__.keys()))
        writer.writeheader()
        for s in stats:
            writer.writerow(s.__dict__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=MODEL_NAME)
    p.add_argument("--layer", type=int, default=LAYER_IDX)
    p.add_argument("--n-tokens", type=int, default=600)
    p.add_argument("--device", type=str, default=DEVICE)
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("key_pca_diagnostics.csv"),
        help="Optional CSV output path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError('This script is CUDA-only. Use "--device cuda".')

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
    _, keys_cpu, _ = capture.to_layer_tensors(layer)  # (H_kv, N, D)
    keys = keys_cpu.to(dtype=torch.float32, device="cpu")

    h_kv, n, d = keys.shape
    print(f"Layer {layer}: H_kv={h_kv}, N={n}, D={d}")

    stats: list[PcaStats] = []

    # pooled
    pooled = keys.reshape(h_kv * n, d).contiguous()
    stats.append(_run_pca_stats(pooled, tag="pooled"))

    # per-head
    for h in range(h_kv):
        stats.append(_run_pca_stats(keys[h], tag=f"head_{h}"))

    _print_table(stats)
    _write_csv(stats, args.output_csv)
    print(f"\nSaved CSV: {args.output_csv}")

    pooled_stats = stats[0]
    low_dim_hint = pooled_stats.k95 / max(1, pooled_stats.d)
    print("\nInterpretation hint:")
    print(
        f"pooled K95/D = {pooled_stats.k95}/{pooled_stats.d} = {low_dim_hint:.3f}. "
        "Lower means stronger low-dimensional structure."
    )


if __name__ == "__main__":
    main()

