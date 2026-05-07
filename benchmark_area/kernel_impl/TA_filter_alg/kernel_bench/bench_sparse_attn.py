"""Benchmark masked attention baselines across sparse top-k mask fractions.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_atomic_fp16 import (
    sdpa_cuda_atomic_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_sparse_v2_4_fp16 import (
    sdpa_cuda_sparse_v2_4_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_sparse_v2_5_fp16 import (
    sdpa_cuda_sparse_v2_5_fp16,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map


BUFFER_SIZE = 256


def split_index_buffer(
    keys: torch.Tensor, values: torch.Tensor, buffer_size: int = BUFFER_SIZE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (H_kv, N, D) keys/values into (index keys/values, buffer keys/values).

    Index = first N - buffer_size; buffer = last ``buffer_size`` rows.
    """
    n_ctx = int(keys.shape[1])
    if n_ctx <= buffer_size:
        raise ValueError(
            f"need N > buffer_size; got N={n_ctx}, buffer_size={buffer_size}"
        )
    n_index = n_ctx - buffer_size
    keys_index = keys[:, :n_index, :].contiguous()
    values_index = values[:, :n_index, :].contiguous()
    buffer_keys = keys[:, n_index:, :].contiguous()
    buffer_values = values[:, n_index:, :].contiguous()
    return keys_index, values_index, buffer_keys, buffer_values


def mask_to_compact_with_buffer_split(
    mask: torch.Tensor, n_index: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert dense int8 mask [Hq, N] into (live_idx [Hq, n_index] int32 over
    the index portion, live_count [Hq] int32, l_buf int).

    Buffer keys are the trailing ``mask.shape[1] - n_index`` columns. Variant A
    sees them as a separate tensor (always all 256 attended), so we strip
    them off the mask before compacting.
    """
    h_q, n_ctx = mask.shape
    n_buf = n_ctx - n_index
    if n_buf < 0:
        raise ValueError(f"n_index={n_index} > n_ctx={n_ctx}")
    mask_index = mask[:, :n_index].contiguous()
    live_count = mask_index.sum(dim=1).to(torch.int32)
    live_idx = torch.zeros(h_q, n_index, dtype=torch.int32, device=mask.device)
    for h in range(h_q):
        idx = mask_index[h].nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        c = idx.numel()
        if c > 0:
            live_idx[h, :c] = idx
    return live_idx, live_count, n_buf


def reference_masked_attention(
    q: torch.Tensor,                    # (H_q, D) fp16
    keys: torch.Tensor,                 # (H_kv, N, D) fp16
    values: torch.Tensor,               # (H_kv, N, D_v) fp16
    mask: torch.Tensor,                 # (H_q, N) int8
    q_head_to_kv: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Dense softmax attention restricted to ``mask`` — fp32 reference.

    Used to measure the relative error of every kernel against the same
    masked softmax it claims to compute.
    """
    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    keys_eff = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    values_eff = values if q_head_to_kv is None else values.index_select(0, q_head_to_kv)
    keys_f = keys_eff.float()
    values_f = values_eff.float()
    qf = q.float()
    scores = torch.einsum("hd,hnd->hn", qf, keys_f) * scale
    scaled = scores.masked_fill(mask == 0, float("-inf"))
    m = scaled.amax(dim=-1, keepdim=True)
    m = m.masked_fill(torch.isinf(m), 0.0)
    e = torch.exp(scaled - m)
    denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    probs = e / denom
    return torch.einsum("hn,hnd->hd", probs, values_f)


def relative_error(out: torch.Tensor, ref: torch.Tensor) -> float:
    out_f = out.float()
    ref_f = ref.float()
    diff = (out_f - ref_f).abs().max().item()
    denom = ref_f.abs().max().item() + 1e-9
    return diff / denom


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def mask_to_compact(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense int8 mask [Hq, N] to (live_idx [Hq, N] int32, live_count [Hq] int32).

    Done outside the timing loop — compact-input kernels are not meant to do
    this work themselves.
    """
    h_q, n_ctx = mask.shape
    live_count = mask.sum(dim=1).to(torch.int32)
    live_idx = torch.zeros(h_q, n_ctx, dtype=torch.int32, device=mask.device)
    for h in range(h_q):
        idx = mask[h].nonzero(as_tuple=False).squeeze(-1).to(torch.int32)
        c = idx.numel()
        if c > 0:
            live_idx[h, :c] = idx
    return live_idx, live_count


def topk_dot_mask(
    q: torch.Tensor,
    keys: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    fraction: float,
) -> torch.Tensor:
    h_q = int(q.shape[0])
    n_ctx = int(keys.shape[1])
    if fraction >= 1.0:
        return torch.ones(h_q, n_ctx, device=q.device, dtype=torch.int8)
    k_eff = max(1, min(n_ctx, int(round(float(fraction) * n_ctx))))
    keys_eff = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eff.float())
    top_idx = torch.topk(scores, k_eff, dim=-1).indices
    mask = torch.zeros(h_q, n_ctx, device=q.device, dtype=torch.int8)
    mask.scatter_(1, top_idx, 1)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.2, 0.1, 0.05])
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--buffer-size", type=int, default=BUFFER_SIZE,
                   help="Trailing rows of N treated as the v2.5 buffer. "
                        "Other baselines see the full N.")
    args = p.parse_args()

    cap = CaptureState.load(args.input_qkv)
    qcpu, kcpu, vcpu = cap.to_layer_tensors(args.layer)
    if vcpu is None:
        raise RuntimeError("Captured values required.")

    keys = kcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    values = vcpu.to(device="cuda", dtype=torch.float32).half().contiguous()
    h_q = int(qcpu.shape[0])
    h_kv, n_ctx, d = keys.shape
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None
    scale = 1.0 / math.sqrt(d)

    # v2.5 split: index = first (N - buffer_size); buffer = last buffer_size.
    keys_index, values_index, buffer_keys, buffer_values = split_index_buffer(
        keys, values, args.buffer_size
    )
    n_index = int(keys_index.shape[1])
    l_buf = int(buffer_keys.shape[1])
    print(
        f"v2.5 split: n_index={n_index} (full attention input for v2.5), "
        f"l_buf={l_buf} (buffer)."
    )

    total_q = qcpu.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride))[
        : args.n_queries
    ]
    queries: list[torch.Tensor] = []
    queries_f32: list[torch.Tensor] = []
    for qi in q_indices:
        q = qcpu[:, qi, :].to(device="cuda", dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        queries_f32.append(q.contiguous())
        queries.append(q.half().contiguous())

    # input_kind:
    #   "mask"            → fn(q, keys[:N], values[:N], mask[:N], q_head_to_kv, scale)
    #   "compact"         → fn(q, keys[:N], values[:N], live_idx[:N], live_count, q_head_to_kv, scale)
    #   "compact_buffer"  → v2.5 only: index portion of keys/values + separate buffer
    #                       tensors; live_idx/live_count cover only the index.
    baselines = [
        ("cuda_atomic",      sdpa_cuda_atomic_fp16,      "mask",           {}),
        ("cuda_sparse_v2_4", sdpa_cuda_sparse_v2_4_fp16, "compact",        {}),
        ("cuda_sparse_v2_5", sdpa_cuda_sparse_v2_5_fp16, "compact_buffer", {}),
    ]

    fractions = [float(f) for f in args.fractions]
    if all(abs(f - 1.0) > 1e-8 for f in fractions):
        fractions = [1.0] + fractions
    fractions = sorted(set(fractions), reverse=True)

    print(f"Hq={h_q} Hkv={h_kv} N={n_ctx} D={d} queries={len(queries)}")
    results_by_baseline: dict[str, dict[float, float]] = {name: {} for name, _, _, _ in baselines}
    relerr_by_baseline: dict[str, dict[float, float]] = {name: {} for name, _, _, _ in baselines}

    for frac in fractions:
        # Mask is computed over the full N for fairness. Buffer keys (last
        # buffer_size rows) are forced "alive" for v2.5 (it always sees them);
        # for everyone else the same mask is applied as-is.
        masks = [topk_dot_mask(q, keys, q_head_to_kv, frac) for q in queries_f32]
        # For v2.5 the buffer is *always* attended; force those mask columns
        # on so the v2.4/atomic baselines compute the same target softmax.
        masks_v25 = []
        for m in masks:
            m2 = m.clone()
            m2[:, n_index:] = 1
            masks_v25.append(m2)
        compacts = [mask_to_compact(m) for m in masks_v25]
        compacts_v25 = [
            mask_to_compact_with_buffer_split(m, n_index) for m in masks_v25
        ]
        torch.cuda.synchronize()
        actual = sum(float(m.float().mean().item()) for m in masks_v25) / len(masks_v25)
        print(f"\nfraction={frac:g} actual={actual:.4f} (buffer always 1)")

        # Build fp32 references per query (used for all baselines this frac).
        refs = [
            reference_masked_attention(
                q, keys, values, mask_v25, q_head_to_kv, scale,
            )
            for q, mask_v25 in zip(queries, masks_v25)
        ]

        for name, fn, input_kind, kwargs in baselines:
            if input_kind == "mask":
                def run() -> None:
                    for q, mask in zip(queries, masks_v25):
                        fn(q, keys, values, mask, q_head_to_kv, scale, **kwargs)

                def first_call() -> torch.Tensor:
                    return fn(queries[0], keys, values, masks_v25[0],
                              q_head_to_kv, scale, **kwargs)
            elif input_kind == "compact":
                def run() -> None:
                    for q, (li, lc) in zip(queries, compacts):
                        fn(q, keys, values, li, lc, q_head_to_kv, scale, **kwargs)

                def first_call() -> torch.Tensor:
                    li, lc = compacts[0]
                    return fn(queries[0], keys, values, li, lc,
                              q_head_to_kv, scale, **kwargs)
            elif input_kind == "compact_buffer":
                def run() -> None:
                    for q, (li, lc, lb) in zip(queries, compacts_v25):
                        fn(
                            q, keys_index, values_index,
                            buffer_keys, buffer_values,
                            li, lc, lb,
                            q_head_to_kv, scale, **kwargs,
                        )

                def first_call() -> torch.Tensor:
                    li, lc, lb = compacts_v25[0]
                    return fn(
                        queries[0], keys_index, values_index,
                        buffer_keys, buffer_values,
                        li, lc, lb,
                        q_head_to_kv, scale, **kwargs,
                    )
            else:
                raise ValueError(f"unknown input_kind={input_kind}")

            # Correctness check: relative error of first query's output vs.
            # fp32 reference (same masked softmax for everyone).
            try:
                out = first_call().clone()
                rel = relative_error(out, refs[0])
                relerr_by_baseline[name][frac] = rel
            except Exception as exc:
                torch.cuda.synchronize()
                relerr_by_baseline[name][frac] = float("nan")
                print(f"  {name:<20s} skipped (rel-err): {type(exc).__name__}: {str(exc).splitlines()[0]}")
                continue

            try:
                ms = time_call(run, args.iters, args.warmup) / len(queries)
                results_by_baseline[name][frac] = ms
                rel_str = (
                    f"{relerr_by_baseline[name][frac]:.2e}"
                    if relerr_by_baseline[name].get(frac) == relerr_by_baseline[name].get(frac)
                    else "nan"
                )
                print(
                    f"  {name:<20s} {ms:9.6f} ms/query   rel_err={rel_str}"
                )
            except Exception as exc:
                torch.cuda.synchronize()
                print(f"  {name:<20s} skipped: {type(exc).__name__}: {str(exc).splitlines()[0]}")

    print("\nSpeedup vs fraction=1.0 (speedup = t@1.0 / t@fraction)")
    print("-" * 72)
    header = ["baseline"] + [f"f={f:g}" for f in fractions]
    print("  " + "  ".join(f"{h:<14s}" for h in header))
    for name, _, _, _ in baselines:
        base = results_by_baseline[name].get(1.0)
        row = [f"{name:<14s}"]
        for frac in fractions:
            t = results_by_baseline[name].get(frac)
            if base is None or t is None:
                cell = "n/a"
            else:
                speedup = base / t if t > 0.0 else float("inf")
                cell = f"{speedup:.3f}x"
            row.append(f"{cell:<14s}")
        print("  " + "  ".join(row))

    print("\nRelative error vs masked-softmax fp32 reference (max_abs_diff / max_abs_ref)")
    print("-" * 72)
    print("  " + "  ".join(f"{h:<14s}" for h in header))
    for name, _, _, _ in baselines:
        row = [f"{name:<14s}"]
        for frac in fractions:
            r = relerr_by_baseline[name].get(frac)
            cell = "n/a" if r is None else (f"{r:.2e}" if r == r else "nan")
            row.append(f"{cell:<14s}")
        print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
