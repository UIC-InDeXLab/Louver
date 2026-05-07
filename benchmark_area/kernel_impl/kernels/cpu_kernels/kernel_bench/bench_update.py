"""CPU micro-benchmark for update kernels.

Mirrors the GPU bench_update report format:
  - fresh (rebuild) baseline per B
  - ms/B and x_fresh speedup columns
  - kept_frac / recall@topk pruning stats
  - summary table across all buffer sizes

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench.bench_update
    python -m ... --B-sweep 32,64,128,256
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
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels import update_kernels
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build as build_v1_0
    from hira.benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import real_update_case
except ModuleNotFoundError:
    from benchmark_area.kernel_impl.kernels.cpu_kernels import update_kernels
    from benchmark_area.kernel_impl.kernels.cpu_kernels.build_v1_0 import build as build_v1_0
    from benchmark_area.kernel_impl.kernels.cpu_kernels.kernel_bench._capture_inputs import real_update_case


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def _subspace_topk_thresholds(q, keys, topk, dim_slices):
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    ths = []
    for s, e in dim_slices:
        qs = q[:, s:e]
        ks = keys[:, :, s:e]
        ss = torch.einsum("hd,hnd->hn", qs, ks)
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0).contiguous(), topk_idx


def _pruning_stats(state, q, keys_expanded, full_keys, topk):
    """Return (kept_frac, topk_recall) for this state.

    kept_frac  = surviving real children / real children
    recall     = true top-k children that survive / topk
    """
    h_q = q.shape[0]
    h_kv = int(state["keys_reord"].shape[0])
    dim_slices = state["dim_slices"]
    n_real = int(state.get("N_used", state["N"]))

    # For GQA: build q_head_to_kv assuming uniform grouping.
    groups = h_q // h_kv if h_q >= h_kv else 1
    q_head_to_kv = torch.arange(h_q, dtype=torch.long).div(groups, rounding_mode="floor")
    q_head_to_kv = q_head_to_kv.clamp_max(h_kv - 1)

    th, topk_idx = _subspace_topk_thresholds(q, keys_expanded, topk, dim_slices)

    cluster_pass_list = []
    for s_idx, (start, end) in enumerate(dim_slices):
        q_sub = q[:, start:end]
        centers = state["centers"][s_idx].index_select(0, q_head_to_kv)
        radii = state["radii"][s_idx].index_select(0, q_head_to_kv)
        scores = torch.einsum("hd,hkd->hk", q_sub, centers)
        scores = scores + q_sub.norm(dim=-1, keepdim=True) * radii
        cluster_pass_list.append(scores >= th[s_idx].unsqueeze(-1))
    cluster_pass = torch.stack(cluster_pass_list, dim=0)

    invalid_mask = state["invalid_mask"].index_select(0, q_head_to_kv)
    survive = torch.ones_like(invalid_mask)
    for s_idx in range(len(dim_slices)):
        a = state["assigns_reord"][s_idx].index_select(0, q_head_to_kv).long()
        gate = cluster_pass[s_idx].gather(1, a) != 0
        survive &= gate
    survive &= ~invalid_mask

    valid_phys = ~invalid_mask
    kept_frac = (
        survive.float().sum(dim=-1)
        / valid_phys.float().sum(dim=-1).clamp_min(1.0)
    ).mean().item()

    perm = state["reorder_perm"].index_select(0, q_head_to_kv).long()
    inv_perm = torch.full((h_q, n_real), -1, dtype=torch.long)
    for h in range(h_q):
        valid_j = valid_phys[h].nonzero(as_tuple=True)[0]
        orig_idx = perm[h, valid_j]
        inv_perm[h, orig_idx] = valid_j
    recall_counts = []
    for h in range(h_q):
        phys_for_topk = inv_perm[h, topk_idx[h]]
        in_index = phys_for_topk >= 0
        survived = survive[h, phys_for_topk.clamp_min(0)] & in_index
        recall_counts.append(survived.sum().item() / max(1, topk))
    recall = sum(recall_counts) / len(recall_counts)

    return kept_frac, recall


def _run_for_B(
    args,
    B: int,
    keys: torch.Tensor,
    values: torch.Tensor | None,
    base_state: dict,
    verbose: bool,
    buffer_keys: torch.Tensor | None = None,
    buffer_values: torch.Tensor | None = None,
) -> dict:
    h_kv, n_old, d = keys.shape
    d_v = values.shape[-1] if values is not None else d
    h_q = args.H_q

    if buffer_keys is None:
        torch.manual_seed(1000 + B)
        buffer_keys = torch.randn(h_kv, B, d)
        buffer_keys = buffer_keys / buffer_keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if buffer_values is None and values is not None:
        buffer_values = torch.randn(h_kv, B, d_v)

    q_batch = torch.randn(args.prune_queries, h_q, d)
    q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    full_keys = torch.cat([keys, buffer_keys], dim=1)
    # For GQA expand: uniform grouping
    groups = h_q // h_kv if h_q >= h_kv else 1
    q2kv = torch.arange(h_q).div(groups, rounding_mode="floor").clamp_max(h_kv - 1)
    keys_expanded = full_keys.index_select(0, q2kv)
    full_values = None
    if values is not None and buffer_values is not None:
        full_values = torch.cat([values, buffer_values], dim=1)

    def fresh_build():
        fk = torch.cat([keys, buffer_keys], dim=1).contiguous()
        fv = torch.cat([values, buffer_values], dim=1).contiguous() if values is not None else None
        build_v1_0(fk, args.bf, args.S, args.refine_iter, values=fv)

    ms_fresh = time_call(fresh_build, iters=args.iters, warmup=args.warmup)
    ms_fresh_per_b = ms_fresh / max(1, B)

    # Fresh rebuild state for pruning stats.
    fk = torch.cat([keys, buffer_keys], dim=1).contiguous()
    fv = torch.cat([values, buffer_values], dim=1).contiguous() if values is not None else None
    fresh_state = build_v1_0(fk, args.bf, args.S, args.refine_iter, values=fv)

    if verbose:
        print(f"  {'build_v1_0 (fresh)':<26s} {'ref':<10s}  {ms_fresh_per_b:10.4f} ms/B")

    kernel_ms: dict[str, float] = {}
    kept_states: dict[str, dict] = {}
    for name, info in sorted(update_kernels().items()):
        def call(fn=info.fn):
            fn(
                base_state, keys, buffer_keys, args.bf, args.S,
                args.update_refine_iter,
                old_values=values, buffer_values=buffer_values,
                return_merged=False,
            )
        try:
            ms = time_call(call, iters=args.iters, warmup=args.warmup)
        except Exception as exc:
            if verbose:
                print(f"  {name:<26s} {info.version:<10s}  FAIL {type(exc).__name__}: {exc}")
            continue
        ms_per_b = ms / max(1, B)
        if verbose:
            print(f"  {name:<26s} {info.version:<10s}  {ms_per_b:10.4f} ms/B  "
                  f"({ms_fresh / ms:5.1f}x vs fresh)")
        kernel_ms[name] = ms
        ret = info.fn(
            base_state, keys, buffer_keys, args.bf, args.S,
            args.update_refine_iter,
            old_values=values, buffer_values=buffer_values,
            return_merged=False,
        )
        kept_states[name] = ret[0]

    # Pruning stats.
    stats_states = {"fresh (rebuild)": fresh_state, **kept_states}
    stats: dict[str, tuple[float, float]] = {}
    for name, st in stats_states.items():
        kept_sum = recall_sum = 0.0
        for qi in range(q_batch.shape[0]):
            kf, rc = _pruning_stats(st, q_batch[qi], keys_expanded, full_keys, args.topk)
            kept_sum += kf
            recall_sum += rc
        n_q = q_batch.shape[0]
        stats[name] = (kept_sum / n_q, recall_sum / n_q)

    if verbose:
        for name, (kf, rc) in stats.items():
            print(f"  pruning[{name:<18s}]: kept_frac={kf:.4f}  recall@{args.topk}={rc:.4f}")

    out: dict = {
        "fresh (rebuild)": {
            "ms": ms_fresh,
            "ms_per_b": ms_fresh_per_b,
            "kept_frac": stats["fresh (rebuild)"][0],
            "recall": stats["fresh (rebuild)"][1],
        }
    }
    for name in kernel_ms:
        out[name] = {
            "ms": kernel_ms[name],
            "ms_per_b": kernel_ms[name] / max(1, B),
            "kept_frac": stats[name][0],
            "recall": stats[name][1],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--H-kv", type=int, default=8)
    p.add_argument("--H-q", type=int, default=8, help="Query heads (for GQA pruning stats)")
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--B", type=int, default=128, help="Buffer size")
    p.add_argument("--B-sweep", type=str, default=None,
                   help="Comma-separated buffer sizes to sweep (overrides --B, synthetic only)")
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--D-v", type=int, default=128)
    p.add_argument("--input-qkv", type=Path, default=None,
                   help="Path to a captured QKV .pt file. Uses real keys/buffer; "
                        "pruning queries are synthetic.")
    p.add_argument("--layer", type=int, default=None,
                   help="Transformer layer to load from --input-qkv.")
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=2)
    p.add_argument("--update-refine-iter", type=int, default=0)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--topk", type=int, default=32,
                   help="Top-k for deriving tight thresholds in pruning stats")
    p.add_argument("--prune-queries", type=int, default=20,
                   help="Number of queries to average pruning stats over")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    capture_buffer_keys = None
    capture_buffer_values = None

    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        real = real_update_case(args.input_qkv, args.layer, args.N, args.B)
        keys = real["keys"]
        values = real["values"]
        capture_buffer_keys = real["buffer_keys"]
        capture_buffer_values = real["buffer_values"]
        b_list = [capture_buffer_keys.shape[1]]
        h_kv, n_old, d = keys.shape
        print(
            f"Using layer {real['layer']}  N={n_old}  B={b_list[0]}  "
            f"of {real['total_keys']} captured keys"
        )
        args.H_kv = h_kv
        args.H_q = h_kv  # no query heads in capture; use symmetric GQA
    else:
        if args.B_sweep is not None:
            b_list = [int(x) for x in args.B_sweep.split(",") if x.strip()]
        else:
            b_list = [args.B]
        torch.manual_seed(args.seed)
        keys = torch.randn(args.H_kv, args.N, args.D)
        keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        values = torch.randn(args.H_kv, args.N, args.D_v)

    base_state = build_v1_0(keys, args.bf, args.S, args.refine_iter, values=values)

    h_kv, n_old, d = keys.shape
    d_v = values.shape[-1] if values is not None else args.D_v
    mode = f"capture layer {real['layer']}" if args.input_qkv is not None else "synthetic"
    print(
        f"cpu update micro-bench [{mode}]: H_q={args.H_q} H_kv={h_kv} "
        f"N={n_old} D={d} D_v={d_v} bf={args.bf} S={args.S}"
    )
    print("=" * 78)

    all_results: dict[int, dict] = {}
    verbose = len(b_list) == 1
    for B in b_list:
        bk = capture_buffer_keys if capture_buffer_keys is not None else None
        bv = capture_buffer_values if capture_buffer_values is not None else None
        if not verbose:
            print(f"[B={B}] ...")
        else:
            print(f"  B={B}")
            print("-" * 78)
        all_results[B] = _run_for_B(
            args, B, keys, values, base_state, verbose=verbose,
            buffer_keys=bk, buffer_values=bv,
        )

    print("=" * 78)
    print("Summary")
    header = (
        f"  {'B':>6s}  {'kernel':<22s}  {'ms/B':>10s}  "
        f"{'x_fresh':>8s}  {'kept':>7s}  {'recall':>7s}"
    )
    print(header)
    print("-" * 78)
    for B in b_list:
        res = all_results[B]
        ms_fresh = res["fresh (rebuild)"]["ms"]
        fresh = res["fresh (rebuild)"]
        print(
            f"  {B:>6d}  {'build_v1_0 (fresh)':<22s}  {fresh['ms_per_b']:>10.4f}  "
            f"{'1.0x':>8s}  {fresh['kept_frac']:>7.4f}  {fresh['recall']:>7.4f}"
        )
        for name in sorted(res):
            if name == "fresh (rebuild)":
                continue
            r = res[name]
            print(
                f"  {B:>6d}  {name:<22s}  {r['ms_per_b']:>10.4f}  "
                f"{ms_fresh / r['ms']:>7.1f}x  "
                f"{r['kept_frac']:>7.4f}  {r['recall']:>7.4f}"
            )
        print()


if __name__ == "__main__":
    main()
