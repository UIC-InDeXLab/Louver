"""Micro-benchmark: compare kept update kernels.

Measures how long it takes to fold a small buffer of new (key, value) rows
into an existing v2_4 index, and verifies that the attention kernel
(v1_5, the generic fallback) still produces correct output on the updated
state (loose gate vs dense reference).

By default the benchmark synthesizes random keys/values. Pass `--input-qkv`
to slice a real decode trace from a captured QKV `.pt` file instead.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_update
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

from hira.benchmark_area.kernel_impl.kernels.build_v2_7 import build as build_v2_7
from hira.benchmark_area.kernel_impl.kernels import update_kernels, attention_kernels
from hira.benchmark_area.kernel_impl.kernels._update_v3_utils import apply_pending_publish
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _q_to_kv_map,
)

UPDATE_WHITELIST = {
    "update_v4_0",
}
SUMMARY_ORDER = (
    "update_v4_0",
)


def _amortized_ms(ms: float, B: int) -> float:
    return ms / max(1, B)


def time_call(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _rebuild_reference(
    old_keys,
    buffer_keys,
    old_values,
    buffer_values,
    bf,
    S,
    refine_iter,
):
    full_keys = torch.cat([old_keys, buffer_keys], dim=1).contiguous()
    full_values = None
    if old_values is not None and buffer_values is not None:
        full_values = torch.cat([old_values, buffer_values], dim=1).contiguous()
    return build_v2_7(full_keys, bf, S, refine_iter, values=full_values), full_keys, full_values


def _dense_attention(q, keys_q, values_q, scale):
    scores = torch.einsum("hd,hnd->hn", q, keys_q) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hn,hnd->hd", probs, values_q)


def _subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace threshold = min over true top-k of q·k restricted to subspace."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices  # (H, topk)
    ths = []
    for s, e in dim_slices:
        qs = q[:, s:e]
        ks = keys[:, :, s:e]
        ss = torch.einsum("hd,hnd->hn", qs, ks)
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0).contiguous(), topk_idx


def _pruning_stats(state, q, keys_q_expanded, full_keys, topk, q_head_to_kv):
    """Return (kept_frac, topk_recall) for this state under the tight gate.

    kept_frac    = # surviving real children / # real children (lower = more pruning)
    topk_recall  = # true top-k children that survive / topk (higher = better quality)
    """
    h_q = q.shape[0]
    h_kv = int(state["keys_reord"].shape[0])
    k_parents = int(state["K"])
    dim_slices = state["dim_slices"]
    groups = h_q // h_kv

    # Tight thresholds from the dense reference top-k over the expanded K keys.
    th, topk_idx_merged = _subspace_topk_thresholds(q, keys_q_expanded, topk, dim_slices)

    s_dim = len(dim_slices)
    cluster_pass_list = []
    for s_idx, (start, end) in enumerate(dim_slices):
        q_sub = q[:, start:end]
        centers = state["centers"][s_idx].index_select(0, q_head_to_kv)
        radii = state["radii"][s_idx].index_select(0, q_head_to_kv)
        scores = torch.einsum("hd,hkd->hk", q_sub, centers)
        scores = scores + q_sub.norm(dim=-1, keepdim=True) * radii
        cluster_pass_list.append(scores >= th[s_idx].unsqueeze(-1))
    cluster_pass = torch.stack(cluster_pass_list, dim=0)

    # Per-child survival: AND across subspaces of cluster_pass at that child's
    # per-subspace assigned cluster. assigns_reord[s] is (H_kv, N_pad) int.
    # Expand to (H_q, N_pad) via q_head_to_kv (GQA).
    invalid_mask = state["invalid_mask"].index_select(0, q_head_to_kv)  # (H_q, N_pad)
    survive = torch.ones_like(invalid_mask)  # (H_q, N_pad) bool
    for s_idx in range(s_dim):
        a = state["assigns_reord"][s_idx].index_select(0, q_head_to_kv).long()  # (H_q, N_pad)
        gate = cluster_pass[s_idx].gather(1, a) != 0                              # (H_q, N_pad)
        survive &= gate
    survive &= ~invalid_mask

    # reorder_perm[h, j] gives the original index (into merged keys) for physical j.
    perm = state["reorder_perm"].index_select(0, q_head_to_kv).long()  # (H_q, N_pad)
    # For invalid slots, perm may be arbitrary; mask them out.
    n_real = int(state.get("N_used", state["N"]))
    valid_phys = ~invalid_mask

    kept_frac = (
        survive.float().sum(dim=-1)
        / valid_phys.float().sum(dim=-1).clamp_min(1.0)
    ).mean().item()

    # Recall: are the true top-k original indices surviving?
    # topk_idx_merged is per-head in merged-key space, shape (H_q, topk).
    # Invert perm to lookup physical position for each original index.
    # We do it per head via scatter.
    n_pad = perm.shape[-1]
    inv_perm = torch.full((h_q, n_real), -1, device=q.device, dtype=torch.long)
    for h in range(h_q):
        valid_j = valid_phys[h].nonzero(as_tuple=True)[0]
        orig_idx = perm[h, valid_j]
        inv_perm[h, orig_idx] = valid_j
    recall_counts = []
    for h in range(h_q):
        phys_for_topk = inv_perm[h, topk_idx_merged[h]]
        in_index = phys_for_topk >= 0
        survived = survive[h, phys_for_topk.clamp_min(0)] & in_index
        recall_counts.append(survived.sum().item() / max(1, topk))
    recall = sum(recall_counts) / len(recall_counts)

    return kept_frac, recall


def _select_real_queries(
    queries_cpu: torch.Tensor,
    prompt_length: int,
    old_len: int,
    B: int,
    n_queries: int,
    *,
    device: str,
) -> tuple[torch.Tensor, list[int]]:
    """Return normalized real queries starting at the update window boundary.

    The first selected query is aligned with the last token in the update buffer.
    Later queries, when requested, are taken from subsequent decode steps in the
    same capture.
    """
    q_start = old_len + B - prompt_length - 1
    if q_start < 0:
        raise ValueError(
            f"Need --N + B to reach generated tokens in the capture, got "
            f"N={old_len}, B={B}, prompt_length={prompt_length}."
        )

    q_stop = min(int(queries_cpu.shape[1]), q_start + max(1, n_queries))
    if q_start >= q_stop:
        raise ValueError(
            f"Capture does not contain any generated queries for N={old_len}, B={B}."
        )

    q_batch = queries_cpu[:, q_start:q_stop, :].to(device=device, dtype=torch.float32)
    q_batch = q_batch.permute(1, 0, 2).contiguous()
    q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q_batch, list(range(q_start, q_stop))


def _load_real_capture(args, max_b: int, *, device: str) -> dict:
    print(f"Loading capture from {args.input_qkv} ...")
    cap = CaptureState.load(args.input_qkv)
    layer_ids = cap.layer_ids()
    if not layer_ids:
        raise ValueError(f"Capture {args.input_qkv} does not contain any layers.")

    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    prompt_length = (
        int(cap.prompt_length)
        if cap.prompt_length is not None
        else int(keys_cpu.shape[1] - queries_cpu.shape[1])
    )
    total_keys = int(keys_cpu.shape[1])

    old_len = max(prompt_length, total_keys - max_b) if args.N is None else args.N
    if old_len < prompt_length:
        raise ValueError(
            f"--N={old_len} is smaller than captured prompt_length={prompt_length}. "
            "The update buffer must come from generated tokens."
        )
    if old_len + max_b > total_keys:
        raise ValueError(
            f"Need N + max(B) <= total captured keys ({total_keys}), got "
            f"N={old_len}, max(B)={max_b}."
        )

    h_q = int(queries_cpu.shape[0])
    h_kv = int(keys_cpu.shape[0])
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, device)

    return {
        "layer": layer,
        "prompt_length": prompt_length,
        "total_keys": total_keys,
        "old_len": old_len,
        "queries_cpu": queries_cpu,
        "keys_cpu": keys_cpu,
        "values_cpu": values_cpu,
        "keys": keys_cpu[:, :old_len, :].to(device=device, dtype=torch.float32).contiguous(),
        "values": (
            values_cpu[:, :old_len, :].to(device=device, dtype=torch.float32).contiguous()
            if values_cpu is not None else None
        ),
        "q_head_to_kv": q_head_to_kv,
    }


def _slice_real_case(real_data: dict, B: int, *, device: str, n_queries: int):
    old_len = int(real_data["old_len"])
    buf_slice = slice(old_len, old_len + B)
    buffer_keys = real_data["keys_cpu"][:, buf_slice, :].to(
        device=device, dtype=torch.float32
    ).contiguous()
    values_cpu = real_data["values_cpu"]
    buffer_values = (
        values_cpu[:, buf_slice, :].to(device=device, dtype=torch.float32).contiguous()
        if values_cpu is not None else None
    )
    q_batch, q_indices = _select_real_queries(
        real_data["queries_cpu"],
        int(real_data["prompt_length"]),
        old_len,
        B,
        n_queries,
        device=device,
    )
    return buffer_keys, buffer_values, q_batch, q_indices


def _run_for_B(
    args,
    B: int,
    keys: torch.Tensor,
    values: torch.Tensor | None,
    base_state: dict,
    q_head_to_kv: torch.Tensor,
    q_batch: torch.Tensor,
    *,
    buffer_keys: torch.Tensor | None = None,
    buffer_values: torch.Tensor | None = None,
    verbose: bool,
) -> dict:
    """Run timing + pruning/correctness for a single buffer size B.

    Returns a dict keyed by kernel name with fields:
        ms, kept_frac, recall, corr_abs
    Plus an entry under "fresh (rebuild)" with ms/kept_frac/recall.
    """
    device = keys.device
    h_kv = int(keys.shape[0])
    d = int(keys.shape[-1])
    d_v = int(values.shape[-1]) if values is not None else d

    if buffer_keys is None:
        torch.manual_seed(1000 + B)  # Different buffer per B, reproducible.
        buffer_keys = torch.randn(h_kv, B, d, device=device, dtype=torch.float32)
    if buffer_values is None and values is not None:
        buffer_values = torch.randn(h_kv, B, d_v, device=device, dtype=torch.float32)
    if (values is None) != (buffer_values is None):
        raise ValueError("old/base values and buffer values must either both be present or both be None.")

    def fresh_build():
        full_keys = torch.cat([keys, buffer_keys], dim=1).contiguous()
        full_values = None
        if values is not None and buffer_values is not None:
            full_values = torch.cat([values, buffer_values], dim=1).contiguous()
        build_v2_7(full_keys, args.bf, args.S, args.refine_iter, values=full_values)

    ms_fresh = time_call(fresh_build, iters=args.iters, warmup=2)
    ms_fresh_per_buf = _amortized_ms(ms_fresh, B)
    if verbose:
        print(f"  {'build_v2_7 (fresh)':<26s} {'ref':<8s}  {ms_fresh_per_buf:10.4f} ms/B")

    kept_states: dict[str, dict] = {}
    kernel_ms: dict[str, float] = {}
    for name, info in sorted(update_kernels().items()):
        if name not in UPDATE_WHITELIST:
            continue
        fn = info.fn

        def call(fn=fn):
            fn(
                base_state, keys, buffer_keys,
                args.bf, args.S, args.refine_iter,
                old_values=values, buffer_values=buffer_values,
            )

        try:
            ms = time_call(call, iters=args.iters, warmup=2)
        except Exception as exc:
            if verbose:
                print(f"  {name:<26s} {info.version:<8s}  FAIL {type(exc).__name__}: {exc}")
            continue
        ms_per_buf = _amortized_ms(ms, B)
        if verbose:
            print(f"  {name:<26s} {info.version:<8s}  {ms_per_buf:10.4f} ms/B  "
                  f"({ms_fresh / ms:5.1f}x vs fresh)")
        kernel_ms[name] = ms
        ret = fn(
            base_state, keys, buffer_keys,
            args.bf, args.S, args.refine_iter,
            old_values=values, buffer_values=buffer_values,
        )
        new_state = ret[0]
        if len(ret) >= 4 and ret[3] is not None:
            new_state = apply_pending_publish(ret[3])
        kept_states[name] = new_state

    # Fresh rebuild state for pruning reference.
    fresh_state, _, _ = _rebuild_reference(
        keys, buffer_keys, values, buffer_values,
        args.bf, args.S, args.refine_iter,
    )
    stats_states = {"fresh (rebuild)": fresh_state, **kept_states}

    # Correctness (attention_v5_14, loose gate vs dense).
    attn = attention_kernels().get("attention_v5_14")
    full_keys = torch.cat([keys, buffer_keys], dim=1)
    keys_expanded = full_keys.index_select(0, q_head_to_kv)
    full_values = None
    values_expanded = None
    if values is not None and buffer_values is not None:
        full_values = torch.cat([values, buffer_values], dim=1)
        values_expanded = full_values.index_select(0, q_head_to_kv)
    scale = 1.0 / math.sqrt(d)
    corr_abs: dict[str, float] = {}
    if attn is not None and values_expanded is not None:
        q0 = q_batch[0]
        out_ref = _dense_attention(q0, keys_expanded, values_expanded, scale)
        empty_buf = torch.empty(h_kv, 0, d, device=device, dtype=torch.float32)
        empty_val = torch.empty(h_kv, 0, d_v, device=device, dtype=torch.float32)
        for name, st in kept_states.items():
            s_eff = len(st["assigns_reord"])
            th_loose = torch.full(
                (s_eff, q_batch.shape[1]),
                float(torch.finfo(torch.float16).min),
                device=device,
                dtype=torch.float16,
            )
            q_norms = torch.stack(
                [
                    q0[:, start:end].norm(dim=-1)
                    for start, end in st["dim_slices"]
                ],
                dim=0,
            ).to(torch.float16)
            th_packed = torch.cat([th_loose, q_norms], dim=0)
            out_ours = attn.fn(
                q=q0.to(torch.float16), th_per_subspace=th_packed, state=st,
                buffer_keys=empty_buf.to(torch.float16),
                buffer_values=empty_val.to(torch.float16),
                keys_children=full_keys, q_head_to_kv=q_head_to_kv, scale=scale,
            )
            corr_abs[name] = (out_ours.float() - out_ref.float()).abs().max().item()
            if verbose:
                print(f"  correctness[{name:<16s}]: max_abs_diff={corr_abs[name]:.4e}")
    elif verbose:
        print("  correctness: skipped (no captured values available)")

    # Pruning stats.
    n_queries = int(q_batch.shape[0])
    stats: dict[str, tuple[float, float]] = {}
    for name, st in stats_states.items():
        kept_sum = 0.0
        recall_sum = 0.0
        for qi in range(n_queries):
            kept, recall = _pruning_stats(
                st, q_batch[qi], keys_expanded, full_keys,
                args.topk, q_head_to_kv,
            )
            kept_sum += kept
            recall_sum += recall
        stats[name] = (kept_sum / n_queries, recall_sum / n_queries)
        if verbose:
            print(f"  pruning[{name:<16s}]: kept_frac={stats[name][0]:.4f}  "
                  f"recall@{args.topk}={stats[name][1]:.4f}")

    out: dict = {
        "fresh (rebuild)": {
            "ms": ms_fresh,
            "ms_per_buf": ms_fresh_per_buf,
            "kept_frac": stats["fresh (rebuild)"][0],
            "recall": stats["fresh (rebuild)"][1],
            "corr_abs": None,
        }
    }
    for name in kernel_ms:
        out[name] = {
            "ms": kernel_ms[name],
            "ms_per_buf": _amortized_ms(kernel_ms[name], B),
            "kept_frac": stats[name][0],
            "recall": stats[name][1],
            "corr_abs": corr_abs.get(name),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--H-kv", type=int, default=8)
    p.add_argument("--H-q", type=int, default=24)
    p.add_argument(
        "--N",
        type=int,
        default=None,
        help="Base index size. Defaults to 4096 for synthetic mode, or the "
             "largest valid capture prefix when --input-qkv is used.",
    )
    p.add_argument("--B", type=int, default=32, help="buffer size")
    p.add_argument("--B-sweep", type=str, default=None,
                   help="Comma-separated buffer sizes to sweep (overrides --B)")
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--D-v", type=int, default=128)
    p.add_argument("--input-qkv", type=Path, default=None,
                   help="Path to a captured QKV .pt file. When set, the "
                        "benchmark uses a real decode trace instead of "
                        "synthetic random tensors.")
    p.add_argument("--layer", type=int, default=15,
                   help="Transformer layer to load from --input-qkv.")
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--topk", type=int, default=32,
                   help="Top-k used to derive tight thresholds for pruning stats")
    p.add_argument("--prune-queries", type=int, default=20,
                   help="Number of distinct queries to average pruning stats over")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if args.B_sweep is not None:
        b_list = [int(x) for x in args.B_sweep.split(",") if x.strip()]
    else:
        b_list = [args.B]
    max_b = max(b_list)

    device = "cuda"
    real_data = None
    if args.input_qkv is None:
        n_old = 4096 if args.N is None else args.N
        torch.manual_seed(0)
        keys = torch.randn(args.H_kv, n_old, args.D, device=device, dtype=torch.float32)
        values = torch.randn(args.H_kv, n_old, args.D_v, device=device, dtype=torch.float32)
        q_head_to_kv = _q_to_kv_map(args.H_q, args.H_kv, device)
        q_batch = torch.randn(args.prune_queries, args.H_q, args.D, device=device, dtype=torch.float32)
        q_batch = q_batch / q_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    else:
        real_data = _load_real_capture(args, max_b, device=device)
        keys = real_data["keys"]
        values = real_data["values"]
        q_head_to_kv = real_data["q_head_to_kv"]
        q_batch = None

    base_state = build_v2_7(keys, args.bf, args.S, args.refine_iter, values=values)

    h_q = int(q_head_to_kv.shape[0])
    h_kv = int(keys.shape[0])
    n_old = int(keys.shape[1])
    d = int(keys.shape[-1])
    d_v = int(values.shape[-1]) if values is not None else d

    if real_data is None:
        print(f"update micro-bench [synthetic]: H_q={h_q} H_kv={h_kv} N={n_old} "
              f"D={d} D_v={d_v} bf={args.bf} S={args.S}")
    else:
        print(f"update micro-bench [capture layer {real_data['layer']}]: "
              f"H_q={h_q} H_kv={h_kv} N={n_old} D={d} D_v={d_v} bf={args.bf} S={args.S}")
        print(f"capture={args.input_qkv}  prompt_length={real_data['prompt_length']}  "
              f"total_keys={real_data['total_keys']}")
    print("=" * 78)

    all_results: dict[int, dict] = {}
    verbose = len(b_list) == 1
    for B in b_list:
        buffer_keys = None
        buffer_values = None
        q_batch_B = q_batch
        q_indices = None
        if real_data is not None:
            buffer_keys, buffer_values, q_batch_B, q_indices = _slice_real_case(
                real_data, B, device=device, n_queries=args.prune_queries
            )
        if not verbose:
            print(f"[B={B}] ...")
        else:
            print(f"  B={B}")
            print("-" * 78)
            if q_indices is not None:
                print(f"    capture window: old_keys=[0:{n_old})  buffer=[{n_old}:{n_old + B})  "
                      f"generated_queries=[{q_indices[0]}:{q_indices[-1] + 1})")
        all_results[B] = _run_for_B(
            args, B, keys, values, base_state,
            q_head_to_kv, q_batch_B,
            buffer_keys=buffer_keys, buffer_values=buffer_values,
            verbose=verbose,
        )

    # Summary table across all Bs.
    print("=" * 78)
    print("Summary")
    header = (
        f"  {'B':>6s}  {'kernel':<18s}  {'ms/B':>10s}  "
        f"{'x_fresh':>8s}  {'kept':>7s}  {'recall':>7s}"
    )
    print(header)
    print("-" * 78)
    for B in b_list:
        res = all_results[B]
        ms_fresh = res["fresh (rebuild)"]["ms"]
        fresh = res["fresh (rebuild)"]
        print(f"  {B:>6d}  {'fresh (rebuild)':<18s}  {fresh['ms_per_buf']:>10.4f}  "
              f"{'1.0x':>8s}  {fresh['kept_frac']:>7.4f}  {fresh['recall']:>7.4f}")
        for name in SUMMARY_ORDER:
            if name not in res:
                continue
            r = res[name]
            print(f"  {B:>6d}  {name:<18s}  {r['ms_per_buf']:>10.4f}  "
                  f"{ms_fresh / r['ms']:>7.1f}x  "
                  f"{r['kept_frac']:>7.4f}  {r['recall']:>7.4f}")
        print()


if __name__ == "__main__":
    main()
