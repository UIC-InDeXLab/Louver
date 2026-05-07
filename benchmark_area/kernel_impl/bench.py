"""End-to-end benchmark: decoding simulation with the subspace k-center index.

For each decoding step i:
  1. (excluded from timing) Compute per-subspace thresholds from the true
     top-k set over all keys up to step i.
  2. (timed) index.attend(query, thresholds) — fused attention over index
     survivors + buffer keys/values.
  3. (timed) baseline_attention(query, all_keys, all_values) — dense softmax.
  4. (timed) baseline_sdpa — torch SDPA reference.
  5. Append the new (k, v) to the decoding buffer.
  6. Every `--update-interval` steps: index.update() — timed separately.

Reports (incremental CSV) in kernel_impl/reports/:
  step, n_keys, attend_ours_ms, dense_attn_ms, sdpa_ms,
  update_ms, amortized_ours_ms, memory_bytes

Usage:
    python -m hira.benchmark_area.kernel_impl.bench \\
        --input-qkv capture.pt --n-steps 2000 --update-interval 256
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.index import (
    IndexConfig,
    SubspaceKCenterIndex,
    baseline_attention,
    baseline_sdpa,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)

DEFAULT_PROMPT = "Benchmark the index over a long decoding trace."
_GREEN = "\033[32m"
_RESET = "\033[0m"
_ATTN_BUCKETS = (64, 128, 256, 512)

# attend() runs before the buffer append in the decode loop, so buffer size
# reaches `update_interval - 1` right before the flush. The attention kernels
# cap buffer at _BUCKETS[-1] (= 512).
_MAX_BUFFER = _ATTN_BUCKETS[-1]
_MAX_UPDATE_INTERVAL = _MAX_BUFFER + 1


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Derive per-subspace thresholds from the full-space top-k set."""
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
    return torch.stack(ths, dim=0)


def packed_subspace_topk_thresholds_fp16(q, keys, topk, dim_slices):
    """Return packed ``(2*S, H_q)`` fp16 thresholds + per-subspace q norms."""
    if q.dtype != torch.float16 or keys.dtype != torch.float16:
        raise RuntimeError(
            "bench.py only supports fp16 query/key inputs for fused attention. "
            f"Got q={q.dtype}, keys={keys.dtype}."
        )
    th = subspace_topk_thresholds(q, keys, topk, dim_slices)
    q_norms = torch.stack(
        [q[:, start:end].norm(dim=-1) for start, end in dim_slices],
        dim=0,
    )
    return torch.cat([th, q_norms], dim=0).contiguous()


def _time_gpu(fn):
    """Single-shot timing. Use for mutating ops (update) where looping is unsafe."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000  # ms


def _time_repeated(fn, iters=10, warmup=3):
    """Avg ms/call across ``iters`` runs with one sync per batch.

    Mirrors ``bench_attention.time_call`` so the per-kernel-launch Python+driver
    sync overhead is amortized instead of dominating (matters for 50-100µs calls).
    Only safe for read-only ops — do not pass anything that mutates index state.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _format_step_ms(value: float, fastest: float) -> str:
    text = f"{value:.3f}ms"
    if value == fastest:
        return f"{_GREEN}{text}{_RESET}"
    return text


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end decoding simulation for the subspace k-center index."
    )

    # ── Input source ──
    p.add_argument("--input-qkv", type=Path, default=None,
                   help="Path to a captured QKV .pt file. If omitted, captures "
                        "fresh from --model.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF model id used when capturing QKV live.")
    p.add_argument("--n-tokens", type=int, default=2000,
                   help="Tokens to capture when --input-qkv is not given.")
    p.add_argument("--layer", type=int, default=15,
                   help="Which transformer layer's Q/K/V to simulate.")

    # ── Simulation window ──
    p.add_argument("--prefill-frac", type=float, default=0.5,
                   help="Fraction of total captured keys treated as prefill "
                        "(index is built on these). The remainder is the "
                        "decoding trace.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Max number of decoding steps to simulate. Defaults to "
                        "all available tokens after prefill. Actual count is "
                        "min(--n-steps, remaining keys, available generated queries).")

    # ── Threshold ──
    p.add_argument("--topk", type=int, default=20,
                   help="k for top-k-derived per-subspace thresholds (threshold "
                        "finding is excluded from timing).")

    # ── Index config ──
    p.add_argument("--bf", type=int, default=4,
                   help="Branching factor: cluster size target (K = ceil(N/bf)).")
    p.add_argument("--n-subspaces", type=int, default=8,
                   help="Number of contiguous dim splits.")
    p.add_argument("--refine-iter", type=int, default=5,
                   help="Lloyd refinement iterations per subspace during build.")
    p.add_argument("--update-mode", choices=["full", "inc"], default="inc",
                   help='"full": rebuild index on all keys via build kernel. '
                        '"inc": mini-index the buffer and merge (update kernel).')
    p.add_argument("--update-interval", type=int, default=256,
                   help=f"Flush the decoding buffer into the index every N "
                        f"steps. Must satisfy 1 <= N <= {_MAX_UPDATE_INTERVAL} "
                        f"(attention kernels bucket the buffer at "
                        f"{_ATTN_BUCKETS}; bucket-aligned values "
                        f"{_ATTN_BUCKETS} minimize CUDA-graph captures).")

    # ── Kernel selection (defaults = fused attention path) ──
    p.add_argument("--build-kernel", default="build_v2_7",
                   help="Module name under kernels/ for build (auto-discovered).")
    p.add_argument("--update-kernel", default=None,
                   help="Module name under kernels/ for update (auto-discovered). "
                        "Defaults to update_v4_0.")
    p.add_argument("--attention-kernel", default="attention_v5_14",
                   help="Module name under kernels/ for fused attention.")
    p.add_argument("--parallel-update", action="store_true",
                   help="Run update on a side CUDA stream concurrent with "
                        "attention. Requires an overlap-aware update kernel "
                        "(default: update_v4_0). Adds step_wall_ms / overlap "
                        "telemetry columns to the CSV.")
    p.add_argument("--update-stream-priority", type=int, default=-1,
                   help="CUDA stream priority for the update stream when "
                        "--parallel-update is set. Lower = lower priority. "
                        "Default -1 lets attention preempt SMs.")

    # ── Output ──
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Defaults to kernel_impl/reports/bench_<update_mode>.csv.")
    p.add_argument("--flush-every", type=int, default=50,
                   help="Flush CSV every N decoding steps.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for k-center seeding (affects build).")
    return p.parse_args()


def load_qkv(args):
    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        cap = _capture_qkv(
            model_name=args.model, prompt_text=DEFAULT_PROMPT,
            n=args.n_tokens, device="cuda",
            torch_dtype=torch.float16, show_progress=True,
        )
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError(
            "Fused attention requires captured values. Re-capture with values."
        )
    return queries_cpu, keys_cpu, values_cpu, layer


def _validate_args(args):
    if args.update_interval < 1 or args.update_interval > _MAX_UPDATE_INTERVAL:
        raise ValueError(
            f"--update-interval={args.update_interval} out of range. "
            f"attention kernels cap buffer at {_MAX_BUFFER} (buckets "
            f"{_ATTN_BUCKETS}), and buffer reaches update_interval-1 before "
            f"flush, so update_interval must be in [1, {_MAX_UPDATE_INTERVAL}]."
        )


def _bucket_for_buffer(l_buf: int) -> int:
    for b in _ATTN_BUCKETS:
        if l_buf <= b:
            return b
    return _ATTN_BUCKETS[-1]


def main():
    args = parse_args()
    _validate_args(args)
    if args.update_kernel is None:
        args.update_kernel = "update_v4_0"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    queries_cpu, keys_cpu, values_cpu, layer = load_qkv(args)
    # Stay in fp16 for a fair comparison: captures are fp16, our attend kernel
    # packs fp16 internally, and SDPA's Flash/mem-efficient backends require
    # fp16/bf16 (fp32 falls back to the `math` backend and underperforms).
    dtype = torch.float16
    keys = keys_cpu.to(device="cuda", dtype=dtype)
    queries = queries_cpu.to(device="cuda", dtype=dtype)
    values = values_cpu.to(device="cuda", dtype=dtype)
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("Not enough keys for decoding — adjust --prefill-frac.")

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill_keys={n_prefill}  decoding_steps={n_decode}")

    # ── Build index on prefill keys/values ──
    cfg = IndexConfig(
        n_subspaces=args.n_subspaces, bf=args.bf, refine_iter=args.refine_iter,
        update_mode=args.update_mode,
        build_kernel=args.build_kernel,
        update_kernel=args.update_kernel,
        attention_kernel=args.attention_kernel,
        parallel_update=args.parallel_update,
        update_stream_priority=args.update_stream_priority,
    )
    index = SubspaceKCenterIndex(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    print(
        f"Build: {build_ms:.1f} ms  "
        f"(build={args.build_kernel}, update={args.update_kernel}, "
        f"attn={args.attention_kernel})"
    )

    # ── Correctness check: fused attend vs. dense attention, on first step. ──
    q0 = queries[:, n_prefill, :]
    qn0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    all_keys_check = keys[:, : n_prefill, :]
    all_values_check = values[:, : n_prefill, :]
    dim_slices = index.state["dim_slices"]
    keys_eval0 = all_keys_check if q_head_to_kv is None else all_keys_check[q_head_to_kv]
    th0 = packed_subspace_topk_thresholds_fp16(
        qn0, keys_eval0, args.topk, dim_slices
    )
    out_ours = index.attend(qn0, th0, q_head_to_kv=q_head_to_kv)
    out_ref = baseline_attention(
        qn0, all_keys_check, all_values_check, q_head_to_kv=q_head_to_kv
    )
    diff = (out_ours.float() - out_ref.float()).abs().max().item()
    rel = diff / (out_ref.float().abs().max().item() + 1e-9)
    print(f"Correctness (attend vs dense): max_abs_diff={diff:.4e}  rel={rel:.4e}")

    # ── Prep output CSV ──
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"bench_{args.update_mode}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys", "k_used", "k_cap",
        "attend_ours_ms", "dense_attn_ms", "sdpa_ms",
        "update_ms", "amortized_ours_ms", "memory_bytes",
        "scanned_parent_frac", "scanned_key_frac",
        # Parallel-update telemetry (zeros when --parallel-update is off).
        "step_wall_ms",          # total wall-clock for the step (event-based)
        "update_kernel_ms",      # GPU kernel time on update_stream (committed at publish step)
        "update_wait_ms",        # host stall when next fire saw a still-pending update
        "update_inflight",       # 1 if a prior update was still pending at start of this step
        "buffer_bucket",         # attention buffer bucket size used (64/128/256/512/0)
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []          # per-step update GPU time (kernel_ms in async, blocking ms in sync)
    update_fire_steps: list[int] = []
    update_wait_costs: list[float] = []
    update_kernel_costs: list[float] = []
    last_update_ms = 0.0

    # Per-bucket bookkeeping for the end-of-run summary (async path only).
    bucket_stats: dict[int, dict[str, float]] = {}
    # Attend timing split (idle vs during-update) — async only.
    attend_idle_sum = attend_idle_n = 0.0, 0
    attend_busy_sum = attend_busy_n = 0.0, 0
    # Use mutable accumulators because we set them via `+=`.
    attend_idle_sum = 0.0; attend_idle_n = 0
    attend_busy_sum = 0.0; attend_busy_n = 0

    parallel = args.parallel_update
    sim_start = time.perf_counter()

    # Aggregates for end-of-run wall vs serial summary.
    sum_wall_ms = 0.0
    sum_attend_ms = 0.0
    sum_dense_ms = 0.0
    sum_sdpa_ms = 0.0
    sum_attend_after_fire_ms = 0.0   # attend during steps where publish happened mid-loop

    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Step start: was a prior update still in flight when this step began?
        update_inflight_at_start = bool(parallel and index.has_pending_update)

        # Lazy publish: if the prior update is finished on the GPU, publish it
        # now (cheap, no host stall). Without this every subsequent fire would
        # see _pending_update=True even when the GPU has long since finished.
        publish_this_step_metrics = None
        if parallel:
            pre_log_n = len(index.update_metrics_log)
            if index.try_publish() and len(index.update_metrics_log) > pre_log_n:
                m = index.update_metrics_log[-1]
                publish_this_step_metrics = m

        # Step-level wall-clock event (event-based to avoid per-step host syncs).
        step_start_evt = torch.cuda.Event(enable_timing=True)
        step_end_evt = torch.cuda.Event(enable_timing=True)
        step_start_evt.record()

        # ── Threshold computation (NOT timed) ──
        all_keys_so_far = keys[:, : token_idx + 1, :]
        all_values_so_far = values[:, : token_idx + 1, :]
        keys_eval = all_keys_so_far if q_head_to_kv is None else all_keys_so_far[q_head_to_kv]
        th = packed_subspace_topk_thresholds_fp16(
            qn, keys_eval, args.topk, index.state["dim_slices"]
        )

        # ── Timed: our fused attention ──
        attend_ours_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )
        if parallel:
            if update_inflight_at_start:
                attend_busy_sum += attend_ours_ms
                attend_busy_n += 1
            else:
                attend_idle_sum += attend_ours_ms
                attend_idle_n += 1

        # Track the buffer-bucket attention actually used this step
        # (covers active + inflight during overlap).
        l_buf_eff = int(index.n_buffered)
        bucket = _bucket_for_buffer(l_buf_eff) if l_buf_eff > 0 else 0
        k_used = int(index.state.get("K_used", index.state["K"]))
        k_cap = int(index.state.get("K_cap", index.state["K"]))

        # ── Pruning measurement (NOT timed) ──
        cp = index.last_cluster_pass()
        if cp is not None:
            parent_alive = cp.all(dim=0)[:, :k_used]   # (H_q, K_used)
            scanned_parent_frac = parent_alive.float().mean().item()
            bf = int(index.state["bf"])
            n_idx = k_used * bf
            n_buf = int(index.n_buffered)
            scanned_key_frac = (
                (scanned_parent_frac * n_idx + n_buf) / max(n_idx + n_buf, 1)
            )
        else:
            scanned_parent_frac = float("nan")
            scanned_key_frac = float("nan")

        # ── Timed: dense attention baseline ──
        dense_attn_ms = _time_repeated(
            lambda: baseline_attention(
                qn, all_keys_so_far, all_values_so_far, q_head_to_kv
            )
        )
        # ── Timed: SDPA baseline ──
        sdpa_ms = _time_repeated(
            lambda: baseline_sdpa(
                qn, all_keys_so_far, all_values_so_far, q_head_to_kv
            )
        )

        # ── Append new (k, v) to buffer ──
        new_key = keys[:, token_idx : token_idx + 1, :]
        new_val = values[:, token_idx : token_idx + 1, :]
        index.append_decoding_kv(new_key, new_val)

        # ── Update every update_interval steps ──
        update_ms = 0.0
        update_kernel_ms_col = 0.0
        update_wait_ms_col = 0.0
        if publish_this_step_metrics is not None:
            # Publish was lazy / non-blocking. Surface the kernel time on this row.
            update_kernel_ms_col = publish_this_step_metrics.kernel_ms
            update_wait_ms_col = publish_this_step_metrics.host_wait_ms
            update_kernel_costs.append(publish_this_step_metrics.kernel_ms)
            update_wait_costs.append(publish_this_step_metrics.host_wait_ms)
            last_update_ms = publish_this_step_metrics.kernel_ms

        if index.needs_update(args.update_interval):
            if parallel:
                pre_log_n = len(index.update_metrics_log)
                index.update_async(fire_step=step)
                update_fire_steps.append(step)
                if len(index.update_metrics_log) > pre_log_n:
                    # Forced publish inside update_async because prior update
                    # wasn't done — host actually waited.
                    m = index.update_metrics_log[-1]
                    if publish_this_step_metrics is None:
                        # Distinct from the lazy publish above.
                        update_kernel_ms_col = m.kernel_ms
                        update_wait_ms_col = m.host_wait_ms
                        update_kernel_costs.append(m.kernel_ms)
                        update_wait_costs.append(m.host_wait_ms)
                    last_update_ms = m.kernel_ms
                    if bucket not in bucket_stats:
                        bucket_stats[bucket] = {"n": 0, "kernel_ms": 0.0, "wait_ms": 0.0}
                    bucket_stats[bucket]["n"] += 1
                    bucket_stats[bucket]["kernel_ms"] += m.kernel_ms
                    bucket_stats[bucket]["wait_ms"] += m.host_wait_ms
                update_ms = update_kernel_ms_col
            else:
                _, update_ms = _time_gpu(index.update)
                last_update_ms = update_ms
                update_kernel_ms_col = update_ms
                update_kernel_costs.append(update_ms)
                update_fire_steps.append(step)
        update_costs.append(update_ms)

        # End-of-step event + sync (only sync if needed for wall measurement).
        step_end_evt.record()
        step_end_evt.synchronize()
        step_wall_ms = step_start_evt.elapsed_time(step_end_evt)
        sum_wall_ms += step_wall_ms
        sum_attend_ms += attend_ours_ms
        sum_dense_ms += dense_attn_ms
        sum_sdpa_ms += sdpa_ms

        # Amortized: serial cost includes update kernel time amortized over all steps.
        amort_update_ms = sum(update_costs) / len(update_costs)
        amort_ours = attend_ours_ms + amort_update_ms

        rows.append({
            "step": step,
            "n_keys": int(all_keys_so_far.shape[1]),
            "k_used": k_used,
            "k_cap": k_cap,
            "attend_ours_ms": round(attend_ours_ms, 4),
            "dense_attn_ms": round(dense_attn_ms, 4),
            "sdpa_ms": round(sdpa_ms, 4),
            "update_ms": round(update_ms, 4),
            "amortized_ours_ms": round(amort_ours, 4),
            "memory_bytes": index.memory_bytes(),
            "scanned_parent_frac": round(scanned_parent_frac, 5),
            "scanned_key_frac": round(scanned_key_frac, 5),
            "step_wall_ms": round(step_wall_ms, 4),
            "update_kernel_ms": round(update_kernel_ms_col, 4),
            "update_wait_ms": round(update_wait_ms_col, 4),
            "update_inflight": int(update_inflight_at_start),
            "buffer_bucket": bucket,
        })

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()
            elapsed = time.perf_counter() - sim_start
            fastest_step_ms = min(attend_ours_ms, dense_attn_ms, sdpa_ms)
            print(
                f"step {step+1}/{n_decode}  "
                f"attend={_format_step_ms(attend_ours_ms, fastest_step_ms)}  "
                f"wall={step_wall_ms:.3f}ms  "
                f"dense={_format_step_ms(dense_attn_ms, fastest_step_ms)}  "
                f"sdpa={_format_step_ms(sdpa_ms, fastest_step_ms)}  "
                f"last_upd={last_update_ms:.2f}ms  "
                f"K={k_used}/{k_cap}  "
                f"scan[parents]={scanned_parent_frac:.3f} "
                f"scan[keys]={scanned_key_frac:.3f}  [{elapsed:.1f}s]"
            )

    # ── Drain any in-flight update so its metrics are committed ──
    if parallel and index.has_pending_update:
        index.wait_for_update()
        # If publish committed metrics, attribute them to a synthetic post-loop entry.
        if len(index.update_metrics_log) > len(update_kernel_costs):
            m = index.update_metrics_log[-1]
            update_kernel_costs.append(m.kernel_ms)
            update_wait_costs.append(m.host_wait_ms)

    # ── End-of-run summary ──
    n_upd = len(update_fire_steps)
    total_kernel_ms = sum(update_kernel_costs)
    mean_kernel_ms = total_kernel_ms / max(n_upd, 1)
    total_wait_ms = sum(update_wait_costs)
    max_wait_ms = max(update_wait_costs) if update_wait_costs else 0.0
    mean_attend_ms = sum_attend_ms / max(n_decode, 1)
    mean_dense_ms = sum_dense_ms / max(n_decode, 1)
    mean_sdpa_ms = sum_sdpa_ms / max(n_decode, 1)
    mean_baselines_ms = mean_dense_ms + mean_sdpa_ms

    print(
        f"\n──── Summary ────"
        f"\nUpdates: {n_upd} fired over {n_decode} steps "
        f"(interval={args.update_interval}, mode={args.update_mode}, "
        f"parallel={parallel})"
    )
    print(f"  avg attention time: {mean_attend_ms:.4f}ms/step")
    print(
        f"  avg baselines time: {mean_baselines_ms:.4f}ms/step "
        f"(dense={mean_dense_ms:.4f}, sdpa={mean_sdpa_ms:.4f})"
    )
    print(
        f"  update kernel time:  total={total_kernel_ms:.1f}ms  "
        f"mean={mean_kernel_ms:.2f}ms/update  "
        f"amortized={total_kernel_ms / max(n_decode, 1):.4f}ms/step"
    )

    if parallel:
        print(
            f"  overlap misses:      {index.n_overlap_misses}/{n_upd}  "
            f"(updates that forced a host stall on the next fire)"
        )
        print(
            f"  host stall:          total={total_wait_ms:.1f}ms  "
            f"mean={total_wait_ms / max(n_upd, 1):.2f}ms/update  "
            f"max={max_wait_ms:.2f}ms"
        )
        # Headline: amortized cost of (attend + update) under serial vs overlap.
        # Serial cost we can construct from the same data: mean attend + mean
        # update kernel time amortized over decode steps (this is what the
        # serial bench's --update-mode=inc would charge per step).
        mean_wall_ms = sum_wall_ms / max(n_decode, 1)
        amort_kernel_ms = total_kernel_ms / max(n_decode, 1)
        serial_cost = mean_attend_ms + amort_kernel_ms
        # Overlap cost ≈ mean wall - dense - sdpa - threshold work (we can't
        # decompose all of it cheaply). Report mean wall directly; the
        # interesting comparison is per-step "how much of update_kernel_ms is
        # hidden by attend".
        hide_ms = max(0.0, amort_kernel_ms - max(0.0, mean_wall_ms - mean_attend_ms))
        # Cleaner derivation: hide ratio = 1 - host_stall_amortized / kernel_amortized
        amort_wait_ms = total_wait_ms / max(n_decode, 1)
        denom = amort_kernel_ms if amort_kernel_ms > 0 else 1.0
        hide_ratio = 1.0 - amort_wait_ms / denom
        hide_ratio = max(0.0, min(1.0, hide_ratio))
        print(
            f"  serial cost (proxy): attend({mean_attend_ms*1000:.1f}us) + "
            f"update_amortized({amort_kernel_ms*1000:.1f}us) = "
            f"{serial_cost*1000:.1f}us/step"
        )
        print(
            f"  hide ratio:          {hide_ratio*100:.1f}%  "
            f"(of update kernel time hidden behind attend; "
            f"100% = no host stall, 0% = fully serialized)"
        )
        print(
            f"  mean step wall:      {mean_wall_ms:.3f}ms  "
            f"(includes dense + sdpa baselines, not just our attend)"
        )
        if attend_idle_n and attend_busy_n:
            mean_idle = attend_idle_sum / attend_idle_n
            mean_busy = attend_busy_sum / attend_busy_n
            ratio = mean_busy / mean_idle if mean_idle > 0 else float("nan")
            print(
                f"  attend contention:   idle={mean_idle:.4f}ms ({attend_idle_n} steps)  "
                f"during_update={mean_busy:.4f}ms ({attend_busy_n} steps)  "
                f"ratio={ratio:.3f}x"
            )
        if bucket_stats:
            print("  per-bucket update kernel time:")
            for b in sorted(bucket_stats):
                bs = bucket_stats[b]
                if bs["n"] == 0:
                    continue
                print(
                    f"    B<={b:>4d}: n={int(bs['n']):3d}  "
                    f"mean_kernel={bs['kernel_ms']/bs['n']:.2f}ms  "
                    f"mean_wait={bs['wait_ms']/bs['n']:.2f}ms"
                )

    print(f"Done. CSV -> {out_csv}")


if __name__ == "__main__":
    main()
