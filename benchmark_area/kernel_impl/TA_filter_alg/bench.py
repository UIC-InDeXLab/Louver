"""End-to-end TA-filter decoding benchmark.

For each decoding step:
    1. (NOT timed) per-head scalar threshold T = topk-th largest exact full
       dot product per head.
    2. (timed) ``index.attend`` — fused filter + sparse-attn (v2.4 buffer-arena
       in Variant B; v2.5 buffer-aware kernel in Variant A).
    3. (timed) dense baseline.
    4. (timed) torch SDPA baseline.
    5. append (k, v) to the buffer.
    6. every BUFFER_SIZE=256 steps: ``index.update`` (sync) or
       ``index.update_async`` (parallel) — fast .cu cluster kernel.

Hardcoded: bf=4, S=4, BUFFER=256.

Usage:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.bench \\
        --input-qkv capture.pt --n-steps 1000 --variant B --parallel-update
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

from hira.benchmark_area.kernel_impl.TA_filter_alg.index import (
    BUFFER_SIZE,
    TAIndex,
    TAIndexConfig,
    baseline_attention,
    baseline_sdpa,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)


_GREEN = "\033[32m"
_RESET = "\033[0m"
DEFAULT_PROMPT = "Benchmark the TA-filter index over a long decoding trace."


def topk_full_dot_threshold(
    q: torch.Tensor, keys: torch.Tensor,
    q_head_to_kv: torch.Tensor | None, topk: int,
) -> torch.Tensor:
    keys_eval = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def _time_gpu(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000


def _time_repeated(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _format(value: float, fastest: float) -> str:
    text = f"{value:.3f}ms"
    return f"{_GREEN}{text}{_RESET}" if value == fastest else text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=2000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--prefill-frac", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--topk", type=int, default=20,
                   help="threshold = topk-th largest exact full-dim dot per head.")
    p.add_argument("--n-growth", type=int, default=None,
                   help="Arena over-allocation in keys (defaults to n_decode + BUFFER_SIZE).")
    p.add_argument("--parallel-update", action="store_true",
                   help="Run update on side stream concurrent with attention.")
    p.add_argument("--update-stream-priority", type=int, default=-1)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
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
    q, k, v = cap.to_layer_tensors(layer)
    if v is None:
        raise RuntimeError("TA-filter requires captured values.")
    return q, k, v, layer


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    queries_cpu, keys_cpu, values_cpu, layer = load_qkv(args)
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
    n_growth = args.n_growth if args.n_growth is not None else n_decode + BUFFER_SIZE

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill_keys={n_prefill}  decoding_steps={n_decode}  n_growth={n_growth}  "
          f"parallel_update={args.parallel_update}")

    cfg = TAIndexConfig(
        n_growth=n_growth,
        parallel_update=args.parallel_update,
        update_stream_priority=args.update_stream_priority,
    )
    index = TAIndex(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"Build: {build_ms:.1f} ms  (TA_build, K_cap={index.state['K_cap']})")

    # Correctness on first decode step.
    q0 = queries[:, n_prefill, :]
    qn0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_check = keys[:, :n_prefill, :]
    values_check = values[:, :n_prefill, :]
    th0 = topk_full_dot_threshold(qn0, keys_check, q_head_to_kv, args.topk).to(torch.float32)
    out_ours = index.attend(qn0, th0, q_head_to_kv=q_head_to_kv)
    out_ref = baseline_attention(qn0, keys_check, values_check, q_head_to_kv=q_head_to_kv)
    diff = (out_ours.float() - out_ref.float()).abs().max().item()
    rel = diff / (out_ref.float().abs().max().item() + 1e-9)
    print(f"Correctness: max_abs_diff={diff:.4e}  rel={rel:.4e}")

    out_csv = args.output_csv or (
        Path(__file__).parent.parent / "reports" / "bench_TA_filter.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys",
        "attend_ours_ms", "dense_attn_ms", "sdpa_ms",
        "update_ms", "amortized_ours_ms", "memory_bytes",
        "k_used", "k_cap", "buffer_len",
        "step_wall_ms", "update_kernel_ms", "update_wait_ms",
        "update_inflight",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    update_costs: list[float] = []
    update_kernel_costs: list[float] = []
    update_wait_costs: list[float] = []
    update_fire_steps: list[int] = []
    last_update_ms = 0.0
    parallel = args.parallel_update

    sum_wall_ms = 0.0
    sum_attend_ms = 0.0
    sum_dense_ms = 0.0
    sum_sdpa_ms = 0.0

    # Attend timing split (idle vs during-update) — async only.
    attend_idle_sum = 0.0; attend_idle_n = 0
    attend_busy_sum = 0.0; attend_busy_n = 0

    sim_start = time.perf_counter()

    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        update_inflight_at_start = bool(parallel and index.has_pending_update)

        # Lazy publish if prior update finished without forcing a stall.
        publish_this_step = None
        if parallel:
            pre_log_n = len(index.update_metrics_log)
            if index.try_publish() and len(index.update_metrics_log) > pre_log_n:
                publish_this_step = index.update_metrics_log[-1]

        step_start_evt = torch.cuda.Event(enable_timing=True)
        step_end_evt = torch.cuda.Event(enable_timing=True)
        step_start_evt.record()

        all_keys_so_far = keys[:, :token_idx + 1, :]
        all_values_so_far = values[:, :token_idx + 1, :]
        th = topk_full_dot_threshold(qn, all_keys_so_far, q_head_to_kv, args.topk).to(torch.float32)

        attend_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )
        if parallel:
            if update_inflight_at_start:
                attend_busy_sum += attend_ms; attend_busy_n += 1
            else:
                attend_idle_sum += attend_ms; attend_idle_n += 1
        dense_ms = _time_repeated(
            lambda: baseline_attention(qn, all_keys_so_far, all_values_so_far, q_head_to_kv)
        )
        sdpa_ms = _time_repeated(
            lambda: baseline_sdpa(qn, all_keys_so_far, all_values_so_far, q_head_to_kv)
        )

        # Append to buffer
        new_k = keys[:, token_idx:token_idx + 1, :]
        new_v = values[:, token_idx:token_idx + 1, :]
        index.append_decoding_kv(new_k, new_v)

        update_ms = 0.0
        update_kernel_ms_col = 0.0
        update_wait_ms_col = 0.0
        if publish_this_step is not None:
            update_kernel_ms_col = publish_this_step.kernel_ms
            update_wait_ms_col = publish_this_step.host_wait_ms
            update_kernel_costs.append(publish_this_step.kernel_ms)
            update_wait_costs.append(publish_this_step.host_wait_ms)
            last_update_ms = publish_this_step.kernel_ms

        if index.needs_update():
            if parallel:
                pre_log_n = len(index.update_metrics_log)
                index.update_async(fire_step=step)
                update_fire_steps.append(step)
                if len(index.update_metrics_log) > pre_log_n:
                    m = index.update_metrics_log[-1]
                    if publish_this_step is None:
                        update_kernel_ms_col = m.kernel_ms
                        update_wait_ms_col = m.host_wait_ms
                        update_kernel_costs.append(m.kernel_ms)
                        update_wait_costs.append(m.host_wait_ms)
                    last_update_ms = m.kernel_ms
                update_ms = update_kernel_ms_col
            else:
                _, update_ms = _time_gpu(index.update)
                last_update_ms = update_ms
                update_kernel_ms_col = update_ms
                update_kernel_costs.append(update_ms)
                update_fire_steps.append(step)
        update_costs.append(update_ms)

        step_end_evt.record()
        step_end_evt.synchronize()
        step_wall_ms = step_start_evt.elapsed_time(step_end_evt)
        sum_wall_ms += step_wall_ms
        sum_attend_ms += attend_ms
        sum_dense_ms += dense_ms
        sum_sdpa_ms += sdpa_ms

        amort = attend_ms + sum(update_costs) / len(update_costs)

        rows.append({
            "step": step,
            "n_keys": int(all_keys_so_far.shape[1]),
            "attend_ours_ms": round(attend_ms, 4),
            "dense_attn_ms": round(dense_ms, 4),
            "sdpa_ms": round(sdpa_ms, 4),
            "update_ms": round(update_ms, 4),
            "amortized_ours_ms": round(amort, 4),
            "memory_bytes": index.memory_bytes(),
            "k_used": int(index.state["K_used"]),
            "k_cap": int(index.state["K_cap"]),
            "buffer_len": index.n_buffered,
            "step_wall_ms": round(step_wall_ms, 4),
            "update_kernel_ms": round(update_kernel_ms_col, 4),
            "update_wait_ms": round(update_wait_ms_col, 4),
            "update_inflight": int(update_inflight_at_start),
        })

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()
            elapsed = time.perf_counter() - sim_start
            fastest = min(attend_ms, dense_ms, sdpa_ms)
            print(
                f"step {step+1}/{n_decode}  "
                f"attend={_format(attend_ms, fastest)}  "
                f"wall={step_wall_ms:.3f}ms  "
                f"dense={_format(dense_ms, fastest)}  "
                f"sdpa={_format(sdpa_ms, fastest)}  "
                f"last_upd={last_update_ms:.2f}ms  "
                f"K={index.state['K_used']}/{index.state['K_cap']}  "
                f"buf={index.n_buffered}  [{elapsed:.1f}s]"
            )

    # Drain any in-flight update.
    if parallel and index.has_pending_update:
        index.wait_for_update()
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

    print(
        f"\n──── Summary ────"
        f"\nparallel_update={parallel}"
        f"\nUpdates: {n_upd} fired over {n_decode} steps"
    )
    print(f"  avg attend time:  {mean_attend_ms:.4f}ms/step")
    print(f"  avg dense:        {mean_dense_ms:.4f}ms/step")
    print(f"  avg sdpa fp16:    {mean_sdpa_ms:.4f}ms/step")
    print(
        f"  update kernel:    total={total_kernel_ms:.1f}ms  "
        f"mean={mean_kernel_ms:.2f}ms/update  "
        f"amortized={total_kernel_ms / max(n_decode, 1):.4f}ms/step"
    )

    if parallel:
        print(
            f"  overlap misses:   {index.n_overlap_misses}/{n_upd}  "
            f"(updates that forced a host stall on the next fire)"
        )
        print(
            f"  host stall:       total={total_wait_ms:.2f}ms  "
            f"mean={total_wait_ms / max(n_upd, 1):.3f}ms/update  "
            f"max={max_wait_ms:.3f}ms"
        )
        amort_kernel_ms = total_kernel_ms / max(n_decode, 1)
        amort_wait_ms = total_wait_ms / max(n_decode, 1)
        denom = amort_kernel_ms if amort_kernel_ms > 0 else 1.0
        hide_ratio = max(0.0, min(1.0, 1.0 - amort_wait_ms / denom))
        print(
            f"  HIDE RATIO:       {hide_ratio*100:.1f}%  "
            f"(of update kernel time hidden behind attend; "
            f"100% = update fully concurrent, 0% = fully serialized)"
        )
        print(
            f"  → was update REALLY parallel?  "
            f"{'YES — fully hidden.' if hide_ratio > 0.95 else (
                'PARTIAL — host-stalled occasionally.' if hide_ratio > 0.5
                else 'NO — mostly blocking decode.')}"
        )
        if attend_idle_n and attend_busy_n:
            mean_idle = attend_idle_sum / attend_idle_n
            mean_busy = attend_busy_sum / attend_busy_n
            ratio = mean_busy / mean_idle if mean_idle > 0 else float("nan")
            print(
                f"  attend contention: idle={mean_idle:.4f}ms ({attend_idle_n} steps)  "
                f"during_update={mean_busy:.4f}ms ({attend_busy_n} steps)  "
                f"ratio={ratio:.3f}x"
            )

    print(f"Done. CSV -> {out_csv}")


if __name__ == "__main__":
    main()
