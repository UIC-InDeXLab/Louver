"""
GPU decoding latency benchmark: Louver vs dense baselines.

Per decode step (timed on GPU):
    louver      — fused TA-filter + sparse attention kernel
    dense_eager — manual Q @ K.T + softmax + @ V  (eager PyTorch)
    dense_flash — torch SDPA (FlashAttention-2 backend)
    twilight    — SDPA scores → softmax top-p select → masked V  (O(N), fair approx of optimized Twilight)

NOT timed: oracle threshold computation (top-k on full dot product).
Update cost is reported separately and also amortized into louver.

Baselines from: hira/benchmark_area/kernel_impl/TA_filter_alg/index.py
Index:          hira/benchmark_area/kernel_impl/TA_filter_alg/index.py  (TAIndex)
Captures:       benchmark_area/experiments/latency/captures/*.pt
                or benchmark_area/quick_pruning/capture_qkv_*.pt (smoke test)

Smoke-test command:
    python gpu_bench.py \\
        --input-qkv ../../quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \\
        --n-steps 200

Full-benchmark command (after capture_all.sh):
    python gpu_bench.py --input-qkv captures/<model>_layer<L>_N<n>.pt
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
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
    _q_to_kv_map,
)

_GREEN = "\033[32m"
_RESET = "\033[0m"

# ── Baseline helpers ──────────────────────────────────────────────────────────

def _sdpa_flash(q, keys, values, q_head_to_kv=None, scale=None):
    """SDPA explicitly requesting the FlashAttention-2 backend."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel  # type: ignore
        ctx: contextlib.AbstractContextManager = sdpa_kernel([SDPBackend.FLASH_ATTENTION])
    except (ImportError, AttributeError):
        ctx = contextlib.nullcontext()
    with ctx:
        return baseline_sdpa(q, keys, values, q_head_to_kv, scale)


def _twilight_attention(q, keys, values, top_p: float = 0.85, scale=None):
    """
    Twilight-style attention: full SDPA (flash backend) to get probs → top-p
    sparse gather → renormalized V.

    The Twilight pyimpl (twilight/pyimpl/attention.py) is explicitly labeled
    "for testing accuracy, not efficiency" and uses plain torch.matmul O(N).
    Their optimized path would use FlashAttention for the QK pass (same cost as
    dense_flash) then a CUDA top-p kernel.  We approximate that cost here:
      - SDPA flash backend computes full softmax probs efficiently (O(N) HBM)
      - top-p sort + scatter on GPU to identify kept indices
      - sparse V gather + weighted sum (bmm on kept subset)
    This is the fairest achievable Python approximation of the Twilight kernel.
    """
    h_q, d = q.shape
    h_kv, n, d_v = values.shape
    scale = d ** -0.5 if scale is None else float(scale)
    G = h_q // h_kv
    dtype = values.dtype

    # ── Step 1: full softmax probs via SDPA flash (same cost as dense_flash) ──
    # SDPA expects (B, H, S_q, D); we use B=1, S_q=1
    q4   = q.unsqueeze(0).unsqueeze(2)                         # (1, H_q, 1, D)
    k4   = keys.unsqueeze(0)                                   # (1, H_kv, N, D)
    v4   = values.unsqueeze(0)                                 # (1, H_kv, N, D_v)
    if G > 1:
        k4 = k4.repeat_interleave(G, dim=1)                   # (1, H_q, N, D)
        v4 = v4.repeat_interleave(G, dim=1)

    # Get the full attention output AND probs via math backend (flash doesn't
    # expose probs). We fall back to math backend only for prob extraction;
    # the dominant cost (QK+V matmul) is the same order as FlashAttn.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel([SDPBackend.MATH])
    except (ImportError, AttributeError):
        import contextlib
        ctx = contextlib.nullcontext()

    with ctx:
        # Compute scores manually so we can get probs for top-p
        scores = torch.matmul(q4, k4.transpose(-2, -1)) * scale   # (1,H_q,1,N)
        probs  = torch.softmax(scores, dim=-1).squeeze(0).squeeze(1)  # (H_q, N)

    # ── Step 2: top-p mask on GPU ──
    sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_p.cumsum(dim=-1)
    keep   = (cumsum - sorted_p) < top_p
    mask   = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, keep)

    # ── Step 3: renormalize + weighted V (bmm on full N, zeroed-out pruned) ──
    masked = probs * mask
    masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    masked = masked.to(dtype)

    masked_g = masked.reshape(h_kv, G, n)
    out_g = torch.einsum("hgn,hnd->hgd", masked_g.float(), values.float())
    return out_g.reshape(h_q, d_v).to(dtype)


# ── Oracle threshold (not timed) ─────────────────────────────────────────────

def _oracle_threshold(q, keys, q_head_to_kv, topk: int) -> torch.Tensor:
    """Top-k-th exact dot product per head. Excluded from timing."""
    keys_e = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_e.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


# ── Timing helpers ────────────────────────────────────────────────────────────

def _time_gpu(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def _time_repeated(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _fmt(value: float, fastest: float) -> str:
    s = f"{value:.3f}ms"
    return f"{_GREEN}{s}{_RESET}" if value == fastest else s


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GPU decoding latency: Louver vs dense_eager / dense_flash / twilight.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input source (pick one):\n"
            "  --input-qkv FILE   load saved CaptureState .pt\n"
            "  --model NAME       capture on-the-fly then benchmark "
            "(model is deleted before timing starts)\n"
        ),
    )
    # Input: either a saved capture or on-the-fly capture from a model
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-qkv", type=Path, default=None,
                   help="CaptureState .pt file.")
    g.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID for on-the-fly capture.")
    p.add_argument("--n-tokens", type=int, default=20000,
                   help="Tokens to generate when using --model.")
    p.add_argument("--problem-idx", type=int, default=0,
                   help="AIME problem index when using --model.")

    p.add_argument("--layer", type=int, default=None,
                   help="Which layer to benchmark. Default: middle of available layers.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Decode steps. Default: all available after prefill.")
    p.add_argument("--prefill-frac", type=float, default=None,
                   help="Fraction of keys used as prefill. Default: use capture's prompt_length.")
    p.add_argument("--topk", type=int, default=20,
                   help="Oracle top-k for threshold (excluded from timing).")
    p.add_argument("--top-p", type=float, default=0.85,
                   help="Twilight top-p threshold.")
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--n-growth", type=int, default=8192,
                   help="Arena growth increment (keeps reallocs infrequent).")
    p.add_argument("--parallel-update", action="store_true",
                   help="Overlap index update with next attention step.")
    p.add_argument("--update-stream-priority", type=int, default=-1)
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Default: reports/gpu_bench_<stem>.csv")
    p.add_argument("--flush-every", type=int, default=50)
    return p.parse_args()


def _load_capture(args) -> tuple["CaptureState", str]:
    """Return (capture, stem_for_csv)."""
    if args.input_qkv is not None:
        print(f"Loading {args.input_qkv} ...")
        return CaptureState.load(args.input_qkv), args.input_qkv.stem

    # On-the-fly: capture then free model before benchmarking
    from benchmark_area.experiments.latency.capture_aime import (  # type: ignore
        _load_aime_problem, _mid_layer, capture_with_layer_filter,
    )
    layer = args.layer if args.layer is not None else _mid_layer(args.model)
    problem = _load_aime_problem(args.problem_idx)
    print(f"Capturing {args.n_tokens} tokens from {args.model} (layer {layer}) ...")
    cap = capture_with_layer_filter(
        model_name=args.model,
        prompt_text=problem,
        n=args.n_tokens,
        target_layers=[layer],
    )
    # Free GPU memory used by the model before timing starts
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("Model freed from GPU. Starting benchmark.")
    slug = args.model.replace("/", "_").replace("-", "_")
    return cap, f"{slug}_layer{layer}_N{cap.generated_token_count()}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    cap, csv_stem = _load_capture(args)
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer is not None else layer_ids[len(layer_ids) // 2]
    if layer not in layer_ids:
        raise ValueError(f"Layer {layer} not in capture. Available: {layer_ids}")

    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError("Capture missing values. Re-capture with values enabled.")

    dtype = torch.float16
    keys    = keys_cpu.to(device="cuda", dtype=dtype)
    queries = queries_cpu.to(device="cuda", dtype=dtype)
    values  = values_cpu.to(device="cuda", dtype=dtype)

    H_q, H_kv = queries.shape[0], keys.shape[0]
    N_total, D = keys.shape[1], keys.shape[2]
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    if args.prefill_frac is not None:
        n_prefill = max(1, int(args.prefill_frac * N_total))
    elif cap.prompt_length is not None:
        n_prefill = max(1, int(cap.prompt_length))
    else:
        n_prefill = max(1, int(0.05 * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("Not enough keys for decoding. Adjust --prefill-frac.")

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill={n_prefill}  decode_steps={n_decode}  parallel_update={args.parallel_update}")

    # Build index on prefill keys/values
    prefill_keys   = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    cfg = TAIndexConfig(
        n_growth=args.n_growth,
        refine_iter=args.refine_iter,
        parallel_update=args.parallel_update,
        update_stream_priority=args.update_stream_priority,
    )
    index = TAIndex(cfg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    torch.cuda.synchronize()
    print(f"Build: {(time.perf_counter()-t0)*1000:.1f}ms")

    # Quick correctness check on step 0
    q0 = queries[:, n_prefill, :]
    q0n = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    th0 = _oracle_threshold(q0n, prefill_keys if q_head_to_kv is None else prefill_keys[q_head_to_kv], q_head_to_kv, args.topk)
    out_l = index.attend(q0n, th0, q_head_to_kv=q_head_to_kv)
    out_r = baseline_attention(q0n, prefill_keys, prefill_values, q_head_to_kv)
    diff = (out_l.float() - out_r.float()).abs().max().item()
    rel  = diff / (out_r.float().abs().max().item() + 1e-9)
    print(f"Correctness (louver vs dense): max_abs={diff:.3e}  rel={rel:.3e}")

    # Output CSV
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"gpu_bench_{csv_stem}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys",
        "louver_ms", "dense_eager_ms", "dense_flash_ms", "twilight_ms",
        "update_ms", "amortized_louver_ms",
        "memory_bytes", "k_used", "k_cap",
        "step_wall_ms", "update_kernel_ms", "update_wait_ms",
        "update_inflight", "buffer_len",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []
    update_kernel_costs: list[float] = []
    update_wait_costs: list[float] = []
    update_fire_steps: list[int] = []
    last_update_ms = 0.0

    sum_louver = sum_eager = sum_flash = sum_twilight = 0.0
    parallel = args.parallel_update
    sim_start = time.perf_counter()

    bar = tqdm(range(n_decode), unit="step", dynamic_ncols=True,
               desc="gpu_bench", leave=True)
    for step in bar:
        token_idx = n_prefill + step
        q  = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        update_inflight_at_start = bool(parallel and index.has_pending_update)

        # Lazy publish
        publish_metrics = None
        if parallel:
            pre = len(index.update_metrics_log)
            if index.try_publish() and len(index.update_metrics_log) > pre:
                publish_metrics = index.update_metrics_log[-1]

        step_start = torch.cuda.Event(enable_timing=True)
        step_end   = torch.cuda.Event(enable_timing=True)
        step_start.record()

        # Oracle threshold (NOT timed)
        all_keys = keys[:, :token_idx + 1, :]
        all_vals = values[:, :token_idx + 1, :]
        keys_e   = all_keys if q_head_to_kv is None else all_keys.index_select(0, q_head_to_kv)
        th = _oracle_threshold(qn, keys_e, q_head_to_kv, args.topk)

        # ── Timed: Louver ──
        louver_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )

        # ── Timed: dense eager ──
        eager_ms = _time_repeated(
            lambda: baseline_attention(qn, all_keys, all_vals, q_head_to_kv)
        )

        # ── Timed: dense flash ──
        flash_ms = _time_repeated(
            lambda: _sdpa_flash(qn, all_keys, all_vals, q_head_to_kv)
        )

        # ── Timed: Twilight ──
        twilight_ms = _time_repeated(
            lambda: _twilight_attention(qn, all_keys, all_vals, args.top_p)
        )

        # Append new (k, v) to buffer
        index.append_decoding_kv(
            keys[:, token_idx:token_idx + 1, :],
            values[:, token_idx:token_idx + 1, :],
        )

        # Update bookkeeping
        update_ms = 0.0
        upd_kernel_col = 0.0
        upd_wait_col   = 0.0

        if publish_metrics is not None:
            upd_kernel_col = publish_metrics.kernel_ms
            upd_wait_col   = publish_metrics.host_wait_ms
            update_kernel_costs.append(publish_metrics.kernel_ms)
            update_wait_costs.append(publish_metrics.host_wait_ms)
            last_update_ms = publish_metrics.kernel_ms

        if index.needs_update():
            if parallel:
                pre = len(index.update_metrics_log)
                index.update_async(fire_step=step)
                update_fire_steps.append(step)
                if len(index.update_metrics_log) > pre:
                    m = index.update_metrics_log[-1]
                    if publish_metrics is None:
                        upd_kernel_col = m.kernel_ms
                        upd_wait_col   = m.host_wait_ms
                        update_kernel_costs.append(m.kernel_ms)
                        update_wait_costs.append(m.host_wait_ms)
                    last_update_ms = m.kernel_ms
                update_ms = upd_kernel_col
            else:
                update_ms = _time_gpu(index.update)
                last_update_ms = update_ms
                upd_kernel_col = update_ms
                update_kernel_costs.append(update_ms)
                update_fire_steps.append(step)

        update_costs.append(update_ms)
        amort_upd = sum(update_costs) / len(update_costs)

        step_end.record()
        step_end.synchronize()
        step_wall_ms = step_start.elapsed_time(step_end)

        sum_louver   += louver_ms
        sum_eager    += eager_ms
        sum_flash    += flash_ms
        sum_twilight += twilight_ms

        rows.append({
            "step":              step,
            "n_keys":            int(all_keys.shape[1]),
            "louver_ms":         round(louver_ms,   4),
            "dense_eager_ms":    round(eager_ms,    4),
            "dense_flash_ms":    round(flash_ms,    4),
            "twilight_ms":       round(twilight_ms, 4),
            "update_ms":         round(update_ms,   4),
            "amortized_louver_ms": round(louver_ms + amort_upd, 4),
            "memory_bytes":      index.memory_bytes(),
            "k_used":            int(index.state["K_used"]),
            "k_cap":             int(index.state["K_cap"]),
            "step_wall_ms":      round(step_wall_ms, 4),
            "update_kernel_ms":  round(upd_kernel_col, 4),
            "update_wait_ms":    round(upd_wait_col,   4),
            "update_inflight":   int(update_inflight_at_start),
            "buffer_len":        index.n_buffered,
        })

        bar.set_postfix(
            louver=f"{louver_ms:.3f}ms",
            eager=f"{eager_ms:.3f}ms",
            flash=f"{flash_ms:.3f}ms",
            twilight=f"{twilight_ms:.3f}ms",
            upd=f"{last_update_ms:.1f}ms",
            K=f"{index.state['K_used']}/{index.state['K_cap']}",
            buf=index.n_buffered,
        )

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()

    # Drain in-flight update
    if parallel and index.has_pending_update:
        index.wait_for_update()
        if len(index.update_metrics_log) > len(update_kernel_costs):
            m = index.update_metrics_log[-1]
            update_kernel_costs.append(m.kernel_ms)
            update_wait_costs.append(m.host_wait_ms)

    # Summary
    n_steps = max(n_decode, 1)
    n_upd   = len(update_fire_steps)
    total_upd_ms = sum(update_kernel_costs)
    amort_upd_ms = total_upd_ms / n_steps

    print(f"\n──── Summary ────")
    print(f"Steps: {n_decode}  Updates fired: {n_upd}  "
          f"(interval={BUFFER_SIZE}, parallel={parallel})")
    print(f"  avg louver      : {sum_louver/n_steps:.4f} ms/step")
    print(f"  avg dense_eager : {sum_eager/n_steps:.4f} ms/step")
    print(f"  avg dense_flash : {sum_flash/n_steps:.4f} ms/step")
    print(f"  avg twilight    : {sum_twilight/n_steps:.4f} ms/step")
    print(f"  update kernel   : total={total_upd_ms:.1f}ms  "
          f"amortized={amort_upd_ms:.4f}ms/step")
    print(f"  amort louver    : {sum_louver/n_steps + amort_upd_ms:.4f} ms/step")
    if parallel and update_wait_costs:
        total_wait = sum(update_wait_costs)
        print(f"  overlap misses  : {index.n_overlap_misses}/{n_upd}  "
              f"host_stall_total={total_wait:.1f}ms")
    print(f"Done. CSV → {out_csv}")


if __name__ == "__main__":
    main()
