"""
CPU decoding latency benchmark: Louver vs dense baselines.

Per decode step (timed on CPU):
    louver      — Louver CPU TA-filter + sparse attention kernel
    dense_eager — manual Q @ K.T + softmax + @ V  (fp32 einsum)
    torch_sdpa  — torch.nn.functional.scaled_dot_product_attention (math backend)

Twilight is not included: it computes full O(N) attention on the GPU side
and the original paper makes no CPU claims for it.

Captures: benchmark_area/experiments/latency/captures/*.pt
          or benchmark_area/quick_pruning/capture_qkv_*.pt (smoke test)

Smoke-test command:
    python cpu_bench.py \\
        --input-qkv ../../quick_pruning/capture_qkv_12000_Qwen_Qwen2.5-7B-Instruct.pt \\
        --n-steps 200
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_index import (
    BUFFER_SIZE,
    TAIndexCPU,
    TAIndexCPUConfig,
    baseline_dense,
    baseline_sdpa,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _q_to_kv_map,
)

_GREEN = "\033[32m"
_RESET = "\033[0m"


# ── Oracle threshold (not timed) ─────────────────────────────────────────────

def _oracle_threshold(q, keys, q_head_to_kv, topk: int) -> torch.Tensor:
    keys_e = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_e.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


# ── Timing helper ─────────────────────────────────────────────────────────────

def _time_repeated(fn, iters: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def _fmt(value: float, fastest: float) -> str:
    s = f"{value:.3f}ms"
    return f"{_GREEN}{s}{_RESET}" if value == fastest else s


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CPU decoding latency: Louver vs dense_eager / torch_sdpa.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-qkv", type=Path, default=None,
                   help="CaptureState .pt file.")
    g.add_argument("--model", type=str, default=None,
                   help="HuggingFace model ID for on-the-fly capture (requires CUDA for capture).")
    p.add_argument("--n-tokens", type=int, default=20000,
                   help="Tokens to generate when using --model.")
    p.add_argument("--problem-idx", type=int, default=0)
    p.add_argument("--layer", type=int, default=None,
                   help="Which layer. Default: middle of available layers.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Decode steps. Default: all available after prefill.")
    p.add_argument("--prefill-frac", type=float, default=None,
                   help="Fraction of keys as prefill. Default: use capture's prompt_length.")
    p.add_argument("--topk", type=int, default=20,
                   help="Oracle top-k for threshold (not timed).")
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--n-growth", type=int, default=8192)
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Default: reports/cpu_bench_<stem>.csv")
    p.add_argument("--flush-every", type=int, default=50)
    p.add_argument("--num-threads", type=int, default=None,
                   help="PyTorch intra-op threads. Default: all available cores.")
    return p.parse_args()


def _load_capture(args) -> tuple["CaptureState", str]:
    if args.input_qkv is not None:
        print(f"Loading {args.input_qkv} ...")
        return CaptureState.load(args.input_qkv), args.input_qkv.stem
    from benchmark_area.experiments.latency.capture_aime import (  # type: ignore
        _load_aime_problem, _mid_layer, capture_with_layer_filter,
    )
    layer = args.layer if args.layer is not None else _mid_layer(args.model)
    problem = _load_aime_problem(args.problem_idx)
    print(f"Capturing {args.n_tokens} tokens from {args.model} (layer {layer}) ...")
    cap = capture_with_layer_filter(
        model_name=args.model, prompt_text=problem,
        n=args.n_tokens, target_layers=[layer],
    )
    import torch, gc
    torch.cuda.empty_cache(); gc.collect()
    print("Model freed. Starting benchmark.")
    slug = args.model.replace("/", "_").replace("-", "_")
    return cap, f"{slug}_layer{layer}_N{cap.generated_token_count()}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Thread control: set before any torch op
    n_threads = args.num_threads if args.num_threads is not None else torch.get_num_threads()
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))

    cap, csv_stem = _load_capture(args)
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer is not None else layer_ids[len(layer_ids) // 2]
    if layer not in layer_ids:
        raise ValueError(f"Layer {layer} not found. Available: {layer_ids}")

    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError("Capture missing values.")

    # CPU bench uses fp32
    dtype = torch.float32
    keys    = keys_cpu.to(dtype=dtype)
    queries = queries_cpu.to(dtype=dtype)
    values  = values_cpu.to(dtype=dtype)

    H_q, H_kv = queries.shape[0], keys.shape[0]
    N_total, D = keys.shape[1], keys.shape[2]
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cpu") if H_q != H_kv else None

    if args.prefill_frac is not None:
        n_prefill = max(1, int(args.prefill_frac * N_total))
    elif cap.prompt_length is not None:
        n_prefill = max(1, int(cap.prompt_length))
    else:
        n_prefill = max(1, int(0.05 * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("Not enough keys. Adjust --prefill-frac.")

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill={n_prefill}  decode_steps={n_decode}  threads={n_threads}")

    # Build CPU index
    prefill_keys   = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    cfg = TAIndexCPUConfig(n_growth=args.n_growth, refine_iter=args.refine_iter)
    index = TAIndexCPU(cfg)
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    print(f"Build: {(time.perf_counter()-t0)*1000:.1f}ms")

    # Output CSV
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"cpu_bench_{csv_stem}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys",
        "louver_ms", "dense_eager_ms", "torch_sdpa_ms",
        "update_ms", "amortized_louver_ms",
        "memory_bytes", "k_used", "k_cap", "buffer_len",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []
    update_kernel_costs: list[float] = []
    update_fire_steps: list[int] = []
    last_update_ms = 0.0

    sum_louver = sum_eager = sum_sdpa = 0.0
    sim_start = time.perf_counter()

    bar = tqdm(range(n_decode), unit="step", dynamic_ncols=True,
               desc="cpu_bench", leave=True)
    for step in bar:
        token_idx = n_prefill + step
        q  = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # Oracle threshold (NOT timed)
        all_keys = keys[:, :token_idx + 1, :]
        all_vals = values[:, :token_idx + 1, :]
        keys_e   = all_keys if q_head_to_kv is None else all_keys.index_select(0, q_head_to_kv)
        th = _oracle_threshold(qn, keys_e, q_head_to_kv, args.topk)

        # ── Timed: Louver CPU ──
        louver_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv)
        )

        # ── Timed: dense eager ──
        eager_ms = _time_repeated(
            lambda: baseline_dense(qn, all_keys, all_vals, q_head_to_kv)
        )

        # ── Timed: torch SDPA (CPU math backend) ──
        sdpa_ms = _time_repeated(
            lambda: baseline_sdpa(qn, all_keys, all_vals, q_head_to_kv)
        )

        # Append to buffer
        index.append_decoding_kv(
            keys[:, token_idx:token_idx + 1, :],
            values[:, token_idx:token_idx + 1, :],
        )

        # Sync update
        update_ms = 0.0
        if index.needs_update():
            t0 = time.perf_counter()
            index.update()
            update_ms = (time.perf_counter() - t0) * 1000
            last_update_ms = update_ms
            update_kernel_costs.append(update_ms)
            update_fire_steps.append(step)

        update_costs.append(update_ms)
        amort_upd = sum(update_costs) / len(update_costs)

        sum_louver += louver_ms
        sum_eager  += eager_ms
        sum_sdpa   += sdpa_ms

        rows.append({
            "step":              step,
            "n_keys":            int(all_keys.shape[1]),
            "louver_ms":         round(louver_ms, 4),
            "dense_eager_ms":    round(eager_ms,  4),
            "torch_sdpa_ms":     round(sdpa_ms,   4),
            "update_ms":         round(update_ms, 4),
            "amortized_louver_ms": round(louver_ms + amort_upd, 4),
            "memory_bytes":      0,
            "k_used":            int(index.state["K_used"]),
            "k_cap":             int(index.state["K_cap"]),
            "buffer_len":        index.n_buffered,
        })

        bar.set_postfix(
            louver=f"{louver_ms:.3f}ms",
            eager=f"{eager_ms:.3f}ms",
            sdpa=f"{sdpa_ms:.3f}ms",
            upd=f"{last_update_ms:.1f}ms",
            K=f"{index.state['K_used']}/{index.state['K_cap']}",
            buf=index.n_buffered,
        )

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()

    # Summary
    n_steps    = max(n_decode, 1)
    total_upd  = sum(update_kernel_costs)
    amort_upd  = total_upd / n_steps

    print(f"\n──── Summary ────")
    print(f"Steps: {n_decode}  Updates fired: {len(update_fire_steps)}  "
          f"(interval={BUFFER_SIZE})")
    print(f"  avg louver     : {sum_louver/n_steps:.4f} ms/step")
    print(f"  avg dense_eager: {sum_eager/n_steps:.4f} ms/step")
    print(f"  avg torch_sdpa : {sum_sdpa/n_steps:.4f} ms/step")
    print(f"  update kernel  : total={total_upd:.1f}ms  "
          f"amortized={amort_upd:.4f}ms/step")
    print(f"  amort louver   : {sum_louver/n_steps + amort_upd:.4f} ms/step")
    print(f"Done. CSV → {out_csv}")


if __name__ == "__main__":
    main()
