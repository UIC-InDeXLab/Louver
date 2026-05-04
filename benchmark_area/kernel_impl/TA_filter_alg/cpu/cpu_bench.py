"""End-to-end CPU TA-filter decoding benchmark.

For each step:
    1. (NOT timed) per-head scalar threshold = topk-th largest exact dot.
    2. (timed) ``index.attend`` — fused filter + sparse-attn.
    3. (timed) dense baseline (einsum).
    4. (timed) torch SDPA baseline.
    5. append (k, v) to the buffer.
    6. every BUFFER_SIZE=256 steps: ``index.update`` (sync).

Hardcoded: bf=4, S=4, BUFFER=256.

Usage:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_bench \\
        --input-qkv benchmark_area/quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt \\
        --n-steps 200
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

_HIRA = Path(__file__).resolve().parents[4]
for _p in (_HIRA.parent, _HIRA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_index import (
    BUFFER_SIZE, TAIndexCPU, TAIndexCPUConfig,
    baseline_dense, baseline_sdpa,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState, _capture_qkv, _q_to_kv_map,
)


_GREEN = "\033[32m"
_RESET = "\033[0m"
DEFAULT_PROMPT = "Benchmark the TA-filter index over a long decoding trace."


def topk_threshold(q, keys, q_head_to_kv, topk):
    keys_eval = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def _time_repeated(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def _format(value, fastest):
    text = f"{value:.3f}ms"
    return f"{_GREEN}{text}{_RESET}" if value == fastest else text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=4000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--prefill-frac", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=None,
                   help="Decoding steps; default = remaining capture after prefill.")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-growth", type=int, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--flush-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


def load_qkv(args):
    if args.input_qkv is not None:
        print(f"loading capture {args.input_qkv}")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"capturing {args.n_tokens} tokens from {args.model}")
        cap = _capture_qkv(
            model_name=args.model, prompt_text=DEFAULT_PROMPT,
            n=args.n_tokens, device="cpu",
            torch_dtype=torch.float32, show_progress=True,
        )
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    q, k, v = cap.to_layer_tensors(layer)
    if v is None:
        raise RuntimeError("TA-filter requires captured values.")
    return q.float().cpu(), k.float().cpu(), v.float().cpu(), layer


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    queries, keys, values, layer = load_qkv(args)
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cpu") if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("not enough keys for decoding — adjust --prefill-frac")
    n_growth = args.n_growth if args.n_growth is not None else n_decode + BUFFER_SIZE

    print(f"layer {layer}: H_q={H_q} H_kv={H_kv} D={D}")
    print(f"prefill_keys={n_prefill}  decoding_steps={n_decode}  n_growth={n_growth}")

    cfg = TAIndexCPUConfig(n_growth=n_growth, refine_iter=2)
    index = TAIndexCPU(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"build: {build_ms:.1f}ms  K_cap={index.state['K_cap']}")

    # Correctness on first decode step
    q0 = queries[:, n_prefill, :]
    qn0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys_check = keys[:, :n_prefill, :]
    values_check = values[:, :n_prefill, :]
    th0 = topk_threshold(qn0, keys_check, q_head_to_kv, args.topk).to(torch.float32)
    out_ours = index.attend(qn0, th0, q_head_to_kv=q_head_to_kv)
    out_ref = baseline_dense(qn0, keys_check, values_check, q_head_to_kv=q_head_to_kv)
    diff = (out_ours.float() - out_ref.float()).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-9)
    print(f"correctness vs dense: max_abs_diff={diff:.4e}  rel={rel:.4e}")

    out_csv = args.output_csv or (
        Path(__file__).parent.parent.parent / "reports" / "bench_TA_filter_cpu.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys", "attend_ours_ms", "dense_attn_ms", "sdpa_ms",
        "update_ms", "buffer_len", "n_used",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    sum_attend = 0.; sum_dense = 0.; sum_sdpa = 0.; sum_upd = 0.
    n_upd = 0
    sim_start = time.perf_counter()

    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :]
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        all_keys = keys[:, :token_idx + 1, :]
        all_values = values[:, :token_idx + 1, :]
        th = topk_threshold(qn, all_keys, q_head_to_kv, args.topk).to(torch.float32)

        attend_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv),
            iters=args.iters, warmup=args.warmup,
        )
        dense_ms = _time_repeated(
            lambda: baseline_dense(qn, all_keys, all_values, q_head_to_kv),
            iters=args.iters, warmup=args.warmup,
        )
        sdpa_ms = _time_repeated(
            lambda: baseline_sdpa(qn, all_keys, all_values, q_head_to_kv),
            iters=args.iters, warmup=args.warmup,
        )

        new_k = keys[:, token_idx:token_idx + 1, :]
        new_v = values[:, token_idx:token_idx + 1, :]
        index.append_decoding_kv(new_k, new_v)

        update_ms = 0.
        if index.needs_update():
            t0 = time.perf_counter()
            index.update()
            update_ms = (time.perf_counter() - t0) * 1000
            sum_upd += update_ms; n_upd += 1

        sum_attend += attend_ms
        sum_dense += dense_ms
        sum_sdpa += sdpa_ms

        rows.append({
            "step": step,
            "n_keys": int(all_keys.shape[1]),
            "attend_ours_ms": round(attend_ms, 4),
            "dense_attn_ms": round(dense_ms, 4),
            "sdpa_ms": round(sdpa_ms, 4),
            "update_ms": round(update_ms, 4),
            "buffer_len": index.n_buffered,
            "n_used": index.n_indexed,
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
                f"dense={_format(dense_ms, fastest)}  "
                f"sdpa={_format(sdpa_ms, fastest)}  "
                f"N={index.n_indexed}  buf={index.n_buffered}  "
                f"[{elapsed:.1f}s]"
            )

    mean_a = sum_attend / max(n_decode, 1)
    mean_d = sum_dense / max(n_decode, 1)
    mean_s = sum_sdpa / max(n_decode, 1)
    print("\n──── Summary ────")
    print(f"mean attend = {mean_a:.4f}ms/step")
    print(f"mean dense  = {mean_d:.4f}ms/step  speedup vs ours = {mean_d/mean_a:.2f}x")
    print(f"mean sdpa   = {mean_s:.4f}ms/step  speedup vs ours = {mean_s/mean_a:.2f}x")
    print(f"updates: {n_upd}  total_ms={sum_upd:.1f}  amortized={sum_upd/max(n_decode,1):.4f}ms/step")
    print(f"csv -> {out_csv}")


if __name__ == "__main__":
    main()
