"""End-to-end CPU benchmark: decoding simulation with the CPU subspace k-center
index (build_v1.0 + update_v1.0 + attention_v4.3).

Mirrors `kernel_impl/bench.py` (the GPU end-to-end bench) but:
  - all timing on CPU via time.perf_counter (no CUDA events / streams)
  - update is synchronous (no parallel side stream)
  - baselines are CPU dense + CPU SDPA (oneDNN)

For each decoding step i:
  1. (NOT timed) Compute per-subspace thresholds from the true top-k set
     over all keys up to step i (full-space dense scan).
  2. (timed) index.attend(q, th) — fused CPU attention over index + buffer.
  3. (timed) baseline_attention dense.
  4. (timed) baseline_sdpa.
  5. Append (k, v) to the decoding buffer.
  6. Every `--update-interval` steps: index.update() — timed separately.

CPU baselines (all timed per step):
  - dense_fp32 : einsum + softmax + einsum, GQA-aware (manual reference)
  - sdpa_fp32  : torch.nn.functional.scaled_dot_product_attention (PRIMARY)
  - sdpa_bf16  : SDPA with bf16 K/V/Q (kept for completeness)

Reports incremental CSV at kernel_impl/reports/cpu_bench_<update_mode>.csv:
  step, n_keys, attend_ours_ms, dense_fp32_ms, sdpa_fp32_ms, sdpa_bf16_ms,
  update_ms, amortized_ours_ms, memory_bytes

Usage:
    python -m hira.benchmark_area.kernel_impl.cpu_bench \
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

from hira.benchmark_area.kernel_impl.cpu_index import (
    IndexConfigCPU,
    SubspaceKCenterIndexCPU,
    baseline_attention,
    baseline_sdpa,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
)

DEFAULT_PROMPT = "Benchmark the index over a long decoding trace."
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _q_to_kv_map_cpu(num_q_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"H_q={num_q_heads} must be divisible by H_kv={num_kv_heads}"
        )
    return torch.arange(num_q_heads, dtype=torch.int64) // (num_q_heads // num_kv_heads)


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace thresholds derived from full-space top-k set."""
    scores = torch.einsum("hd,hnd->hn", q, keys)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    ths = []
    for s, e in dim_slices:
        ss = torch.einsum("hd,hnd->hn", q[:, s:e], keys[:, :, s:e])
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0)


def _time_cpu(fn) -> tuple[object, float]:
    """Single-shot CPU timing in ms."""
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def _time_repeated(fn, iters: int = 5, warmup: int = 1) -> float:
    """Avg ms/call across `iters` runs (after `warmup`).

    Read-only ops only — do not pass anything that mutates index state.
    """
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000.0


def _format_step_ms(value: float, fastest: float) -> str:
    text = f"{value:.3f}ms"
    if value == fastest:
        return f"{_GREEN}{text}{_RESET}"
    return text


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end CPU decoding simulation for the HIRA index."
    )

    # ── Input source ──
    p.add_argument("--input-qkv", type=Path, default=None,
                   help="Path to a captured QKV .pt file. If omitted, captures "
                        "fresh from --model on CUDA (then moved to CPU).")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF model id used when capturing QKV live.")
    p.add_argument("--n-tokens", type=int, default=2000,
                   help="Tokens to capture when --input-qkv is not given.")
    p.add_argument("--layer", type=int, default=15,
                   help="Which transformer layer's Q/K/V to simulate.")

    # ── Simulation window ──
    p.add_argument("--prefill-frac", type=float, default=0.5,
                   help="Fraction of total captured keys treated as prefill "
                        "(index is built on these). The rest is the decoding trace.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Max decoding steps to simulate. Defaults to remaining "
                        "tokens after prefill (capped by available queries).")

    # ── Threshold ──
    p.add_argument("--topk", type=int, default=20,
                   help="k for top-k-derived per-subspace thresholds (excluded from timing).")

    # ── Index config ──
    p.add_argument("--bf", type=int, default=4,
                   help="Branching factor: cluster size target (K = ceil(N/bf)).")
    p.add_argument("--n-subspaces", type=int, default=8,
                   help="Number of contiguous dim splits.")
    p.add_argument("--refine-iter", type=int, default=5,
                   help="Lloyd refinement iterations per subspace during build.")
    p.add_argument("--update-mode", choices=["full", "inc"], default="inc",
                   help='"full": rebuild on all keys (slow). "inc": mini-index '
                        'the buffer and merge.')
    p.add_argument("--update-refine-iter", type=int, default=0,
                   help="Refine iterations for the incremental update.")
    p.add_argument("--update-interval", type=int, default=256,
                   help="Flush the decoding buffer into the index every N steps.")

    # ── Timing knobs ──
    p.add_argument("--attend-iters", type=int, default=10,
                   help="Per-step repeat count for attend/baseline timing "
                        "(amortizes Python overhead). Default 10 matches "
                        "kernel_bench/bench_attention.py.")
    p.add_argument("--attend-warmup", type=int, default=2,
                   help="Warmup calls per step before timed iters (default 2 "
                        "matches kernel_bench/bench_attention.py).")
    p.add_argument("--threads", type=int, default=None,
                   help="torch.set_num_threads override (default: leave at PyTorch default)")

    # ── Output ──
    p.add_argument("--output-csv", type=Path, default=None,
                   help="Defaults to kernel_impl/reports/cpu_bench_<update_mode>.csv.")
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cap = _capture_qkv(
            model_name=args.model, prompt_text=DEFAULT_PROMPT,
            n=args.n_tokens, device=device,
            torch_dtype=torch.float16, show_progress=True,
        )
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError(
            "Need captured values. Re-capture with values."
        )
    return queries_cpu, keys_cpu, values_cpu, layer


def main():
    args = parse_args()
    if args.update_interval < 1:
        raise ValueError("--update-interval must be >= 1")
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    torch.manual_seed(args.seed)
    n_threads = torch.get_num_threads()

    queries_cpu, keys_cpu, values_cpu, layer = load_qkv(args)
    # Stay in fp32 on CPU: v4.3, SDPA, and dense baselines compare cleanly.
    # Captures are fp16 → upcast.
    dtype = torch.float32
    keys = keys_cpu.to(device="cpu", dtype=dtype).contiguous()
    queries = queries_cpu.to(device="cpu", dtype=dtype).contiguous()
    values = values_cpu.to(device="cpu", dtype=dtype).contiguous()
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map_cpu(H_q, H_kv) if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    max_decode = min(N_total - n_prefill, queries.shape[1] - n_prefill)
    n_decode = max_decode if args.n_steps is None else min(args.n_steps, max_decode)
    if n_decode <= 0:
        raise ValueError("Not enough keys for decoding — adjust --prefill-frac.")

    print(f"Layer {layer}: H_q={H_q} H_kv={H_kv} D={D}  threads={n_threads}")
    print(f"prefill_keys={n_prefill}  decoding_steps={n_decode}  "
          f"update_interval={args.update_interval}  mode={args.update_mode}")

    # ── Build index on prefill keys/values ──
    cfg = IndexConfigCPU(
        n_subspaces=args.n_subspaces, bf=args.bf, refine_iter=args.refine_iter,
        update_mode=args.update_mode,
        update_refine_iter=args.update_refine_iter,
    )
    index = SubspaceKCenterIndexCPU(cfg)
    prefill_keys = keys[:, :n_prefill, :].contiguous()
    prefill_values = values[:, :n_prefill, :].contiguous()
    t0 = time.perf_counter()
    index.build(prefill_keys, prefill_values)
    build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Build: {build_ms:.1f} ms  (build=v1.0, update=v1.0, attn=v4.3)")

    # ── Correctness check on first step ──
    q0 = queries[:, n_prefill, :].contiguous()
    qn0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    keys0 = keys[:, :n_prefill, :]
    values0 = values[:, :n_prefill, :]
    keys_eval0 = keys0 if q_head_to_kv is None else keys0[q_head_to_kv]
    th0 = subspace_topk_thresholds(qn0, keys_eval0, args.topk, index.state["dim_slices"])
    out_ours = index.attend(qn0, th0, q_head_to_kv=q_head_to_kv).float()
    out_ref = baseline_attention(qn0, keys0, values0, q_head_to_kv=q_head_to_kv)
    diff = (out_ours - out_ref).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-9)
    print(f"Correctness (attend vs dense): max_abs_diff={diff:.4e}  rel={rel:.4e}")

    # ── CSV setup ──
    out_csv = args.output_csv or (
        Path(__file__).parent / "reports" / f"cpu_bench_{args.update_mode}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step", "n_keys", "k_used", "k_cap",
        "attend_ours_ms",
        "dense_fp32_ms", "sdpa_fp32_ms", "sdpa_bf16_ms",
        "update_ms", "amortized_ours_ms", "memory_bytes",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows: list[dict] = []
    update_costs: list[float] = []
    last_update_ms = 0.0
    sum_attend_ms = 0.0
    sum_dense_ms = 0.0
    sum_sdpa_fp32_ms = 0.0
    sum_sdpa_bf16_ms = 0.0
    sim_start = time.perf_counter()

    for step in range(n_decode):
        token_idx = n_prefill + step
        q = queries[:, token_idx, :].contiguous()
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        # NOT timed: views over fp32 K/V history. Force contiguity so SDPA
        # doesn't pay a hidden internal copy of a non-contig view (the
        # baselines in kernel_bench/bench_attention.py use contiguous K/V).
        all_keys_so_far = keys[:, : token_idx + 1, :].contiguous()
        all_values_so_far = values[:, : token_idx + 1, :].contiguous()
        keys_eval = all_keys_so_far if q_head_to_kv is None else all_keys_so_far[q_head_to_kv]
        th = subspace_topk_thresholds(qn, keys_eval, args.topk, index.state["dim_slices"])

        # NOT timed: pre-convert only the bf16 baseline inputs.
        qn_bf16 = qn.to(torch.bfloat16).contiguous()
        keys_bf16 = all_keys_so_far.to(torch.bfloat16).contiguous()
        values_bf16 = all_values_so_far.to(torch.bfloat16).contiguous()

        # ── Timed: our fused full-AND attention (v4.3) ──
        attend_ms = _time_repeated(
            lambda: index.attend(qn, th, q_head_to_kv=q_head_to_kv),
            iters=args.attend_iters, warmup=args.attend_warmup,
        )
        # ── Timed: dense fp32 baseline (manual reference) ──
        dense_ms = _time_repeated(
            lambda: baseline_attention(qn, all_keys_so_far, all_values_so_far, q_head_to_kv),
            iters=args.attend_iters, warmup=args.attend_warmup,
        )
        # ── Timed: SDPA fp32 baseline (PRIMARY) ──
        sdpa_fp32_ms = _time_repeated(
            lambda: baseline_sdpa(qn, all_keys_so_far, all_values_so_far, q_head_to_kv),
            iters=args.attend_iters, warmup=args.attend_warmup,
        )
        # ── Timed: SDPA bf16 baseline ──
        sdpa_bf16_ms = _time_repeated(
            lambda: baseline_sdpa(qn_bf16, keys_bf16, values_bf16, q_head_to_kv),
            iters=args.attend_iters, warmup=args.attend_warmup,
        )

        k_used = int(index.state.get("K_used", index.state["K"]))
        k_cap = int(index.state.get("K_cap", index.state["K"]))

        # ── Append (k, v) to buffer ──
        new_k = keys[:, token_idx : token_idx + 1, :]
        new_v = values[:, token_idx : token_idx + 1, :]
        index.append_decoding_kv(new_k, new_v)

        # ── Sync update every update_interval steps ──
        update_ms = 0.0
        if index.needs_update(args.update_interval):
            _, update_ms = _time_cpu(index.update)
            last_update_ms = update_ms
        update_costs.append(update_ms)

        sum_attend_ms += attend_ms
        sum_dense_ms += dense_ms
        sum_sdpa_fp32_ms += sdpa_fp32_ms
        sum_sdpa_bf16_ms += sdpa_bf16_ms
        amort_update_ms = sum(update_costs) / len(update_costs)
        amort_ours = attend_ms + amort_update_ms

        rows.append({
            "step": step,
            "n_keys": int(all_keys_so_far.shape[1]),
            "k_used": k_used,
            "k_cap": k_cap,
            "attend_ours_ms": round(attend_ms, 4),
            "dense_fp32_ms": round(dense_ms, 4),
            "sdpa_fp32_ms": round(sdpa_fp32_ms, 4),
            "sdpa_bf16_ms": round(sdpa_bf16_ms, 4),
            "update_ms": round(update_ms, 4),
            "amortized_ours_ms": round(amort_ours, 4),
            "memory_bytes": index.memory_bytes(),
        })

        if (step + 1) % args.flush_every == 0 or step == n_decode - 1:
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
            rows.clear()
            elapsed = time.perf_counter() - sim_start
            fastest_step_ms = min(attend_ms, dense_ms, sdpa_fp32_ms, sdpa_bf16_ms)
            print(
                f"step {step+1}/{n_decode}  "
                f"attend={_format_step_ms(attend_ms, fastest_step_ms)}  "
                f"dense={_format_step_ms(dense_ms, fastest_step_ms)}  "
                f"sdpa_fp32={_format_step_ms(sdpa_fp32_ms, fastest_step_ms)}  "
                f"sdpa_bf16={_format_step_ms(sdpa_bf16_ms, fastest_step_ms)}  "
                f"last_upd={last_update_ms:.2f}ms  K={k_used}/{k_cap}  [{elapsed:.1f}s]"
            )

    # ── Summary ──
    n_upd = sum(1 for x in update_costs if x > 0.0)
    total_update_ms = sum(update_costs)
    mean_attend = sum_attend_ms / max(n_decode, 1)
    mean_dense = sum_dense_ms / max(n_decode, 1)
    mean_sdpa_fp32 = sum_sdpa_fp32_ms / max(n_decode, 1)
    mean_sdpa_bf16 = sum_sdpa_bf16_ms / max(n_decode, 1)
    amort_update = total_update_ms / max(n_decode, 1)
    serial_cost = mean_attend + amort_update

    print(
        f"\n──── Summary ────"
        f"\nUpdates: {n_upd} fired over {n_decode} steps "
        f"(interval={args.update_interval}, mode={args.update_mode})"
    )
    print(f"  avg ours (attend):    {mean_attend:.4f} ms/step")
    print(f"  avg dense_fp32:       {mean_dense:.4f} ms/step")
    print(f"  avg sdpa_fp32:        {mean_sdpa_fp32:.4f} ms/step  (PRIMARY)")
    print(f"  avg sdpa_bf16:        {mean_sdpa_bf16:.4f} ms/step")
    print(
        f"  update kernel time:   total={total_update_ms:.1f}ms  "
        f"mean={total_update_ms / max(n_upd, 1):.2f}ms/update  "
        f"amortized={amort_update:.4f}ms/step"
    )
    print(
        f"  serial cost (ours):   attend({mean_attend*1000:.1f}us) + "
        f"update_amortized({amort_update*1000:.1f}us) = "
        f"{serial_cost*1000:.1f}us/step"
    )

    baselines = {
        "dense_fp32": mean_dense,
        "sdpa_fp32": mean_sdpa_fp32,
        "sdpa_bf16": mean_sdpa_bf16,
    }
    fastest_label, fastest_ms = min(baselines.items(), key=lambda kv: kv[1])
    primary_ms = baselines["sdpa_fp32"]
    print(f"  fastest baseline:     {fastest_label} = {fastest_ms:.4f} ms/step")
    if serial_cost > 0:
        print(
            f"  speedup vs sdpa_fp32: {primary_ms / serial_cost:.2f}x"
            f"  (sdpa_fp32={primary_ms:.4f}ms vs ours={serial_cost:.4f}ms)"
        )
        print(
            f"  speedup vs fastest:   {fastest_ms / serial_cost:.2f}x"
            f"  ({fastest_label}={fastest_ms:.4f}ms vs ours={serial_cost:.4f}ms)"
        )
    print(f"Done. CSV -> {out_csv}")


if __name__ == "__main__":
    main()
