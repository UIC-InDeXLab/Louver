"""Tune attention_v2_6 launch hyperparameters on the local GPU.

This is intentionally a small, direct script. It mirrors the attention portion
of ``kernels/kernel_bench/bench_attention.py`` for attention_v2_6, then
monkey-patches launch-time knobs before each timed run.

Example:
    python tuning.py \
        --input ../quick_pruning/capture_qkv_20000_Qwen_Qwen2.5-7B-Instruct.pt \
        --bf 4 --S 8 --buffer-len 256
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels import attention_v1_17, attention_v2_6
from hira.benchmark_area.kernel_impl.kernels.build_v2_7 import build as build_v2_7
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)


DEFAULT_PROMPT = "Tune attention_v2_6."


@dataclass(frozen=True, order=True)
class TrialConfig:
    num_splits: int
    parents_per_prog: int
    num_warps: int
    num_stages: int


def parse_int_list(raw: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not vals:
        raise argparse.ArgumentTypeError("empty integer list")
    return vals


def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def subspace_topk_thresholds(q, keys, topk, dim_slices):
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


def time_call(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def clear_v2_6_caches(state: dict) -> None:
    # A config change can alter Triton constexprs without changing tensor
    # pointers, so force a fresh fixed workspace / graph per trial.
    state.pop("_attn_v2_6_fixed", None)


def apply_trial_config(cfg: TrialConfig, bf: int) -> None:
    attention_v2_6._INDEX_NUM_WARPS = int(cfg.num_warps)
    attention_v2_6._INDEX_NUM_STAGES = int(cfg.num_stages)

    # attention_v2_6 calls attention_v1_17._parents_per_prog_for_bf(), which
    # derives PARENTS_PER_PROG from target child columns. Patch both targets so
    # this direct trial value is used for low- and high-group layouts.
    target_cols = int(cfg.parents_per_prog) * int(bf)
    attention_v1_17._TARGET_COLS_PER_CHUNK = target_cols
    attention_v1_17._TARGET_COLS_PER_CHUNK_HIGH_GROUPS = target_cols


def restore_defaults(defaults: dict) -> None:
    attention_v2_6._INDEX_NUM_WARPS = defaults["num_warps"]
    attention_v2_6._INDEX_NUM_STAGES = defaults["num_stages"]
    attention_v1_17._TARGET_COLS_PER_CHUNK = defaults["target_cols"]
    attention_v1_17._TARGET_COLS_PER_CHUNK_HIGH_GROUPS = defaults["target_cols_high"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Tune attention_v2_6 num_splits / parent tile / launch knobs."
    )
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=1000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--buffer-len", type=int, default=256)
    p.add_argument(
        "--num-splits-sweep",
        type=parse_int_list,
        default=parse_int_list("64,72,80,85,96,112,128,170"),
    )
    p.add_argument(
        "--parents-per-prog-sweep",
        type=parse_int_list,
        default=parse_int_list("16,32,64"),
        help=(
            "Direct values for Triton's PARENTS_PER_PROG constexpr. "
            "Only values where parents_per_prog * BF is a power of two are valid."
        ),
    )
    p.add_argument("--warps-sweep", type=parse_int_list, default=parse_int_list("4"))
    p.add_argument("--stages-sweep", type=parse_int_list, default=parse_int_list("3"))
    p.add_argument("--top-results", type=int, default=10)
    p.add_argument(
        "--no-sdpa",
        action="store_true",
        help="Skip the fp16 SDPA reference timing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    defaults = {
        "num_warps": attention_v2_6._INDEX_NUM_WARPS,
        "num_stages": attention_v2_6._INDEX_NUM_STAGES,
        "target_cols": attention_v1_17._TARGET_COLS_PER_CHUNK,
        "target_cols_high": attention_v1_17._TARGET_COLS_PER_CHUNK_HIGH_GROUPS,
    }

    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        cap = _capture_qkv(
            model_name=args.model,
            prompt_text=DEFAULT_PROMPT,
            n=args.n_tokens,
            device="cuda",
            torch_dtype=torch.float16,
            show_progress=True,
        )

    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError("attention_v2_6 tuning requires captured values.")

    keys_full = keys_cpu.to(device="cuda", dtype=torch.float32)
    values_full = values_cpu.to(device="cuda", dtype=torch.float32)
    queries = queries_cpu
    h_q = queries.shape[0]
    h_kv, n_total, d = keys_full.shape
    d_v = values_full.shape[-1]
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None
    groups = h_q // h_kv if q_head_to_kv is not None else 1

    if args.buffer_len < 0 or args.buffer_len >= n_total:
        raise ValueError(f"--buffer-len must be in [0, {n_total - 1}]")
    n_index = n_total - args.buffer_len
    keys = keys_full[:, :n_index, :].contiguous()
    values = values_full[:, :n_index, :].contiguous()
    buffer_keys = keys_full[:, n_index:, :].contiguous().to(torch.float16)
    buffer_values = values_full[:, n_index:, :].contiguous().to(torch.float16)
    keys_full_f16 = keys_full.to(torch.float16)
    values_full_f16 = values_full.to(torch.float16)

    print(
        f"attention_v2_6 tuning: layer={layer} H_q={h_q} H_kv={h_kv} "
        f"groups={groups} N_idx={n_index} N_buf={args.buffer_len} D={d} "
        f"BF={args.bf} S={args.S}"
    )
    print("Building build_v2_7 state ...")
    state = build_v2_7(
        keys,
        bf=args.bf,
        n_subspaces=args.S,
        refine_iter=args.refine_iter,
        values=values,
    )

    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )[: args.n_queries]
    keys_eval = keys_full if q_head_to_kv is None else keys_full[q_head_to_kv]
    query_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for qi in q_indices:
        q = queries[:, qi, :].to(device="cuda", dtype=torch.float32)
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = subspace_topk_thresholds(qn, keys_eval, args.topk, state["dim_slices"])
        q_norms = torch.stack(
            [qn[:, start:end].norm(dim=-1) for start, end in state["dim_slices"]],
            dim=0,
        )
        packed = torch.cat([th, q_norms], dim=0).to(torch.float16).contiguous()
        query_pairs.append((qn.to(torch.float16).contiguous(), packed))

    scale = 1.0 / (d ** 0.5)
    default_ppp = attention_v1_17._parents_per_prog_for_bf(args.bf, groups)
    default_cfg = TrialConfig(
        num_splits=attention_v2_6._NUM_SPLITS,
        parents_per_prog=default_ppp,
        num_warps=attention_v2_6._INDEX_NUM_WARPS,
        num_stages=attention_v2_6._INDEX_NUM_STAGES,
    )

    requested_configs = {
        TrialConfig(ns, ppp, nw, st)
        for ns, ppp, nw, st in itertools.product(
            args.num_splits_sweep,
            args.parents_per_prog_sweep,
            args.warps_sweep,
            args.stages_sweep,
        )
    }
    requested_configs.add(default_cfg)
    invalid_configs = [
        cfg for cfg in requested_configs
        if not is_power_of_two(cfg.parents_per_prog * args.bf)
    ]
    if invalid_configs:
        invalid_ppp = sorted({cfg.parents_per_prog for cfg in invalid_configs})
        print(
            "Skipping invalid parents_per_prog values "
            f"{invalid_ppp}: parents_per_prog * BF must be a power of two."
        )
    configs = {
        cfg for cfg in requested_configs
        if is_power_of_two(cfg.parents_per_prog * args.bf)
    }
    configs = sorted(configs)

    def run_attention_trial() -> None:
        for qn, th in query_pairs:
            attention_v2_6.attend(
                q=qn,
                th_per_subspace=th,
                state=state,
                buffer_keys=buffer_keys,
                buffer_values=buffer_values,
                keys_children=keys,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
                num_splits=current_cfg.num_splits,
            )

    results: list[tuple[float, TrialConfig]] = []
    try:
        print(
            f"Trials: {len(configs)} configs x {len(query_pairs)} queries x "
            f"{args.iters} timed iters"
        )
        for idx, cfg in enumerate(configs, start=1):
            current_cfg = cfg
            apply_trial_config(cfg, args.bf)
            clear_v2_6_caches(state)
            try:
                ms = time_call(run_attention_trial, iters=args.iters, warmup=args.warmup)
                per_q = ms / len(query_pairs)
                results.append((per_q, cfg))
                mark = "  default" if cfg == default_cfg else ""
                print(
                    f"[{idx:02d}/{len(configs):02d}] "
                    f"splits={cfg.num_splits:3d} ppp={cfg.parents_per_prog:2d} "
                    f"warps={cfg.num_warps} stages={cfg.num_stages}: "
                    f"{per_q:.4f} ms/query{mark}"
                )
            except Exception as exc:
                torch.cuda.synchronize()
                print(
                    f"[{idx:02d}/{len(configs):02d}] "
                    f"splits={cfg.num_splits:3d} ppp={cfg.parents_per_prog:2d} "
                    f"warps={cfg.num_warps} stages={cfg.num_stages}: "
                    f"FAILED ({type(exc).__name__}: {exc})"
                )
    finally:
        restore_defaults(defaults)
        clear_v2_6_caches(state)

    if not results:
        raise RuntimeError("all tuning trials failed")

    results.sort(key=lambda x: x[0])
    best_ms, best_cfg = results[0]
    default_ms = next((ms for ms, cfg in results if cfg == default_cfg), None)

    print("\nTop configs:")
    for rank, (ms, cfg) in enumerate(results[: args.top_results], start=1):
        delta = ""
        if default_ms is not None:
            delta = f"  delta_vs_default={ms - default_ms:+.4f}"
        print(
            f"{rank:2d}. {ms:.4f} ms/query  "
            f"splits={cfg.num_splits} parents_per_prog={cfg.parents_per_prog} "
            f"warps={cfg.num_warps} stages={cfg.num_stages}{delta}"
        )

    print("\nBest attention_v2_6 hyperparameters:")
    print(f"  num_splits:       {best_cfg.num_splits}")
    print(f"  parents_per_prog: {best_cfg.parents_per_prog}")
    print(f"  index_num_warps:  {best_cfg.num_warps}")
    print(f"  index_num_stages: {best_cfg.num_stages}")
    if default_ms is not None:
        print(f"  default_time:     {default_ms:.4f} ms/query")
        print(f"  best_time:        {best_ms:.4f} ms/query")
        print(f"  improvement:      {default_ms - best_ms:+.4f} ms/query")

    if not args.no_sdpa:
        def run_sdpa_fp16() -> None:
            for qn, _ in query_pairs:
                q4 = qn.view(1, h_q, 1, d)
                k4 = keys_full_f16.view(1, h_kv, n_total, d)
                v4 = values_full_f16.view(1, h_kv, n_total, d_v)
                torch.nn.functional.scaled_dot_product_attention(
                    q4,
                    k4,
                    v4,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=(groups > 1),
                )

        sdpa_ms = time_call(run_sdpa_fp16, iters=args.iters, warmup=args.warmup)
        print(f"  sdpa_fp16_time:   {sdpa_ms / len(query_pairs):.4f} ms/query")


if __name__ == "__main__":
    main()
