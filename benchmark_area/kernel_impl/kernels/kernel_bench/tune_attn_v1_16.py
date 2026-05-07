"""Tune attention_v1_16 buffer kernel per bucket.

Sweeps (BUF_COLS_PER_PROG, num_warps, num_stages) for each bucket in
{64, 128, 256, 512}, with CUDA graph capture enabled, and prints the winning
config per (bucket, model) setup.

Also validates correctness (vs dense reference) once per bucket using the
winning config.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.tune_attn_v1_16 \\
        --input-qkv benchmark_area/quick_pruning/capture_qkv_8000_meta-llama_Llama-3.2-3B-Instruct.pt \\
        --S 8 --layer 15
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

from hira.benchmark_area.kernel_impl.kernels.attention_v1_16 import attend as attend_v1_16
from hira.benchmark_area.kernel_impl.kernels.build_v2_4 import build as build_v2_4
from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map

BUCKETS = (64, 128, 256, 512)

SWEEP_CFGS = [
    (cols, warps, stages)
    for cols in (16, 32, 64, 128)
    for warps in (2, 4, 8)
    for stages in (2, 3, 4)
]


def _time_attn(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _dense_attention(q, keys_q, values_q, scale):
    scores = torch.einsum("hd,hnd->hn", q, keys_q) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hn,hnd->hd", probs, values_q)


def _build_case(args):
    device = "cuda"
    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
        layer_ids = cap.layer_ids()
        layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
        queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
        keys = keys_cpu.to(device=device, dtype=torch.float32)
        values = values_cpu.to(device=device, dtype=torch.float32)
        queries = queries_cpu.to(device=device, dtype=torch.float32)
        h_q = queries.shape[0]
        h_kv = keys.shape[0]
        print(f"layer {layer}: H_q={h_q} H_kv={h_kv} D={keys.shape[-1]}")
    else:
        torch.manual_seed(0)
        h_q, h_kv = args.H_q, args.H_kv
        d, d_v = args.D, args.D_v
        n = args.N
        keys = torch.randn(h_kv, n + max(BUCKETS), d, device=device, dtype=torch.float32)
        values = torch.randn(h_kv, n + max(BUCKETS), d_v, device=device, dtype=torch.float32)
        queries = torch.randn(h_q, n + max(BUCKETS) + 1, d, device=device, dtype=torch.float32)
        print(f"[synthetic] H_q={h_q} H_kv={h_kv} D={d} D_v={d_v} N_base={n}")

    q_head_to_kv = _q_to_kv_map(h_q, h_kv, device) if h_q != h_kv else None

    n_base = args.N if args.input_qkv is None else max(
        args.N if args.N is not None else 0,
        int(keys.shape[1]) - max(BUCKETS) - 1,
    )
    n_base = min(n_base, int(keys.shape[1]) - max(BUCKETS))
    if n_base <= 0:
        raise RuntimeError(f"Not enough keys: n_total={keys.shape[1]} max_buf={max(BUCKETS)}")

    base_keys = keys[:, :n_base, :].contiguous()
    base_values = values[:, :n_base, :].contiguous()

    # Build state. S from args, BF=4 by default.
    print(f"Building v2_4 state: bf={args.bf} S={args.S} N_base={n_base}")
    state = build_v2_4(base_keys, args.bf, args.S, args.refine_iter, values=base_values)

    # Query (single decoding step).
    q_idx = n_base  # next query after the base index
    if q_idx >= queries.shape[1]:
        q_idx = queries.shape[1] - 1
    q = queries[:, q_idx, :].contiguous()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return {
        "state": state,
        "keys_full": keys,
        "values_full": values,
        "q": q,
        "q_head_to_kv": q_head_to_kv,
        "n_base": n_base,
        "h_q": int(q.shape[0]),
        "h_kv": int(keys.shape[0]),
    }


def _make_buffer(case, bucket: int, device: str):
    """Slice buffer keys/values covering [n_base : n_base + bucket)."""
    n_base = case["n_base"]
    buf_slice = slice(n_base, n_base + bucket)
    buffer_keys = case["keys_full"][:, buf_slice, :].contiguous()
    buffer_values = case["values_full"][:, buf_slice, :].contiguous()
    return buffer_keys, buffer_values


def _loose_threshold(state, h_q: int, device: str) -> torch.Tensor:
    s = len(state["assigns_reord"])
    return torch.full((s, h_q), -1.0e9, device=device, dtype=torch.float32)


def _threshold_topk(q, keys_q, topk: int, dim_slices) -> torch.Tensor:
    scores = torch.einsum("hd,hnd->hn", q, keys_q)
    k = min(topk, scores.shape[-1])
    topk_idx = scores.topk(k, dim=-1).indices
    ths = []
    for s, e in dim_slices:
        qs = q[:, s:e]
        ks = keys_q[:, :, s:e]
        ss = torch.einsum("hd,hnd->hn", qs, ks)
        sub_top = ss.gather(1, topk_idx)
        ths.append(sub_top.min(dim=1).values)
    return torch.stack(ths, dim=0).contiguous()


def _reset_cache(state: dict) -> None:
    """Force recapture (config changed)."""
    for key in ("_attn_v1_16_fixed", "_attn_v1_16_layout"):
        state.pop(key, None)


def _check_correctness(case, bucket: int, cfg_tuple: tuple[int, int, int],
                       tight: bool) -> float:
    state = case["state"]
    q = case["q"]
    h_q = case["h_q"]
    device = "cuda"
    keys_full = case["keys_full"]
    values_full = case["values_full"]
    q_head_to_kv = case["q_head_to_kv"]
    n_base = case["n_base"]
    buffer_keys, buffer_values = _make_buffer(case, bucket, device)
    d = int(q.shape[-1])
    scale = 1.0 / math.sqrt(d)

    full_keys = torch.cat([keys_full[:, :n_base, :], buffer_keys], dim=1)
    full_values = torch.cat([values_full[:, :n_base, :], buffer_values], dim=1)
    keys_q = full_keys if q_head_to_kv is None else full_keys.index_select(0, q_head_to_kv)
    values_q = full_values if q_head_to_kv is None else full_values.index_select(0, q_head_to_kv)
    dense = _dense_attention(q, keys_q, values_q, scale)

    if tight:
        th = _threshold_topk(q, keys_q, 64, state["dim_slices"])
    else:
        th = _loose_threshold(state, h_q, device)

    _reset_cache(state)
    state["_attn_v1_16_buffer_cfg"] = {
        bucket: {"cols": cfg_tuple[0], "num_warps": cfg_tuple[1], "num_stages": cfg_tuple[2]},
    }
    state["_attn_v1_16_use_cuda_graphs"] = True

    out = attend_v1_16(
        q=q, th_per_subspace=th, state=state,
        buffer_keys=buffer_keys, buffer_values=buffer_values,
        keys_children=full_keys, q_head_to_kv=q_head_to_kv, scale=scale,
    )
    return (out.float() - dense.float()).abs().max().item()


def _bench_cfg(case, bucket: int, cfg_tuple: tuple[int, int, int],
               iters: int, warmup: int) -> float:
    state = case["state"]
    q = case["q"]
    h_q = case["h_q"]
    device = "cuda"
    buffer_keys, buffer_values = _make_buffer(case, bucket, device)
    d = int(q.shape[-1])
    scale = 1.0 / math.sqrt(d)
    th = _loose_threshold(state, h_q, device)
    q_head_to_kv = case["q_head_to_kv"]
    full_keys = case["keys_full"][:, : case["n_base"] + bucket, :]

    _reset_cache(state)
    state["_attn_v1_16_buffer_cfg"] = {
        bucket: {"cols": cfg_tuple[0], "num_warps": cfg_tuple[1], "num_stages": cfg_tuple[2]},
    }
    state["_attn_v1_16_use_cuda_graphs"] = True

    def call():
        attend_v1_16(
            q=q, th_per_subspace=th, state=state,
            buffer_keys=buffer_keys, buffer_values=buffer_values,
            keys_children=full_keys, q_head_to_kv=q_head_to_kv, scale=scale,
        )

    try:
        return _time_attn(call, iters=iters, warmup=warmup)
    except Exception:
        # Typically OutOfResources (shared memory) for large cols * stages.
        return float("inf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--H-q", type=int, default=24)
    p.add_argument("--H-kv", type=int, default=8)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--D", type=int, default=128)
    p.add_argument("--D-v", type=int, default=128)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8, choices=[8, 16])
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--buckets", type=str, default="64,128,256,512")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--check-correctness", action="store_true",
                   help="Verify each bucket's winner vs dense reference (slow).")
    p.add_argument("--quick", action="store_true",
                   help="Use a smaller sweep space for a fast pass.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    for b in buckets:
        if b not in BUCKETS:
            raise ValueError(f"bucket {b} not in {BUCKETS}")

    case = _build_case(args)

    cfgs = SWEEP_CFGS
    if args.quick:
        cfgs = [(c, w, s) for (c, w, s) in cfgs if s != 4 and c != 16]
    print(f"\nSweeping {len(cfgs)} configs per bucket "
          f"({'quick' if args.quick else 'full'}).")
    print("=" * 80)

    best_per_bucket: dict[int, tuple[float, tuple[int, int, int]]] = {}
    for bucket in buckets:
        print(f"\n[bucket {bucket}]")
        print(f"  {'cols':>5s}  {'warps':>5s}  {'stages':>6s}  {'ms':>8s}")
        print("  " + "-" * 36)
        best_ms = float("inf")
        best_cfg: tuple[int, int, int] | None = None
        # Valid: cols <= bucket (no point tiling bigger than bucket).
        bucket_cfgs = [(c, w, s) for (c, w, s) in cfgs if c <= bucket]
        for cfg in bucket_cfgs:
            ms = _bench_cfg(case, bucket, cfg, iters=args.iters, warmup=args.warmup)
            marker = ""
            if ms < best_ms:
                best_ms = ms
                best_cfg = cfg
                marker = "  <-- best"
            print(f"  {cfg[0]:>5d}  {cfg[1]:>5d}  {cfg[2]:>6d}  {ms:>8.4f}{marker}")
        if best_cfg is None:
            print(f"  [bucket {bucket}] ALL FAILED")
            continue
        best_per_bucket[bucket] = (best_ms, best_cfg)

    print("\n" + "=" * 80)
    print("Winners per bucket:")
    print(f"  {'bucket':>6s}  {'cols':>5s}  {'warps':>5s}  {'stages':>6s}  {'ms':>8s}")
    print("  " + "-" * 44)
    for bucket in buckets:
        if bucket not in best_per_bucket:
            continue
        ms, cfg = best_per_bucket[bucket]
        print(f"  {bucket:>6d}  {cfg[0]:>5d}  {cfg[1]:>5d}  {cfg[2]:>6d}  {ms:>8.4f}")

    print("\n  Code snippet for _DEFAULT_BUFFER_CFG:")
    print("  _DEFAULT_BUFFER_CFG = {")
    for bucket in buckets:
        if bucket not in best_per_bucket:
            continue
        _, cfg = best_per_bucket[bucket]
        print(f"      {bucket:>3d}: " + "{" +
              f'"cols": {cfg[0]}, "num_warps": {cfg[1]}, "num_stages": {cfg[2]}' + "},")
    print("  }")

    if args.check_correctness:
        print("\nCorrectness check (winning config per bucket, vs dense):")
        for bucket in buckets:
            if bucket not in best_per_bucket:
                continue
            _, cfg = best_per_bucket[bucket]
            diff_loose = _check_correctness(case, bucket, cfg, tight=False)
            print(f"  bucket={bucket:>3d}  loose-th  max_abs_diff={diff_loose:.4e}")


if __name__ == "__main__":
    main()
