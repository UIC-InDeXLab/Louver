"""Micro-benchmark for TA-filter attention kernels.

Discovers ``TA_attention_v_*`` modules under ``TA_filter_alg/kernels`` and
times them against fp16 dense attention baselines (einsum and torch SDPA).

Usage:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.kernel_bench.bench_TA_attention \\
        --input-qkv /path/to/capture.pt --topk 20

The TA-filter threshold T per query head is the kth-largest exact full-vector
dot product (k=topk).  This matches the threshold used in
``analyze_subspace_kcenter_rows.py`` and is the natural top-k recovery target
for the attention output.

Correctness is checked against a reference TA implementation (the same as
``TA_attention_v_1_0``) running in fp32: every kernel's output should match
that reference up to fp16 noise.
"""

from __future__ import annotations

import argparse
import copy
import glob
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg import attention_kernels, build_kernels
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_atomic_fp16 import (
    sdpa_cuda_atomic_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_6_fp16 import (
    sdpa_cuda_sparse_v1_6_fp16,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.baselines._sdpa_cuda_sparse_v1_20_fp16 import (
    sdpa_cuda_sparse_v1_20_fp16,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)


_GREEN = "\033[32m"
_RESET = "\033[0m"


def topk_full_dot_threshold(
    q: torch.Tensor,
    keys_full: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    topk: int,
) -> torch.Tensor:
    """T per head = k-th largest exact q.k value (full dim)."""
    keys_eval = keys_full if q_head_to_kv is None else keys_full.index_select(
        0, q_head_to_kv
    )
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def reference_attention(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    q_head_to_kv: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """fp32 reference: TA-filter sweep, scalar T survival, buffer fold, softmax @V."""
    from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.commons._TA_common import (
        build_selected_clusters,
        compute_centroid_scores,
        expand_for_query,
        per_key_candidate_mask,
        stop_depth_per_head,
    )

    scores_h_s_k = compute_centroid_scores(
        q=q,
        centers_padded_f16=state["centers_padded_f16"],
        dim_slices=state["dim_slices"],
        q_head_to_kv=q_head_to_kv,
    )
    sorted_scores, order = torch.sort(scores_h_s_k, dim=-1, descending=True)
    threshold_f32 = threshold.float()
    depth = stop_depth_per_head(sorted_scores, threshold_f32)
    selected = build_selected_clusters(order, depth)
    cand_mask = per_key_candidate_mask(
        selected=selected,
        assigns_padded=state["assigns_padded"],
        q_head_to_kv=q_head_to_kv,
    )                                                               # (H_q, N_pad)

    keys_padded_f16 = state["keys_padded_f16"]
    values_padded_f16 = state["values_padded_f16"]
    invalid_mask = state["invalid_mask"]
    keys_eff = expand_for_query(keys_padded_f16, q_head_to_kv).float()
    values_eff = expand_for_query(values_padded_f16, q_head_to_kv).float()
    invalid_eff = expand_for_query(invalid_mask, q_head_to_kv)

    qf = q.float()
    raw_scores = torch.einsum("hd,hnd->hn", qf, keys_eff)
    survive = cand_mask & (~invalid_eff) & (raw_scores >= threshold_f32.unsqueeze(-1))
    scaled = (raw_scores * scale).masked_fill(~survive, float("-inf"))

    has_buf = (
        buffer_keys is not None
        and buffer_values is not None
        and int(buffer_keys.shape[1]) > 0
    )
    if has_buf:
        kbuf = expand_for_query(buffer_keys, q_head_to_kv).float()
        vbuf = expand_for_query(buffer_values, q_head_to_kv).float()
        buf_scores = torch.einsum("hd,hnd->hn", qf, kbuf) * scale
        scaled = torch.cat([scaled, buf_scores], dim=-1)
        values_eff = torch.cat([values_eff, vbuf], dim=1)

    m = scaled.amax(dim=-1, keepdim=True)
    m = m.masked_fill(torch.isinf(m), 0.0)
    e = torch.exp(scaled - m)
    denom = e.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    probs = e / denom
    out = torch.einsum("hn,hnd->hd", probs, values_eff)
    return out


def time_call(fn, iters: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def format_ms(per_q: float, baseline: float | None) -> str:
    text = f"{per_q:8.3f}"
    if baseline is not None and per_q < baseline:
        return f"{_GREEN}{text}{_RESET}"
    return text


def _parse_kernel_version(name: str) -> tuple[int, int]:
    # Expects basename like TA_attention_v_11_2 / TA_build_v_10_0.
    base = name.rsplit(".", 1)[-1]
    parts = base.split("_")
    if len(parts) < 5:
        return (0, 0)
    try:
        return (int(parts[-2]), int(parts[-1]))
    except ValueError:
        return (0, 0)


def _build_for_attention(attn_name: str, build_registry: dict) -> str:
    major, minor = _parse_kernel_version(attn_name)
    build_by_base = {name.rsplit(".", 1)[-1]: name for name in build_registry}

    def resolve(base_name: str) -> str | None:
        return build_by_base.get(base_name)

    candidates = []
    exact = f"TA_build_v_{major}_{minor}"
    exact_key = resolve(exact)
    if exact_key is not None:
        candidates.append(exact_key)
    if major >= 13:
        k = resolve("TA_build_v_13_0")
        if k is not None:
            candidates.append(k)
    if major >= 11:
        k = resolve("TA_build_v_11_0")
        if k is not None:
            candidates.append(k)
    if major >= 10:
        k = resolve("TA_build_v_10_0")
        if k is not None:
            candidates.append(k)
    if major >= 7:
        k = resolve("TA_build_v_7_0")
        if k is not None:
            candidates.append(k)
    k = resolve("TA_build_v_1_1")
    if k is not None:
        candidates.append(k)
    for name in candidates:
        if name in build_registry:
            return name
    available = ", ".join(sorted(build_registry))
    raise ValueError(f"no compatible build kernel found for {attn_name}; available: {available}")


def _run_once(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    if args.input_qkv is not None:
        print(f"Loading capture from {args.input_qkv} ...")
        cap = CaptureState.load(args.input_qkv)
    else:
        print(f"Capturing {args.n_tokens} tokens from {args.model} ...")
        cap = _capture_qkv(
            model_name=args.model,
            prompt_text="Benchmark.",
            n=args.n_tokens,
            device="cuda",
            torch_dtype=torch.float16,
            show_progress=True,
        )

    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    queries_cpu, keys_cpu, values_cpu = cap.to_layer_tensors(layer)
    if values_cpu is None:
        raise RuntimeError("TA-filter attention requires captured values.")

    keys_full = keys_cpu.to(device="cuda", dtype=torch.float32)
    values_full = values_cpu.to(device="cuda", dtype=torch.float32)
    queries = queries_cpu

    h_q = queries.shape[0]
    h_kv, n_total, d = keys_full.shape
    d_v = int(values_full.shape[-1])
    q_head_to_kv = _q_to_kv_map(h_q, h_kv, "cuda") if h_q != h_kv else None

    if args.buffer_len < 0 or args.buffer_len >= n_total:
        raise ValueError(f"--buffer-len must be in [0, {n_total - 1}], got {args.buffer_len}")

    n_index = n_total - args.buffer_len
    keys_index = keys_full[:, :n_index, :].contiguous()
    values_index = values_full[:, :n_index, :].contiguous()
    if args.buffer_len:
        buffer_keys = keys_full[:, n_index:, :].contiguous()
        buffer_values = values_full[:, n_index:, :].contiguous()
    else:
        buffer_keys = torch.empty(h_kv, 0, d, device="cuda", dtype=torch.float32)
        buffer_values = torch.empty(h_kv, 0, d_v, device="cuda", dtype=torch.float32)

    build_registry = build_kernels()
    built_states: dict[str, dict] = {}

    scale = 1.0 / math.sqrt(d)

    # Pick a sweep of queries (descending stride from end).
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )[: args.n_queries]

    # Pre-compute (q_fp16, threshold_fp32) pairs for fair timing (no prep inside).
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    pairs_fp32: list[tuple[torch.Tensor, torch.Tensor]] = []
    for qi in q_indices:
        q = queries[:, qi, :].to(device="cuda", dtype=torch.float32)
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        th = topk_full_dot_threshold(qn, keys_full, q_head_to_kv, args.topk)
        pairs_fp32.append((qn, th.float()))
        pairs.append((qn.to(torch.float16), th.float()))

    print(f"Layer {layer}: queries={len(pairs)} topk={args.topk}")
    print("-" * 70)

    keys_full_f16 = keys_full.half()
    values_full_f16 = values_full.half()
    buffer_keys_f16 = buffer_keys.to(torch.float16)
    buffer_values_f16 = buffer_values.to(torch.float16)

    groups = (h_q // h_kv) if q_head_to_kv is not None else 1
    n_full = n_total

    # ── baselines (dense attention) ──
    def dense_attn_fp16():
        for qn, _ in pairs:
            if groups == 1:
                scores = torch.einsum("hd,hnd->hn", qn, keys_full_f16) * scale
                probs = torch.softmax(scores.float(), dim=-1).half()
                _ = torch.einsum("hn,hnd->hd", probs, values_full_f16)
            else:
                q_hg = qn.view(h_kv, groups, d)
                scores = torch.einsum("hgd,hnd->hgn", q_hg, keys_full_f16) * scale
                probs = torch.softmax(scores.float(), dim=-1).half()
                _ = torch.einsum("hgn,hnd->hgd", probs, values_full_f16)

    dense_ms = time_call(dense_attn_fp16, iters=args.iters)
    dense_per_q = dense_ms / len(pairs)
    print(f"  {'dense attn fp16':<28s} {'-':<6s}  {dense_per_q:8.3f} ms/query")

    def sdpa_fp16():
        for qn, _ in pairs:
            q4 = qn.view(1, h_q, 1, d)
            k4 = keys_full_f16.view(1, h_kv, n_full, d)
            v4 = values_full_f16.view(1, h_kv, n_full, d_v)
            _ = torch.nn.functional.scaled_dot_product_attention(
                q4, k4, v4,
                is_causal=False,
                scale=scale,
                enable_gqa=(groups > 1),
            )

    sdpa_ms = time_call(sdpa_fp16, iters=args.iters)
    sdpa_per_q = sdpa_ms / len(pairs)
    print(f"  {'sdpa fp16':<28s} {'-':<6s}  {sdpa_per_q:8.3f} ms/query")

    mask_all_true = torch.ones(h_q, n_full, device="cuda", dtype=torch.int8)

    def sdpa_cuda_atomic_fp16_loop():
        for qn, _ in pairs:
            _ = sdpa_cuda_atomic_fp16(
                q=qn,
                keys_f16=keys_full_f16,
                values_f16=values_full_f16,
                mask_i8=mask_all_true,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
            )

    try:
        sdpa_cuda_atomic_ms = time_call(sdpa_cuda_atomic_fp16_loop, iters=args.iters)
        sdpa_cuda_atomic_per_q = sdpa_cuda_atomic_ms / len(pairs)
        print(
            f"  {'sdpa_cuda_atomic_fp16':<28s} {'-':<6s}  "
            f"{sdpa_cuda_atomic_per_q:8.3f} ms/query"
        )
    except Exception as exc:
        torch.cuda.synchronize()
        if args.strict:
            raise
        print(f"  {'sdpa_cuda_atomic_fp16':<28s} {'-':<6s}  skipped: {type(exc).__name__}: {exc}")

    def sdpa_cuda_sparse_v1_6_fp16_loop():
        for qn, _ in pairs:
            _ = sdpa_cuda_sparse_v1_6_fp16(
                q=qn,
                keys_f16=keys_full_f16,
                values_f16=values_full_f16,
                mask_i8=mask_all_true,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
            )

    try:
        sdpa_cuda_sparse_ms = time_call(sdpa_cuda_sparse_v1_6_fp16_loop, iters=args.iters)
        sdpa_cuda_sparse_per_q = sdpa_cuda_sparse_ms / len(pairs)
        print(
            f"  {'sdpa_cuda_sparse_v1_6_fp16':<28s} {'-':<6s}  "
            f"{sdpa_cuda_sparse_per_q:8.3f} ms/query"
        )
    except Exception as exc:
        torch.cuda.synchronize()
        if args.strict:
            raise
        print(f"  {'sdpa_cuda_sparse_v1_6_fp16':<28s} {'-':<6s}  skipped: {type(exc).__name__}: {exc}")

    def sdpa_cuda_sparse_v1_20_fp16_loop():
        for qn, _ in pairs:
            _ = sdpa_cuda_sparse_v1_20_fp16(
                q=qn,
                keys_f16=keys_full_f16,
                values_f16=values_full_f16,
                mask_i8=mask_all_true,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
            )

    try:
        sdpa_cuda_sparse_v1_20_ms = time_call(sdpa_cuda_sparse_v1_20_fp16_loop, iters=args.iters)
        sdpa_cuda_sparse_v1_20_per_q = sdpa_cuda_sparse_v1_20_ms / len(pairs)
        print(
            f"  {'sdpa_cuda_sparse_v1_20_fp16':<28s} {'-':<6s}  "
            f"{sdpa_cuda_sparse_v1_20_per_q:8.3f} ms/query"
        )
    except Exception as exc:
        torch.cuda.synchronize()
        if args.strict:
            raise
        print(f"  {'sdpa_cuda_sparse_v1_20_fp16':<28s} {'-':<6s}  skipped: {type(exc).__name__}: {exc}")

    # ── TA kernels ──
    print("-" * 70)
    # print(f"TA-filter attention (T = {args.topk}-th largest exact dot per head)")
    results: list[tuple[str, float]] = []
    skipped: list[str] = []

    for name, info in sorted(attention_kernels().items()):
        label = f"{name} ({info.version})"
        build_name = _build_for_attention(name, build_registry)
        build_ver = build_registry[build_name].version
        if build_name not in built_states:
            build_info = build_registry[build_name]
            # print(
            #     f"Building TA index with {build_name} ({build_info.version}) for {name}: "
            #     f"H_q={h_q} H_kv={h_kv} N_idx={n_index} N_buf={args.buffer_len} "
            #     f"D={d} S={args.S} bf={args.bf} ..."
            # )
            t0 = time.perf_counter()
            built_states[build_name] = build_info.fn(
                keys=keys_index,
                bf=args.bf,
                n_subspaces=args.S,
                refine_iter=args.refine_iter,
                values=values_index,
            )
            torch.cuda.synchronize()
            # print(f"TA index build ({build_name}): {(time.perf_counter() - t0) * 1000:.1f} ms")
        state = built_states[build_name]
        try:
            qn, th = pairs[0]
            info.fn(
                q=qn,
                threshold=th,
                state=state,
                buffer_keys=buffer_keys_f16 if args.buffer_len else None,
                buffer_values=buffer_values_f16 if args.buffer_len else None,
                q_head_to_kv=q_head_to_kv,
                scale=scale,
            )
        except Exception as exc:
            torch.cuda.synchronize()
            if args.strict:
                raise
            skipped.append(label)
            print(
                f"  {name:<28s} {info.version:<6s}  {build_ver:<8s}  "
                f"skipped: {type(exc).__name__}: {exc}"
            )
            continue

        def attend_loop():
            for qn, th in pairs:
                info.fn(
                    q=qn,
                    threshold=th,
                    state=state,
                    buffer_keys=buffer_keys_f16 if args.buffer_len else None,
                    buffer_values=buffer_values_f16 if args.buffer_len else None,
                    q_head_to_kv=q_head_to_kv,
                    scale=scale,
                )

        try:
            ms = time_call(attend_loop, iters=args.iters)
        except Exception as exc:
            torch.cuda.synchronize()
            if args.strict:
                raise
            skipped.append(label)
            print(
                f"  {name:<28s} {info.version:<6s}  {build_ver:<8s}  "
                f"skipped (timing): {type(exc).__name__}: {exc}"
            )
            continue
        per_q = ms / len(pairs)
        results.append((name, per_q))
        ms_text = format_ms(per_q, sdpa_per_q)
        print(f"  {name:<28s} {info.version:<6s}  {build_ver:<8s}  {ms_text} ms/query")

    print("-" * 70)
    if results:
        best = min(results, key=lambda r: r[1])
        print(f"Fastest TA kernel: {best[0]} at {best[1]:.3f} ms/query")
        print(f"sdpa fp16 baseline: {sdpa_per_q:.3f} ms/query "
              f"(ratio {best[1] / sdpa_per_q:.2f}x)")
    else:
        print("No TA kernels succeeded.")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")


def _expand_input_paths(args) -> list[Path | None]:
    paths: list[Path] = []
    for p in args.inputs or []:
        paths.append(Path(p))
    if args.input_glob:
        for match in sorted(glob.glob(args.input_glob)):
            paths.append(Path(match))
    if args.input_qkv is not None:
        paths.append(args.input_qkv)

    if not paths:
        return [None]

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", "--input", dest="input_qkv", type=Path, default=None)
    p.add_argument("--inputs", nargs="*", default=None,
                   help="One or more capture files to benchmark sequentially.")
    p.add_argument("--input-glob", type=str, default=None,
                   help="Glob of capture files to benchmark sequentially.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=2000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--topk", type=int, default=20,
                   help="k for the threshold T = kth-largest exact full-dot.")
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--buffer-len", "--buffer", dest="buffer_len", type=int, default=0,
                   help="Hold the final N - buffer_len rows out as the decoding buffer.")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    input_paths = _expand_input_paths(args)
    for idx, path in enumerate(input_paths):
        run_args = copy.copy(args)
        run_args.input_qkv = path
        if len(input_paths) > 1:
            print("=" * 70)
            print(f"Capture {idx + 1}/{len(input_paths)}: {path}")
        _run_once(run_args)


if __name__ == "__main__":
    main()
