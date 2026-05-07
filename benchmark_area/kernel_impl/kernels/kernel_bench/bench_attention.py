"""Micro-benchmark: compare all search_v* kernels.

Requires real keys/queries because search speed depends on distribution.
Either pass --input-qkv path.pt or let it capture from --model.

Usage:
    python -m hira.benchmark_area.kernel_impl.kernels.kernel_bench.bench_search \
        --input-qkv /path/to/capture.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.kernels import attention_kernels, search_kernels
from hira.benchmark_area.kernel_impl.kernels.build_v2_7 import build as build_v2_7
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _capture_qkv,
    _q_to_kv_map,
)

ATTENTION_BUILD_KERNELS = {
    "attention_v5_14": "build_v2_7",
}

FP16_ONLY_ATTENTION_KERNELS = {"attention_v5_14"}

BUILD_FNS = {
    "build_v2_7": build_v2_7,
}


def subspace_topk_thresholds(q, keys, topk, dim_slices):
    """Per-subspace thresholds derived from the true full-space top-k set."""
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


def time_call(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


_GREEN = "\033[32m"
_RESET = "\033[0m"


def format_ms(per_q: float, fp16_baseline: float | None = None) -> str:
    text = f"{per_q:8.3f}"
    if fp16_baseline is not None and per_q < fp16_baseline:
        return f"{_GREEN}{text}{_RESET}"
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, default=None)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--n-tokens", type=int, default=1000)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--bf", type=int, default=4)
    p.add_argument("--S", type=int, default=8)
    p.add_argument("--refine-iter", type=int, default=5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--buffer-len",
        type=int,
        default=0,
        help="Hold the final N-buffer_len rows out as the decoding buffer.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first kernel error instead of skipping incompatible kernels.",
    )
    args = p.parse_args()

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
    keys_full = keys_cpu.to(device="cuda", dtype=torch.float32)
    values_full = (
        values_cpu.to(device="cuda", dtype=torch.float32)
        if values_cpu is not None
        else None
    )
    queries = queries_cpu
    H_q = queries.shape[0]
    H_kv, N, D = keys_full.shape
    D_v = int(values_full.shape[-1]) if values_full is not None else D
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    if args.buffer_len < 0 or args.buffer_len >= N:
        raise ValueError(f"--buffer-len must be in [0, {N - 1}], got {args.buffer_len}")

    n_index = N - args.buffer_len
    keys = keys_full[:, :n_index, :].contiguous()
    values = (
        values_full[:, :n_index, :].contiguous() if values_full is not None else None
    )

    if args.buffer_len:
        buffer = keys_full[:, n_index:, :].contiguous()
        value_buffer = (
            values_full[:, n_index:, :].contiguous()
            if values_full is not None
            else None
        )
    else:
        buffer = torch.empty(H_kv, 0, D, device="cuda", dtype=torch.float32)
        value_buffer = (
            torch.empty(H_kv, 0, D_v, device="cuda", dtype=torch.float32)
            if values is not None
            else None
        )
    state_cache: dict[str, dict] = {}

    def get_state(build_name: str) -> dict:
        state = state_cache.get(build_name)
        if state is None:
            if build_name == "build_v2_7":
                state = BUILD_FNS[build_name](
                    keys, args.bf, args.S, args.refine_iter, values=values
                )
            else:
                state = BUILD_FNS[build_name](keys, args.bf, args.S, args.refine_iter)
            state_cache[build_name] = state
        return state

    # Pre-compute per-subspace thresholds over a sweep of queries; average across them.
    total_q = queries.shape[1]
    stride = max(1, total_q // args.n_queries)
    q_indices = list(
        range(total_q - 1, max(0, total_q - args.n_queries * stride) - 1, -stride)
    )[: args.n_queries]

    required_builds = {
        ATTENTION_BUILD_KERNELS[name]
        for name in attention_kernels()
        if name in ATTENTION_BUILD_KERNELS
    }
    query_pairs_by_build: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    query_pairs_fp16_by_build: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    keys_eval = keys_full if q_head_to_kv is None else keys_full[q_head_to_kv]

    # Precompute (q, thresholds) pairs per build layout to avoid including them in timing.
    for build_name in sorted(required_builds):
        state = get_state(build_name)
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        pairs_fp16: list[tuple[torch.Tensor, torch.Tensor]] = []
        for qi in q_indices:
            q = queries[:, qi, :].to(device="cuda", dtype=torch.float32)
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            th = subspace_topk_thresholds(qn, keys_eval, args.topk, state["dim_slices"])
            pairs.append((qn, th))
            q_norms = torch.stack(
                [qn[:, start:end].norm(dim=-1) for start, end in state["dim_slices"]],
                dim=0,
            )
            th_packed_fp16 = torch.cat(
                [th, q_norms],
                dim=0,
            ).to(torch.float16)
            pairs_fp16.append((qn.to(torch.float16), th_packed_fp16))
        query_pairs_by_build[build_name] = pairs
        query_pairs_fp16_by_build[build_name] = pairs_fp16

    print(
        f"search micro-bench: layer {layer} H_q={H_q} H_kv={H_kv} "
        f"N_idx={n_index} N_buf={args.buffer_len} D={D} S={args.S}"
    )
    print("-" * 70)

    def bench_fn(fn, state, query_pairs):
        def f():
            for qn, th in query_pairs:
                fn(
                    q=qn,
                    th_per_subspace=th,
                    state=state,
                    buffer_keys=buffer,
                    keys_children=keys,
                    q_head_to_kv=q_head_to_kv,
                )

        return f

    results = []
    attention_results: list[tuple[str, str, float]] = []
    skipped: list[str] = []

    def _record_skip(label: str) -> None:
        skipped.append(label)

    def _try_bench(label: str, build_name: str, fn) -> float | None:
        try:
            ms = time_call(fn, iters=args.iters, warmup=3)
            return ms
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if args.strict:
                raise
            _record_skip(label)
            print(f"  {label:<24s} skipped")
            return None

    # for name, info in sorted(search_kernels().items()):
    #     build_name = SEARCH_BUILD_KERNELS.get(name, "build_v1_0")
    #     label = f"{name} ({info.version})"
    #     try:
    #         state = get_state(build_name)
    #         query_pairs = query_pairs_by_build[build_name]
    #     except Exception as exc:
    #         if args.strict:
    #             raise
    #         _record_skip(label)
    #         print(f"  {label:<24s} skipped")
    #         continue
    #     ms = _try_bench(name, build_name, bench_fn(info.fn, state, query_pairs))
    #     if ms is None:
    #         continue
    #     per_q = ms / len(query_pairs)
    #     results.append((label, per_q))
    #     print(f"  {name:<24s} {info.version:<6s}  {per_q:8.3f} ms/query  [{build_name}]")

    # Torch baseline: brute-force dot product over all keys.
    # GQA-aware: keep K at H_kv and use grouped einsum rather than expanding to
    # H_q. Matches how real attention runs with GQA (no materialized H_q copies).
    baseline_pairs = query_pairs_by_build.get("build_v1_0")
    if baseline_pairs is None:
        baseline_pairs = next(iter(query_pairs_by_build.values()))
    baseline_pairs_fp16 = query_pairs_fp16_by_build.get("build_v1_0")
    if baseline_pairs_fp16 is None:
        baseline_pairs_fp16 = next(iter(query_pairs_fp16_by_build.values()))

    groups = H_q // H_kv if q_head_to_kv is not None else 1
    keys_full_f16 = keys_full.half()
    buffer_f16 = buffer.to(torch.float16)
    value_buffer_f16 = (
        value_buffer.to(torch.float16) if value_buffer is not None else None
    )

    def baseline():
        for qn, _ in baseline_pairs:
            if groups == 1:
                _ = torch.einsum("hd,hnd->hn", qn, keys_full)
            else:
                q_hg = qn.view(H_kv, groups, D)
                _ = torch.einsum("hgd,hnd->hgn", q_hg, keys_full)

    ms = time_call(baseline, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("torch_baseline (full dot)", per_q))
    print(f"  {'torch_baseline':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    def matmul_baseline():
        for qn, _ in baseline_pairs:
            if groups == 1:
                _ = torch.matmul(keys_full, qn.unsqueeze(-1)).squeeze(-1)
            else:
                q_hg = qn.view(H_kv, groups, D)
                _ = torch.matmul(q_hg, keys_full.transpose(-1, -2))

    ms = time_call(matmul_baseline, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("matmul baseline", per_q))
    print(f"  {'matmul baseline':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    # FP16 baselines for fair comparison with fp16-key search kernels

    def baseline_fp16():
        for qn, _ in baseline_pairs_fp16:
            if groups == 1:
                _ = torch.einsum("hd,hnd->hn", qn, keys_full_f16)
            else:
                q_hg = qn.view(H_kv, groups, D)
                _ = torch.einsum("hgd,hnd->hgn", q_hg, keys_full_f16)

    ms = time_call(baseline_fp16, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("torch_baseline fp16", per_q))
    print(f"  {'torch_baseline fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    def matmul_baseline_fp16():
        for qn, _ in baseline_pairs_fp16:
            if groups == 1:
                _ = torch.matmul(keys_full_f16, qn.unsqueeze(-1)).squeeze(-1)
            else:
                q_hg = qn.view(H_kv, groups, D)
                _ = torch.matmul(q_hg, keys_full_f16.transpose(-1, -2))

    ms = time_call(matmul_baseline_fp16, iters=args.iters, warmup=3)
    per_q = ms / len(baseline_pairs)
    results.append(("matmul baseline fp16", per_q))
    print(f"  {'matmul baseline fp16':<24s} {'-':<6s}  {per_q:8.3f} ms/query")

    # ── Fused attention kernels + attention baselines ──
    if values is not None:
        print("-" * 70)
        print("Attention (fused search + softmax + @V → (H_q, D_v))")
        import math

        scale = 1.0 / math.sqrt(D)
        values_full_f16 = values_full.half()
        successful_attention: list[tuple[str, object, str]] = []
        attention_section_rows: list[tuple[str, str, float, str | None]] = []

        def attention_inputs(
            attn_name: str,
            build_name: str,
        ) -> tuple[
            list[tuple[torch.Tensor, torch.Tensor]],
            torch.Tensor,
            torch.Tensor | None,
            torch.dtype,
        ]:
            if attn_name in FP16_ONLY_ATTENTION_KERNELS:
                return (
                    query_pairs_fp16_by_build[build_name],
                    buffer_f16,
                    value_buffer_f16,
                    torch.float16,
                )
            return (
                query_pairs_by_build[build_name],
                buffer,
                value_buffer,
                torch.float32,
            )

        def loose_threshold(
            attn_name: str,
            state: dict,
            qn: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            s_subspaces = len(state["dim_slices"])
            fill_value = (
                -1e9 if dtype == torch.float32 else float(torch.finfo(dtype).min)
            )
            th_loose = torch.full(
                (s_subspaces, H_q), fill_value, device="cuda", dtype=dtype
            )
            if attn_name in FP16_ONLY_ATTENTION_KERNELS:
                q_norms = torch.stack(
                    [
                        qn.float()[:, start:end].norm(dim=-1)
                        for start, end in state["dim_slices"]
                    ],
                    dim=0,
                ).to(dtype)
                return torch.cat([th_loose, q_norms], dim=0)
            return th_loose

        for name, info in sorted(attention_kernels().items()):
            if name not in ATTENTION_BUILD_KERNELS:
                continue
            build_name = ATTENTION_BUILD_KERNELS[name]
            label = f"{name} ({info.version})"
            try:
                state = get_state(build_name)
                query_pairs, buffer_arg, value_buffer_arg, _ = attention_inputs(
                    name, build_name
                )
            except Exception as exc:
                if args.strict:
                    raise
                _record_skip(label)
                print(f"  {label:<24s} skipped")
                continue

            def attend_fn():
                for qn, th in query_pairs:
                    info.fn(
                        q=qn,
                        th_per_subspace=th,
                        state=state,
                        buffer_keys=buffer_arg,
                        buffer_values=value_buffer_arg,
                        keys_children=keys,
                        q_head_to_kv=q_head_to_kv,
                        scale=scale,
                    )

            ms = _try_bench(name, build_name, attend_fn)
            if ms is None:
                continue
            per_q = ms / len(query_pairs)
            results.append((label, per_q))
            attention_results.append((name, info.version, per_q))
            successful_attention.append((name, info, build_name))
            attention_section_rows.append((name, info.version, per_q, build_name))

        # Dense attention baseline (fp32 math, matches our fused output dtype).
        # GQA-aware: no H_q expansion on K/V.
        def dense_attn_fp32():
            for qn, _ in baseline_pairs:
                if groups == 1:
                    scores = torch.einsum("hd,hnd->hn", qn, keys_full) * scale
                    probs = torch.softmax(scores, dim=-1)
                    _ = torch.einsum("hn,hnd->hd", probs, values_full)
                else:
                    q_hg = qn.view(H_kv, groups, D)
                    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys_full) * scale
                    probs = torch.softmax(scores, dim=-1)
                    _ = torch.einsum("hgn,hnd->hgd", probs, values_full)

        ms = time_call(dense_attn_fp32, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("dense attn fp32", per_q))
        attention_section_rows.append(("dense attn fp32", "-", per_q, None))

        # FP16 dense attention.
        def dense_attn_fp16():
            for qn, _ in baseline_pairs_fp16:
                if groups == 1:
                    scores = torch.einsum("hd,hnd->hn", qn, keys_full_f16) * scale
                    probs = torch.softmax(scores.float(), dim=-1).half()
                    _ = torch.einsum("hn,hnd->hd", probs, values_full_f16)
                else:
                    q_hg = qn.view(H_kv, groups, D)
                    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys_full_f16) * scale
                    probs = torch.softmax(scores.float(), dim=-1).half()
                    _ = torch.einsum("hgn,hnd->hgd", probs, values_full_f16)

        ms = time_call(dense_attn_fp16, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("dense attn fp16", per_q))
        attention_section_rows.append(("dense attn fp16", "-", per_q, None))

        # SDPA (flash backend chosen automatically). Uses enable_gqa so K/V stay
        # at H_kv — mirrors baseline_sdpa in index.py.
        def sdpa_baseline():
            for qn, _ in baseline_pairs:
                q4 = qn.view(1, H_q, 1, D)
                k4 = keys_full.view(1, H_kv, N, D)
                v4 = values_full.view(1, H_kv, N, D_v)
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q4,
                    k4,
                    v4,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=(groups > 1),
                )

        ms = time_call(sdpa_baseline, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("sdpa fp32", per_q))
        attention_section_rows.append(("sdpa fp32", "-", per_q, None))

        def sdpa_baseline_fp16():
            for qn, _ in baseline_pairs_fp16:
                q4 = qn.view(1, H_q, 1, D)
                k4 = keys_full_f16.view(1, H_kv, N, D)
                v4 = values_full_f16.view(1, H_kv, N, D_v)
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q4,
                    k4,
                    v4,
                    is_causal=False,
                    scale=scale,
                    enable_gqa=(groups > 1),
                )

        ms = time_call(sdpa_baseline_fp16, iters=args.iters, warmup=3)
        per_q = ms / len(baseline_pairs)
        results.append(("sdpa fp16", per_q))
        attention_section_rows.append(("sdpa fp16", "-", per_q, None))

        sdpa_fp16_per_q = per_q
        for row_name, row_version, row_ms, row_build in attention_section_rows:
            time_text = format_ms(row_ms, sdpa_fp16_per_q)
            if row_build is None:
                print(f"  {row_name:<24s} {row_version:<6s}  {time_text} ms/query")
            else:
                print(
                    f"  {row_name:<24s} {row_version:<6s}  "
                    f"{time_text} ms/query  [{row_build}]"
                )

        # Correctness checks: compare fused attention to dense attention.
        #   - tight gate (as timed): expected sparse-approximation error.
        #   - loose gate (all parents pass): should match dense to fp16 noise.
        if successful_attention:
            first_attn_name, info, build_name = successful_attention[0]
            state = get_state(build_name)
            first_pairs, first_buffer, first_value_buffer, first_th_dtype = (
                attention_inputs(first_attn_name, build_name)
            )
            qn0, th0 = first_pairs[0]
            th_loose = loose_threshold(first_attn_name, state, qn0, first_th_dtype)
            # Reference: GQA-aware dense attention (no H_q expansion).
            qn0_ref = qn0.float()
            if groups == 1:
                scores_ref = torch.einsum("hd,hnd->hn", qn0_ref, keys_full) * scale
                probs_ref = torch.softmax(scores_ref, dim=-1)
                out_ref = torch.einsum("hn,hnd->hd", probs_ref, values_full)
            else:
                q0_hg = qn0_ref.view(H_kv, groups, D)
                scores_ref = torch.einsum("hgd,hnd->hgn", q0_hg, keys_full) * scale
                probs_ref = torch.softmax(scores_ref, dim=-1)
                out_ref = torch.einsum("hgn,hnd->hgd", probs_ref, values_full).reshape(
                    H_q, D_v
                )
            ref_scale = out_ref.float().abs().max().item() + 1e-9

            for tag, th_used in (("tight(pruned)", th0), ("loose(all pass)", th_loose)):
                out_ours = info.fn(
                    q=qn0,
                    th_per_subspace=th_used,
                    state=state,
                    buffer_keys=first_buffer,
                    buffer_values=first_value_buffer,
                    keys_children=keys,
                    q_head_to_kv=q_head_to_kv,
                    scale=scale,
                )
                diff = (out_ours.float() - out_ref.float()).abs().max().item()
                print(
                    f"  correctness[{tag:<15s}]: max_abs_diff={diff:.4e}  "
                    f"rel={diff / ref_scale:.4e} ({first_attn_name})"
                )

            # Additional per-kernel correctness using loose gate, for non-first kernels.
            for attn_name, info_k, build_k in successful_attention[1:]:
                state_k = get_state(build_k)
                pairs_k, buffer_k, value_buffer_k, th_dtype_k = attention_inputs(
                    attn_name, build_k
                )
                qn_k, _ = pairs_k[0]
                th_loose_k = loose_threshold(attn_name, state_k, qn_k, th_dtype_k)
                try:
                    out_k = info_k.fn(
                        q=qn_k,
                        th_per_subspace=th_loose_k,
                        state=state_k,
                        buffer_keys=buffer_k,
                        buffer_values=value_buffer_k,
                        keys_children=keys,
                        q_head_to_kv=q_head_to_kv,
                        scale=scale,
                    )
                    diff_k = (out_k.float() - out_ref.float()).abs().max().item()
                    print(
                        f"  correctness[loose(all pass)]: max_abs_diff={diff_k:.4e}  "
                        f"rel={diff_k / ref_scale:.4e} ({attn_name})"
                    )
                except Exception as exc:
                    print(
                        f"  correctness[{attn_name}] FAILED: {type(exc).__name__}: {exc}"
                    )

    print("-" * 70)
    if results:
        best = min(results, key=lambda r: r[1])
        print(f"Fastest: {best[0]} at {best[1]:.3f} ms/query")
    else:
        print("Fastest: none (all kernels failed or were skipped)")
    if attention_results:
        best_attention = min(attention_results, key=lambda r: r[2])
        print(
            f"Fastest attention kernel: {best_attention[0]} "
            f"({best_attention[1]}) at {best_attention[2]:.3f} ms/query"
        )
    else:
        print("Fastest attention kernel: none (all attention kernels failed or were skipped)")
    if skipped:
        print("Skipped kernels:")
        for label in skipped:
            print(f"  {label:<24s} skipped")


if __name__ == "__main__":
    main()
