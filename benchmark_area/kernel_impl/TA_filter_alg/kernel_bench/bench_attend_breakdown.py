"""Compare ``attend`` vs ``filter + sparse_attn`` timed in isolation.

Builds a TAIndex identical to what ``bench.py`` would have at decode-step 0
and times three things on the same query:

    1. ``filter_only_ms``      — ta_filter v8 alone (writes live_idx).
    2. ``sparse_attn_only_ms`` — sparse_attn v2.5 alone, reusing the
                                 live_idx from (1).
    3. ``attend_ms``           — the public ``index.attend``.

If ``attend_ms ≈ filter_only_ms + sparse_attn_only_ms`` then the only
overhead in ``attend`` is kernel-launch latency between the two kernels.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.kernel_impl.TA_filter_alg.index import (
    BUFFER_SIZE,
    TAIndex,
    TAIndexConfig,
    _filter_with_workspace,
)
from hira.benchmark_area.kernel_impl.TA_filter_alg.kernels.sparse_attn._sdpa_cuda_sparse_v2_5_fp16 import (
    sdpa_cuda_sparse_v2_5_fp16,
)
from hira.benchmark_area.quick_pruning.pruning_bench_utils import (
    CaptureState,
    _q_to_kv_map,
)


def time_repeated(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def topk_full_dot_threshold(q, keys, q_head_to_kv, topk):
    keys_eval = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, required=True)
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--prefill-frac", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--l-buf", type=int, default=0,
                   help="How many buffer keys to feed at probe time.")
    p.add_argument("--n-growth", type=int, default=BUFFER_SIZE * 4,
                   help="Match to bench.py's K_cap by setting this to "
                        "n_decode + BUFFER_SIZE.")
    p.add_argument("--iters", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    cap = CaptureState.load(args.input_qkv)
    layer_ids = cap.layer_ids()
    layer = args.layer if args.layer in layer_ids else layer_ids[len(layer_ids) // 2]
    qcpu, kcpu, vcpu = cap.to_layer_tensors(layer)
    if vcpu is None:
        raise RuntimeError("Captured values required.")

    keys = kcpu.to(device="cuda", dtype=torch.float16)
    values = vcpu.to(device="cuda", dtype=torch.float16)
    queries = qcpu.to(device="cuda", dtype=torch.float16)
    H_q = queries.shape[0]
    H_kv, N_total, D = keys.shape
    q_head_to_kv = _q_to_kv_map(H_q, H_kv, "cuda") if H_q != H_kv else None

    n_prefill = max(1, int(args.prefill_frac * N_total))
    if args.l_buf > BUFFER_SIZE:
        raise ValueError(f"l_buf must be <= {BUFFER_SIZE}")
    if n_prefill + args.l_buf >= queries.shape[1]:
        raise ValueError("not enough queries for the requested split")

    print(f"H_q={H_q} H_kv={H_kv} D={D} prefill={n_prefill} l_buf={args.l_buf}")

    cfg = TAIndexConfig(n_growth=args.n_growth)
    index = TAIndex(cfg)
    index.build(
        keys[:, :n_prefill, :].contiguous(),
        values[:, :n_prefill, :].contiguous(),
    )

    for s in range(args.l_buf):
        tok = n_prefill + s
        index.append_decoding_kv(
            keys[:, tok:tok + 1, :], values[:, tok:tok + 1, :]
        )

    q_idx = n_prefill + args.l_buf
    q = queries[:, q_idx, :]
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    th = topk_full_dot_threshold(
        qn, keys[:, :q_idx, :], q_head_to_kv, args.topk
    ).to(torch.float32)

    # Warm everything (compiles + caches).
    index.attend(qn, th, q_head_to_kv=q_head_to_kv)
    torch.cuda.synchronize()

    ws = index._ws
    scale = 1.0 / (D ** 0.5)

    def filter_only():
        _filter_with_workspace(
            q=qn, threshold=th, state=index.state,
            q_head_to_kv=q_head_to_kv,
            live_idx=ws["live_idx_filter"], live_count=ws["live_count"],
            top_scores=ws["top_scores"], top_indices=ws["top_indices"],
            depth=ws["depth"],
        )

    # Pre-run filter once so live_idx_filter is populated; sparse_only times
    # only sparse_attn v2.5.
    filter_only(); torch.cuda.synchronize()

    def sparse_only():
        sdpa_cuda_sparse_v2_5_fp16(
            q=qn,
            keys_f16=index.state["keys_padded_f16"],
            values_f16=index.state["values_padded_f16"],
            buffer_keys_f16=index._buf_keys_arena,
            buffer_values_f16=index._buf_values_arena,
            live_idx=ws["live_idx_filter"],
            live_count=ws["live_count"],
            l_buf=index._l_buf,
            q_head_to_kv=q_head_to_kv,
            scale=scale,
        )

    def attend_full():
        index.attend(qn, th, q_head_to_kv=q_head_to_kv)

    filter_ms = time_repeated(filter_only, iters=args.iters)
    sparse_ms = time_repeated(sparse_only, iters=args.iters)
    attend_ms = time_repeated(attend_full, iters=args.iters)

    sum_iso = filter_ms + sparse_ms
    glue = attend_ms - sum_iso

    print()
    print(f"  filter_only_ms      = {filter_ms:.4f}")
    print(f"  sparse_attn_only_ms = {sparse_ms:.4f}")
    print(f"  filter + sparse     = {sum_iso:.4f}")
    print(f"  attend_ms           = {attend_ms:.4f}")
    print(f"  glue (attend - sum) = {glue:+.4f}  ({glue/attend_ms*100:+.1f}% of attend)")
    print()
    if abs(glue) <= max(0.005, 0.10 * attend_ms):
        print("→ attend ≈ filter + sparse_attn  (within 5 µs / 10%).")
    else:
        print("→ measurable glue overhead.")


if __name__ == "__main__":
    main()
