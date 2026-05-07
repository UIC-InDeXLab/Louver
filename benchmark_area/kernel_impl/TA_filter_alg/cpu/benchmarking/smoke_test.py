"""Quick correctness + perf sanity check on synthetic data.

Run:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.benchmarking.smoke_test
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_HIRA = Path(__file__).resolve().parents[5]
_PARENT = _HIRA.parent
for p in (_PARENT, _HIRA):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_index import (
    TAIndexCPU, TAIndexCPUConfig, baseline_dense, baseline_sdpa,
)


def topk_threshold(q: torch.Tensor, keys: torch.Tensor,
                   q_head_to_kv: torch.Tensor | None, topk: int) -> torch.Tensor:
    keys_eval = keys if q_head_to_kv is None else keys.index_select(0, q_head_to_kv)
    scores = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, scores.shape[-1])
    top_vals, _ = scores.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous()


def run():
    torch.manual_seed(0)
    H_q, H_kv, N, D = 24, 8, 4096, 128
    q_head_to_kv = torch.arange(H_q) // (H_q // H_kv)

    keys = torch.randn(H_kv, N, D)
    values = torch.randn(H_kv, N, D)
    q = torch.randn(H_q, D)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cfg = TAIndexCPUConfig(n_growth=512, refine_iter=2)
    index = TAIndexCPU(cfg).build(keys, values)

    th = topk_threshold(qn, keys, q_head_to_kv, topk=20)

    out_ours = index.attend(qn, th, q_head_to_kv=q_head_to_kv)
    out_ref = baseline_dense(qn, keys, values, q_head_to_kv=q_head_to_kv)
    diff = (out_ours - out_ref).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-9)
    print(f"correctness: max_abs_diff={diff:.4e}  rel={rel:.4e}")

    # Quick timing
    for _ in range(3):
        index.attend(qn, th, q_head_to_kv=q_head_to_kv)
    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        index.attend(qn, th, q_head_to_kv=q_head_to_kv)
    t_ours = (time.perf_counter() - t0) / iters * 1000

    for _ in range(3):
        baseline_dense(qn, keys, values, q_head_to_kv=q_head_to_kv)
    t0 = time.perf_counter()
    for _ in range(iters):
        baseline_dense(qn, keys, values, q_head_to_kv=q_head_to_kv)
    t_dense = (time.perf_counter() - t0) / iters * 1000

    for _ in range(3):
        baseline_sdpa(qn, keys, values, q_head_to_kv=q_head_to_kv)
    t0 = time.perf_counter()
    for _ in range(iters):
        baseline_sdpa(qn, keys, values, q_head_to_kv=q_head_to_kv)
    t_sdpa = (time.perf_counter() - t0) / iters * 1000

    print(f"attend ours = {t_ours:.3f}ms  dense = {t_dense:.3f}ms  sdpa = {t_sdpa:.3f}ms  "
          f"speedup vs dense = {t_dense / t_ours:.2f}x  vs sdpa = {t_sdpa / t_ours:.2f}x")

    # Update sanity
    print("\nbuffer-fill + update test")
    for i in range(256):
        nk = torch.randn(H_kv, D)
        nv = torch.randn(H_kv, D)
        index.append_decoding_kv(nk, nv)
    t0 = time.perf_counter()
    index.update()
    print(f"update() = {(time.perf_counter() - t0) * 1000:.2f}ms  N_used={index.n_indexed}")


if __name__ == "__main__":
    run()
