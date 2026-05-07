"""Debug correctness — force all-alive via very low threshold, compare to dense."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_HIRA = Path(__file__).resolve().parents[5]
for p in (_HIRA.parent, _HIRA):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_index import (
    TAIndexCPU, TAIndexCPUConfig, baseline_dense,
)


def main():
    torch.manual_seed(0)
    H_q, H_kv, N, D = 4, 2, 256, 128
    q_head_to_kv = torch.arange(H_q) // (H_q // H_kv)
    keys = torch.randn(H_kv, N, D)
    values = torch.randn(H_kv, N, D)
    q = torch.randn(H_q, D)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    cfg = TAIndexCPUConfig(n_growth=0, refine_iter=2)
    index = TAIndexCPU(cfg).build(keys, values)

    # Very low threshold — force depth = full → all parents selected.
    th = torch.full((H_q,), -1e9, dtype=torch.float32)

    out_ours = index.attend(qn, th, q_head_to_kv=q_head_to_kv)
    # Use the index's stored (fp16-rounded) keys/values for a fair compare.
    keys_idx = index.state["keys_padded_f32"][:, :index.state["N_used"], :]
    values_idx = index.state["values_padded_f32"][:, :index.state["N_used"], :]
    out_ref = baseline_dense(qn, keys_idx, values_idx, q_head_to_kv=q_head_to_kv)
    diff = (out_ours - out_ref).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-9)
    print(f"all-alive: max_abs_diff={diff:.4e}  rel={rel:.4e}")
    print(f"  ours[0,:4]={out_ours[0,:4].tolist()}")
    print(f"  ref [0,:4]={out_ref[0,:4].tolist()}")


if __name__ == "__main__":
    main()
