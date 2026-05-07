"""attention_v1.38 — v1.34 with SDPA-style split heuristic."""

from __future__ import annotations

import math

import torch

from . import attention_v1_34 as _v34

KERNEL_VERSION = "v1.38"


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    if batch_nheads_mblocks >= int(0.8 * num_sms):
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1
    eff = [0.0] * max_splits
    max_eff = 0.0
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        n_waves = float(batch_nheads_mblocks * s) / float(num_sms)
        cur = n_waves / math.ceil(n_waves)
        eff[s - 1] = cur
        max_eff = max(max_eff, cur)
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        if eff[s - 1] >= 0.85 * max_eff:
            return s
    return 1


def attend(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None,
    buffer_values: torch.Tensor | None,
    keys_children: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    num_splits: int = 32,
) -> torch.Tensor:
    layout = _v34._v31._get_layout_fp16(state, q_head_to_kv, q if q.is_contiguous() else q.contiguous())
    parents_per_prog = _v34._v31._parents_per_prog_for_bf(int(layout["bf"]), int(layout["groups"]))
    max_splits = int(state.get("_attn_v1_38_max_splits", max(num_splits, 64)))
    num_n_blocks = math.ceil(int(layout["K"]) / parents_per_prog)
    num_sms = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
    num_splits_eff = _num_splits_heuristic(
        batch_nheads_mblocks=int(layout["base_heads"]),
        num_sms=num_sms,
        num_n_blocks=num_n_blocks,
        max_splits=max_splits,
    )
    return _v34.attend(
        q=q,
        th_per_subspace=th_per_subspace,
        state=state,
        buffer_keys=buffer_keys,
        buffer_values=buffer_values,
        keys_children=keys_children,
        q_head_to_kv=q_head_to_kv,
        scale=scale,
        num_splits=num_splits_eff,
    )


KERNEL = attend
