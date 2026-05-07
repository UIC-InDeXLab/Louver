"""attention_v1.4 — BF16 keys/values + AVX-512 VDPBF16PS dot.

Inputs (q, th, buffer_keys, buffer_values) must be bf16 — the bench is
responsible for converting outside the timed region (mirrors how the GPU
v5_14 kernel insists on fp16 inputs).

The state's keys_reord / values_reord / anchor centers are converted to bf16
on first call and cached on the state dict (`state["_v1_4_cache"]`).
"""

from __future__ import annotations

import math

import torch

from ._cpu_ext_loader import attention_ext

KERNEL_VERSION = "cpu_v1.4"

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = attention_ext("v1_4")
    return _EXT


def attend(
    q: torch.Tensor,
    th_per_subspace: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor | None = None,
    buffer_values: torch.Tensor | None = None,
    keys_children: torch.Tensor | None = None,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return _ext().attend(
        q, th_per_subspace, state,
        buffer_keys, buffer_values,
        q_head_to_kv, float(scale),
    )


KERNEL = attend
