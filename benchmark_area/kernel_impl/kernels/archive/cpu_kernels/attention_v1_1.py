"""attention_v1.1 — single-pass online softmax + (h_q, parent-tile) splits."""

from __future__ import annotations

import math

import torch

from ._cpu_ext_loader import attention_ext

KERNEL_VERSION = "cpu_v1.1"

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = attention_ext("v1_1")
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
