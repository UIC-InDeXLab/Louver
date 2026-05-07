"""Python wrapper for the fused CPU TA-filter attend kernel."""
from __future__ import annotations

import math
import torch

from .._cpu_loader import load_ext

KERNEL_VERSION = "v1.0"

_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_ext(
            "ta_cpu_attend_v1",
            "attend/attend_v1.cpp",
        )
    return _EXT


def attend(
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict,
    buffer_keys: torch.Tensor,
    buffer_values: torch.Tensor,
    l_buf: int,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    if q.dtype != torch.float32:
        q = q.float()
    if threshold.dtype != torch.float32:
        threshold = threshold.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if q_head_to_kv is None:
        q2kv = torch.empty(0, dtype=torch.int64)
    else:
        q2kv = q_head_to_kv.to(torch.int64)

    return _ext().attend(
        q.contiguous(),
        state["centers_padded_f32"],
        state["assigns_padded_i32"],
        state["keys_padded_f32"],
        state["values_padded_f32"],
        state["invalid_mask"],
        state["dim_offsets"],
        state["dim_widths"],
        threshold.contiguous(),
        q2kv,
        buffer_keys,
        buffer_values,
        int(state["K_used"]),
        int(state["N_used"]),
        int(l_buf),
        float(scale),
    )
