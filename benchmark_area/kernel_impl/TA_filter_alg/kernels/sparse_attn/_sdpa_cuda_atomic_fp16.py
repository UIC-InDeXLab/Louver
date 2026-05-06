"""Custom CUDA masked fp16 decode-SDPA baseline.

The attention math is implemented in ``sdpa_cuda_atomic_kernel.cu``.  This
Python file only builds/caches the extension and owns reusable workspaces.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_EXT = None
_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    base = Path(__file__).resolve().parent
    if "CUDA_HOME" not in os.environ:
        for _p in ["/usr/local/cuda", "/usr/local/cuda-12.8", "/usr/local/cuda-12.4", "/usr/local/cuda-12"]:
            if os.path.isdir(_p):
                os.environ["CUDA_HOME"] = _p
                break
    os.environ["PATH"] = f"{os.environ.get('CUDA_HOME', '/usr/local/cuda')}/bin:" + os.environ.get("PATH", "")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        try:
            import torch as _torch
            maj, min_ = _torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{maj}.{min_}"
        except Exception:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    _EXT = load(
        name="ta_sdpa_cuda_atomic_fp16",
        sources=[
            str(base / "sdpa_cuda_atomic.cpp"),
            str(base / "sdpa_cuda_atomic_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def sdpa_cuda_atomic_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    mask_i8: torch.Tensor,
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    num_splits: int = 34,
) -> torch.Tensor:
    del q_head_to_kv
    if q.dtype != torch.float16 or keys_f16.dtype != torch.float16 or values_f16.dtype != torch.float16:
        raise TypeError("sdpa_cuda_atomic_fp16 expects fp16 q/keys/values")
    if mask_i8.dtype != torch.int8:
        raise TypeError("sdpa_cuda_atomic_fp16 expects int8 mask")
    h_q, d = q.shape
    h_kv, n_ctx, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    if d != 128 or d_k != 128 or d_v != 128:
        raise ValueError("sdpa_cuda_atomic_fp16 currently specializes D=Dv=128")
    if h_q % h_kv != 0:
        raise ValueError("sdpa_cuda_atomic_fp16 expects grouped GQA")
    if mask_i8.shape != (h_q, n_ctx):
        raise ValueError("mask_i8 must be (H_q, N)")
    if scale is None:
        scale = d ** -0.5

    key = (q.device.index, h_q, h_kv, n_ctx, d, d_v, int(num_splits))
    ws = _CACHE.get(key)
    cols = (n_ctx + int(num_splits) - 1) // int(num_splits)
    if ws is None:
        ws = {
            "partial_m": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32),
            "partial_l": torch.empty(h_q, num_splits, device=q.device, dtype=torch.float32),
            "partial_o": torch.empty(h_q, num_splits, d_v, device=q.device, dtype=torch.float32),
            "scores": torch.empty(h_q, num_splits, cols, device=q.device, dtype=torch.float32),
            "counters": torch.empty(h_q, device=q.device, dtype=torch.int32),
            "out": torch.empty(h_q, d_v, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    ext = _load_ext()
    return ext.forward(
        q.contiguous(),
        keys_f16.contiguous(),
        values_f16.contiguous(),
        mask_i8.contiguous(),
        ws["partial_m"],
        ws["partial_l"],
        ws["partial_o"],
        ws["scores"],
        ws["counters"],
        ws["out"],
        float(scale),
        int(num_splits),
    )
