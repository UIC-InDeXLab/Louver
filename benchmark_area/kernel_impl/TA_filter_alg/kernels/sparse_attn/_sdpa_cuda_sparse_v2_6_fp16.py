"""v2.6 sparse decode-SDPA: bitmap-driven (no live_idx).

Consumes ``parent_alive_bitmap`` (output of ta_filter v9) + ``assigns_packed``
directly.  Per (hq, split): walks ``[0, N_used)`` index keys, checks the
bitmap for each key's 4 subspace parents, runs online softmax for survivors;
then walks ``[0, l_buf)`` buffer keys (always alive).
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
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    _EXT = load(
        name="ta_sdpa_cuda_sparse_v2_6_fp16",
        sources=[
            str(base / "sdpa_cuda_sparse_v2_6.cpp"),
            str(base / "sdpa_cuda_sparse_v2_6_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def sdpa_cuda_sparse_v2_6_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buffer_keys_f16: torch.Tensor,
    buffer_values_f16: torch.Tensor,
    parent_alive_bitmap: torch.Tensor,    # (Hq, 4, K_words) int32
    assigns_packed: torch.Tensor,         # (Hkv, N_pad) int64
    N_used: int,
    l_buf: int,
    q_head_to_kv=None,
    scale: float | None = None,
    *,
    num_splits: int = 32,
):
    del q_head_to_kv
    if (
        q.dtype != torch.float16
        or keys_f16.dtype != torch.float16
        or values_f16.dtype != torch.float16
        or buffer_keys_f16.dtype != torch.float16
        or buffer_values_f16.dtype != torch.float16
    ):
        raise TypeError("v2.6 expects fp16 q/keys/values/buffer")
    if parent_alive_bitmap.dtype != torch.int32:
        raise TypeError("v2.6 expects int32 parent_alive_bitmap")
    if assigns_packed.dtype != torch.int64:
        raise TypeError("v2.6 expects int64 assigns_packed")

    h_q, d = q.shape
    h_kv, n_pad, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    b_max = int(buffer_keys_f16.shape[1])
    if d != 128 or d_k != 128 or d_v != 128:
        raise ValueError("v2.6 specializes D=Dv=128")
    if h_q % h_kv != 0:
        raise ValueError("v2.6 expects grouped GQA")
    if not (0 <= int(l_buf) <= b_max):
        raise ValueError(f"l_buf={l_buf} out of range [0, {b_max}]")
    if not (0 <= int(N_used) <= n_pad):
        raise ValueError(f"N_used={N_used} out of range [0, {n_pad}]")
    if scale is None:
        scale = d ** -0.5

    splits = int(num_splits)
    key = (q.device.index, h_q, h_kv, n_pad, b_max, d, d_v, splits)
    ws = _CACHE.get(key)
    if ws is None:
        ws = {
            "partial_m": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_l": torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_o": torch.empty(h_q, splits, d_v, device=q.device, dtype=torch.float32),
            "counters":  torch.zeros(h_q, device=q.device, dtype=torch.int32),
            "out":       torch.empty(h_q, d_v, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    ext = _load_ext()
    return ext.forward(
        q.contiguous(), keys_f16.contiguous(), values_f16.contiguous(),
        buffer_keys_f16.contiguous(), buffer_values_f16.contiguous(),
        parent_alive_bitmap.contiguous(), assigns_packed.contiguous(),
        ws["partial_m"], ws["partial_l"], ws["partial_o"],
        ws["counters"], ws["out"],
        float(scale), int(N_used), int(l_buf), splits,
    )
