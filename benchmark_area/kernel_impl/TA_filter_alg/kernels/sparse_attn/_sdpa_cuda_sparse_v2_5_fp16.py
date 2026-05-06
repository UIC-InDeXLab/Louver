"""v2.5 sparse decode-SDPA: v2.4 + buffer-aware tail loop.

Accepts a pair of buffer tensors ``(buffer_keys, buffer_values)`` of shape
``(H_kv, B_max, 128)`` and an ``l_buf`` count.  The kernel processes
``live_count[hq] + l_buf`` keys per head — the first ``live_count[hq]``
indexed via ``live_idx``, the trailing ``l_buf`` fetched directly from the
buffer slots ``[0:l_buf]``.

Eliminates the live_idx → live_idx_attn copy + per-head buffer scatter
that Variant B performs in Python before sparse_attn v2.4.
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
        name="ta_sdpa_cuda_sparse_v2_5_fp16",
        sources=[
            str(base / "sdpa_cuda_sparse_v2_5.cpp"),
            str(base / "sdpa_cuda_sparse_v2_5_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def sdpa_cuda_sparse_v2_5_fp16(
    q: torch.Tensor,
    keys_f16: torch.Tensor,
    values_f16: torch.Tensor,
    buffer_keys_f16: torch.Tensor,    # (H_kv, B_max, 128)
    buffer_values_f16: torch.Tensor,  # (H_kv, B_max, 128)
    live_idx: torch.Tensor,
    live_count: torch.Tensor,
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
        raise TypeError("v2.5 expects fp16 q/keys/values/buffer")
    if live_idx.dtype != torch.int32 or live_count.dtype != torch.int32:
        raise TypeError("v2.5 expects int32 live_idx and live_count")

    h_q, d = q.shape
    h_kv, n_pad, d_k = keys_f16.shape
    d_v = int(values_f16.shape[-1])
    b_max = int(buffer_keys_f16.shape[1])
    if d != 128 or d_k != 128 or d_v != 128:
        raise ValueError("v2.5 specializes D=Dv=128")
    if h_q % h_kv != 0:
        raise ValueError("v2.5 expects grouped GQA")
    if live_idx.shape != (h_q, n_pad):
        raise ValueError(
            f"v2.5 live_idx must be (Hq={h_q}, Npad={n_pad}); "
            f"got {tuple(live_idx.shape)}"
        )
    if live_count.shape != (h_q,):
        raise ValueError(
            f"v2.5 live_count must be (Hq={h_q},); got {tuple(live_count.shape)}"
        )
    if buffer_values_f16.shape != (h_kv, b_max, d_v):
        raise ValueError(
            f"v2.5 buffer_values must be (Hkv={h_kv}, B_max={b_max}, Dv={d_v}); "
            f"got {tuple(buffer_values_f16.shape)}"
        )
    if not (0 <= int(l_buf) <= b_max):
        raise ValueError(f"l_buf={l_buf} out of range [0, {b_max}]")
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
        live_idx.contiguous(), live_count.contiguous(),
        ws["partial_m"], ws["partial_l"], ws["partial_o"],
        ws["counters"], ws["out"],
        float(scale), int(l_buf), splits,
    )
