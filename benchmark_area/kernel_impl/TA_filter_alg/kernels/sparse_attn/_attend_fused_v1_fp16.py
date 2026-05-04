"""attend_fused_v1 — single-launch fused filter+sparse_attn coop kernel."""
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
        name="ta_attend_fused_v1",
        sources=[
            str(base / "attend_fused_v1.cpp"),
            str(base / "attend_fused_v1_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def attend_fused_v1_fp16(
    q: torch.Tensor,
    centers_padded_f16: torch.Tensor,    # (S=4, Hkv, K, max_w)
    dim_offsets: torch.Tensor,
    dim_widths: torch.Tensor,
    q_head_to_kv: torch.Tensor | None,
    threshold: torch.Tensor,             # (Hq,) fp32
    keys_f16: torch.Tensor,              # (Hkv, Npad, 128)
    values_f16: torch.Tensor,
    buffer_keys_f16: torch.Tensor,       # (Hkv, Bmax, 128)
    buffer_values_f16: torch.Tensor,
    assigns_packed: torch.Tensor,        # (Hkv, Npad) int64
    N_used: int,
    l_buf: int,
    scale: float | None = None,
    *,
    num_splits: int = 32,
):
    if q.dtype != torch.float16:
        raise TypeError("attend_fused_v1 expects q fp16")
    if threshold.dtype != torch.float32:
        threshold = threshold.float()

    h_q, d = q.shape
    h_kv = int(keys_f16.shape[0])
    n_pad = int(keys_f16.shape[1])
    b_max = int(buffer_keys_f16.shape[1])
    k_clusters = int(centers_padded_f16.shape[2])
    k_words = (k_clusters + 31) // 32
    if d != 128:
        raise ValueError("attend_fused_v1 specializes D=128")
    if scale is None:
        scale = d ** -0.5

    splits = int(num_splits)
    key = (q.device.index, h_q, h_kv, n_pad, b_max, k_clusters, k_words, splits)
    ws = _CACHE.get(key)
    if ws is None:
        ws = {
            "top_scores":  torch.empty(h_q, 4, 256, device=q.device, dtype=torch.float32),
            "top_indices": torch.empty(h_q, 4, 256, device=q.device, dtype=torch.int32),
            "depth":       torch.empty(h_q, device=q.device, dtype=torch.int32),
            "parent_alive_bitmap": torch.zeros(h_q, 4, k_words, device=q.device, dtype=torch.int32),
            "partial_m":   torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_l":   torch.empty(h_q, splits, device=q.device, dtype=torch.float32),
            "partial_o":   torch.empty(h_q, splits, d, device=q.device, dtype=torch.float32),
            "counters":    torch.zeros(h_q, device=q.device, dtype=torch.int32),
            "out":         torch.empty(h_q, d, device=q.device, dtype=torch.float16),
        }
        _CACHE.clear()
        _CACHE[key] = ws

    if q_head_to_kv is None:
        q_head_to_kv_t = torch.empty(0, device=q.device, dtype=torch.long)
    else:
        q_head_to_kv_t = q_head_to_kv.contiguous()

    ext = _load_ext()
    ext.forward(
        q.contiguous(),
        centers_padded_f16.contiguous(),
        dim_offsets.contiguous(),
        dim_widths.contiguous(),
        q_head_to_kv_t,
        threshold.contiguous(),
        keys_f16.contiguous(), values_f16.contiguous(),
        buffer_keys_f16.contiguous(), buffer_values_f16.contiguous(),
        assigns_packed.contiguous(),
        ws["top_scores"], ws["top_indices"], ws["depth"], ws["parent_alive_bitmap"],
        ws["partial_m"], ws["partial_l"], ws["partial_o"],
        ws["counters"], ws["out"],
        float(scale), int(N_used), int(l_buf), splits,
    )
    return ws["out"].clone() if False else ws["out"]
