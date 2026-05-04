"""ta_filter v10.0 — non-cooperative two-kernel filter (score | depth+bitmap).

Same outputs as v9.0 (top_scores, top_indices, depth, parent_alive_bitmap),
but launched as two non-cooperative kernels in stream order instead of one
coop kernel with grid_sync. Drops the coop launch overhead at the cost of
one extra kernel-launch latency.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

KERNEL_VERSION = "v10.0"

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
        name="ta_filter_v10_0",
        sources=[
            str(base / "ta_filter_v_10_0.cpp"),
            str(base / "ta_filter_cuda_kernel_v_10_0.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def _workspace(*, device, h_q, k_words):
    key = (device.index, h_q, k_words)
    ws = _CACHE.get(key)
    if ws is not None:
        return ws
    ws = {
        "top_scores":  torch.empty(h_q, 4, 256, device=device, dtype=torch.float32),
        "top_indices": torch.empty(h_q, 4, 256, device=device, dtype=torch.int32),
        "depth":       torch.empty(h_q, device=device, dtype=torch.int32),
        "parent_alive_bitmap": torch.zeros(
            h_q, 4, k_words, device=device, dtype=torch.int32
        ),
    }
    _CACHE.clear()
    _CACHE[key] = ws
    return ws


def ta_filter_v10_0(q, threshold, state, q_head_to_kv=None, *, ws=None):
    if q.dtype != torch.float16:
        raise TypeError("ta_filter_v10_0 expects q fp16")
    if threshold.dtype != torch.float32:
        threshold = threshold.float()
    q = q.contiguous()
    threshold = threshold.contiguous()

    centers = state["centers_padded_f16"].contiguous()
    dim_offsets = state["dim_offsets"].contiguous()
    dim_widths = state["dim_widths"].contiguous()
    h_q = int(q.shape[0])
    k_clusters = int(centers.shape[2])
    k_words = (k_clusters + 31) // 32

    if int(centers.shape[0]) != 4 or int(state["bf"]) != 4:
        raise ValueError("ta_filter_v10_0 requires S=4, bf=4")

    if q_head_to_kv is None:
        q_head_to_kv_t = torch.empty(0, device=q.device, dtype=torch.long)
    else:
        q_head_to_kv_t = q_head_to_kv.contiguous()

    if ws is None:
        ws = _workspace(device=q.device, h_q=h_q, k_words=k_words)

    ext = _load_ext()
    ext.filter(
        q, centers, dim_offsets, dim_widths, q_head_to_kv_t, threshold,
        ws["top_scores"], ws["top_indices"], ws["depth"],
        ws["parent_alive_bitmap"],
        int(k_clusters),
    )
    return ws


KERNEL = ta_filter_v10_0
