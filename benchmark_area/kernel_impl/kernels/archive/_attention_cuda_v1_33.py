"""JIT loader for the v1.33 CUDA sparse attention index kernel."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_EXT = None
_EXT_ERROR: Exception | None = None
_SRC = Path(__file__).with_name("_cuda_attn_v1_33.cu")


def _keys_blocks_cuda(layout: dict) -> torch.Tensor:
    keys_cuda = layout.get("_keys_blocks_f16_cuda")
    src = layout["keys_blocks_t_f16"]
    key = (src.data_ptr(), tuple(src.shape))
    if keys_cuda is not None and layout.get("_keys_blocks_f16_cuda_key") == key:
        return keys_cuda
    keys_cuda = src.permute(0, 1, 3, 2).contiguous()
    layout["_keys_blocks_f16_cuda"] = keys_cuda
    layout["_keys_blocks_f16_cuda_key"] = key
    return keys_cuda


def _anchor_pack(layout: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    anchor_s = int(layout["anchor_subspace"])
    key = (
        int(layout["centers"].data_ptr()),
        int(layout["radii"].data_ptr()),
        anchor_s,
    )
    cached = layout.get("_attn_v1_33_anchor_pack")
    if cached is not None and cached["key"] == key:
        return cached["centers"], cached["radii"], cached["width"]

    centers = layout["centers"][anchor_s].contiguous()
    radii = layout["radii"][anchor_s].contiguous()
    width = int(layout["dim_widths"][anchor_s].item())
    layout["_attn_v1_33_anchor_pack"] = {
        "key": key,
        "centers": centers,
        "radii": radii,
        "width": width,
    }
    return centers, radii, width


def _ensure_ninja_on_path() -> None:
    bin_dir = str(Path(sys.executable).resolve().parent)
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path


def load_ext():
    global _EXT, _EXT_ERROR
    if _EXT is not None:
        return _EXT
    if _EXT_ERROR is not None:
        raise _EXT_ERROR

    from torch.utils.cpp_extension import load

    _ensure_ninja_on_path()
    try:
        _EXT = load(
            name="hira_cuda_attn_v1_33",
            sources=[str(_SRC)],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        )
        return _EXT
    except Exception as exc:  # pragma: no cover - build/runtime dependent
        _EXT_ERROR = exc
        raise


def is_supported(
    q: torch.Tensor,
    layout: dict,
) -> bool:
    _, _, width = _anchor_pack(layout)
    return (
        q.is_cuda
        and q.dtype == torch.float16
        and q.shape[1] == 128
        and int(layout["D_v"]) == 128
        and int(layout["bf"]) == 4
        and int(layout["num_subspaces"]) == 8
        and int(layout["groups"]) <= 8
        and width == 16
    )


def run_fused_attn_index_cuda_v1_33(
    q: torch.Tensor,
    q_norm_anchor: torch.Tensor,
    th_anchor: torch.Tensor,
    layout: dict,
    h_q: int,
    groups: int,
    num_splits: int,
    scale: float,
    out_m: torch.Tensor,
    out_l: torch.Tensor,
    out_o: torch.Tensor,
) -> None:
    mod = load_ext()
    centers, radii, _ = _anchor_pack(layout)
    mod.sparse_attn_index_v1_33(
        q,
        q_norm_anchor,
        th_anchor,
        centers,
        radii,
        _keys_blocks_cuda(layout),
        layout["values_blocks_f16"],
        layout["invalid_blocks_i8"],
        out_m,
        out_l,
        out_o,
        h_q,
        int(layout["base_heads"]),
        int(layout["K"]),
        num_splits,
        groups,
        int(layout["dim_offsets"][int(layout["anchor_subspace"])].item()),
        float(scale),
    )
