"""update_v1.1 — fast .cu cluster path for the TA-filter incremental update.

Replaces ``update_v1_0``'s host-recursive balanced PCA tree with a single
.cu kernel that:

    1. Per (s, h_kv): projects each of the 256 buffer keys onto a single
       per-subspace axis (sum-of-w-dims), CUB-sorts, and groups into 64
       clusters of 4 by sorted rank.
    2. Writes new centers into ``centers_padded_f16[s, h, K_used:K_used+64]``.
    3. Writes new assigns into ``assigns_padded[s, h, N_used:N_used+256]``.

Buffer key/value scatter into the keys/values arena (single .copy_() each)
and packed-assigns construction stay as torch ops since they're already
GPU-bound and dwarfed by the kernel.

The phase split (data-only writes vs. publish) is preserved so the call
site can launch this on a side stream and overlap with attention; the
publish step (``apply_publish``) flips invalid_mask + writes
_assigns_packed_u64_v34 + bumps K_used/N_used on the attention stream
after waiting on the update event.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

KERNEL_VERSION = "v1.1"

BF = 4
S = 4
BUFFER_SIZE = 256
K_BUF = BUFFER_SIZE // BF


_EXT = None


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
        name="ta_update_v1_1",
        sources=[
            str(base / "update_v1_1.cpp"),
            str(base / "update_v1_1_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


def update_v1_1(
    state: dict[str, Any],
    buffer_keys: torch.Tensor,    # (H_kv, 256, D) fp16
    buffer_values: torch.Tensor,  # (H_kv, 256, D_v) fp16
) -> dict[str, Any]:
    """Phase 1 — write cluster + buffer data into arena tail.

    Returns the publish dict to be applied via ``apply_publish``.
    """
    if buffer_keys.shape[1] != BUFFER_SIZE:
        raise ValueError(
            f"update_v1_1 requires exactly {BUFFER_SIZE} buffer keys; "
            f"got {buffer_keys.shape[1]}"
        )

    h_kv = int(state["centers_padded_f16"].shape[1])
    k_used = int(state["K_used"])
    n_used = int(state["N_used"])
    k_cap = int(state["K_cap"])
    n_cap = int(state["N_pad"])

    if k_used + K_BUF > k_cap:
        raise RuntimeError(
            f"arena full: K_used={k_used} + K_BUF={K_BUF} > K_cap={k_cap}"
        )
    if n_used + BUFFER_SIZE > n_cap:
        raise RuntimeError(
            f"arena full: N_used={n_used} + B={BUFFER_SIZE} > N_pad={n_cap}"
        )

    centers_arena = state["centers_padded_f16"]
    assigns_arena = state["assigns_padded"]
    keys_arena = state["keys_padded_f16"]
    values_arena = state["values_padded_f16"]

    # ── 1. .cu cluster kernel — writes centers + assigns in one launch. ──
    ext = _load_ext()
    ext.cluster(
        buffer_keys.contiguous(),
        state["dim_offsets"].contiguous(),
        state["dim_widths"].contiguous(),
        centers_arena,
        assigns_arena,
        k_used,
        n_used,
    )

    # ── 2. Buffer key/value scatter into arena tail. ──
    keys_arena[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_keys)
    values_arena[:, n_used:n_used + BUFFER_SIZE, :].copy_(buffer_values)

    # ── 3. Build packed assigns row to be committed at publish.  Each row of
    #       packed combines the 4 subspace cluster ids of one key. ──
    a_slice = assigns_arena[:, :, n_used:n_used + BUFFER_SIZE].to(torch.int64) & 0xFFFF
    packed_buf = (
        a_slice[0]
        | (a_slice[1] << 16)
        | (a_slice[2] << 32)
        | (a_slice[3] << 48)
    ).contiguous()  # (H_kv, B)

    pending = {
        "state": state,
        "k_used_after": k_used + K_BUF,
        "n_used_after": n_used + BUFFER_SIZE,
        "n_used_before": n_used,
        "n_added": BUFFER_SIZE,
        "packed_buf": packed_buf,
    }
    return pending


def apply_publish(pending: dict[str, Any]) -> None:
    """Phase 2 — flip invalid flags / publish packed assigns / bump counters."""
    state = pending["state"]
    n0 = pending["n_used_before"]
    n_add = pending["n_added"]
    state["invalid_mask"][:, n0:n0 + n_add] = False
    state["_assigns_packed_u64_v34"][:, n0:n0 + n_add] = pending["packed_buf"]
    state["K_used"] = pending["k_used_after"]
    state["N_used"] = pending["n_used_after"]


KERNEL = update_v1_1
