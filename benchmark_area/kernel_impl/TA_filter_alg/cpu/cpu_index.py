"""CPU TA-filter index — fused attend + sync update over a 256-key buffer.

Mirrors the CUDA ``TAIndex`` decoding loop:
    1. ``attend(q, threshold)`` — fused filter + sparse-attn over arena ∪ buffer.
    2. ``append_decoding_kv(k, v)`` — appends to the buffer.
    3. Every BUFFER_SIZE=256 steps: ``update()`` clusters the buffer into 64 new
       parents and appends them to the arena tail.

Hard-coded: S=4, bf=4, BUFFER_SIZE=256.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .build import BF, BUFFER_SIZE, K_BUF, S_FIXED, build_state, expand_arena
from .attend.attend_v1 import attend as attend_v1
from .attend.attend_v2 import attend as attend_v2
from .attend.attend_v3 import attend as attend_v3
from .update.update_v1 import update as update_v1


_ATTEND_FNS = {"v1": attend_v1, "v2": attend_v2, "v3": attend_v3}


@dataclass
class TAIndexCPUConfig:
    n_growth: int = 4096
    refine_iter: int = 5
    attend_version: str = "v3"


class TAIndexCPU:
    """CPU TA-filter index."""

    def __init__(self, cfg: TAIndexCPUConfig | None = None):
        self.cfg = cfg or TAIndexCPUConfig()
        self.state: dict[str, Any] | None = None
        self._buf_keys: torch.Tensor | None = None
        self._buf_values: torch.Tensor | None = None
        self._l_buf: int = 0
        self._steps_since_update = 0

    def build(self, keys: torch.Tensor, values: torch.Tensor) -> "TAIndexCPU":
        if keys.device.type != "cpu":
            raise ValueError("CPU index requires CPU tensors")
        keys = keys.float().contiguous()
        values = values.float().contiguous()
        h_kv, n_init, d = keys.shape
        d_v = int(values.shape[-1])

        state = build_state(keys, values, refine_iter=self.cfg.refine_iter)
        n_cap = int(math.ceil((n_init + self.cfg.n_growth) / BF) * BF)
        k_cap = n_cap // BF
        expand_arena(state, k_cap=k_cap, n_cap=n_cap)

        self._buf_keys = torch.zeros(h_kv, BUFFER_SIZE, d, dtype=torch.float32)
        self._buf_values = torch.zeros(h_kv, BUFFER_SIZE, d_v, dtype=torch.float32)

        self.state = state
        self._l_buf = 0
        self._steps_since_update = 0
        return self

    def append_decoding_kv(self, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        if new_value.dim() == 2:
            new_value = new_value.unsqueeze(1)
        if self._l_buf >= BUFFER_SIZE:
            raise RuntimeError(
                f"buffer overflow: l_buf={self._l_buf} >= {BUFFER_SIZE}"
            )
        self._buf_keys[:, self._l_buf:self._l_buf + 1, :].copy_(new_key.float())
        self._buf_values[:, self._l_buf:self._l_buf + 1, :].copy_(new_value.float())
        self._l_buf += 1
        self._steps_since_update += 1

    def needs_update(self, interval: int = BUFFER_SIZE) -> bool:
        return self._steps_since_update >= interval

    def update(self) -> None:
        if self._l_buf == 0:
            self._steps_since_update = 0
            return
        if self._l_buf != BUFFER_SIZE:
            raise RuntimeError(
                f"update() requires full {BUFFER_SIZE}-key buffer; got {self._l_buf}"
            )
        # Auto-expand arena if needed (each update adds BUFFER_SIZE/BF new parents)
        new_parents = BUFFER_SIZE // BF
        k_used = int(self.state["K_used"])
        k_cap  = int(self.state["K_cap"])
        if k_used + new_parents > k_cap:
            grow_k = max(new_parents * 64, self.cfg.n_growth // BF)
            new_k_cap = k_cap + grow_k
            new_n_cap = new_k_cap * BF
            expand_arena(self.state, k_cap=new_k_cap, n_cap=new_n_cap)
        update_v1(self.state, self._buf_keys, self._buf_values)
        self._buf_keys.zero_()
        self._buf_values.zero_()
        self._l_buf = 0
        self._steps_since_update = 0

    def attend(
        self,
        q: torch.Tensor,
        threshold: torch.Tensor,
        q_head_to_kv: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("call build() first")
        if scale is None:
            scale = q.shape[-1] ** -0.5
        fn = _ATTEND_FNS.get(self.cfg.attend_version, attend_v2)
        return fn(
            q.float(),
            threshold.float(),
            self.state,
            self._buf_keys,
            self._buf_values,
            self._l_buf,
            q_head_to_kv=q_head_to_kv,
            scale=float(scale),
        )

    @property
    def n_indexed(self) -> int:
        return int(self.state["N_used"]) if self.state is not None else 0

    @property
    def n_buffered(self) -> int:
        return self._l_buf


def baseline_dense(q, keys, values, q_head_to_kv=None, scale=None):
    h_q, d = q.shape
    h_kv = keys.shape[0]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return torch.einsum("hn,hnd->hd", probs, values)
    g = h_q // h_kv
    q_g = q.view(h_kv, g, d)
    scores = torch.einsum("hgd,hnd->hgn", q_g, keys) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    return torch.einsum("hgn,hnd->hgd", probs, values).reshape(h_q, values.shape[-1])


def baseline_sdpa(q, keys, values, q_head_to_kv=None, scale=None):
    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    q4 = q.view(1, h_q, 1, d)
    k4 = keys.view(1, h_kv, n, d)
    v4 = values.view(1, h_kv, n, d_v)
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, scale=scale, enable_gqa=(h_q != h_kv),
    )
    return out.view(h_q, d_v)
