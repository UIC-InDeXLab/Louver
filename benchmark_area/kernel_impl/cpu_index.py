"""CPU end-to-end wrapper around the best HIRA CPU kernels.

Mirrors `kernel_impl/index.py` but specialized for the CPU C++/AVX-512 path:
build_v1.0 + update_v1.0 + attention_v4.3 (full-AND GQA bitmask gating).
Update is synchronous — no side-stream / parallel path.

Decoding loop usage:
    idx = SubspaceKCenterIndexCPU(IndexConfigCPU())
    idx.build(prefill_keys_fp32, prefill_values_fp32)
    for step in range(n_decode):
        # Outside the timed region:
        th = th_per_subspace_fp32.contiguous()
        # Inside the timed region:
        out = idx.attend(q_fp32, th, q_head_to_kv=...)
        # Append + maybe update (NOT timed):
        idx.append_decoding_kv(new_k_fp32, new_v_fp32)
        if idx.needs_update(update_interval):
            idx.update()

Design notes — attend() does not need per-step dtype preparation:
    * fp32 K/V history lives in self.keys / self.values (used by update only)
    * fp32 buffer lives in self.buffer / self.values_buffer (consumed by update)
    * v4.3 consumes fp32 q, thresholds, index state, and buffer tensors directly

Net effect: attend() runs only the v4.3 kernel call — same cost the
kernel_bench micro-bench measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .kernels.cpu_kernels.attention_v4_3 import attend as _attend_v4_3
from .kernels.cpu_kernels.build_v1_0 import build as _build_v1_0
from .kernels.cpu_kernels.update_v1_0 import update as _update_v1_0


@dataclass
class IndexConfigCPU:
    n_subspaces: int = 8
    bf: int = 4
    refine_iter: int = 5
    update_mode: str = "inc"   # "full" | "inc"
    update_refine_iter: int = 0


def _empty_like_dim1(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Empty (H, 0, D) tensor matching `t`'s H/D and (optionally) dtype."""
    return torch.empty(
        t.shape[0], 0, t.shape[2],
        dtype=dtype if dtype is not None else t.dtype,
    )


class SubspaceKCenterIndexCPU:
    """CPU-only, single-stream index. Holds fp32 K/V and decode buffer."""

    def __init__(self, cfg: IndexConfigCPU):
        self.cfg = cfg
        self.state: dict | None = None
        self.keys: torch.Tensor | None = None              # (H_kv, N, D)  fp32
        self.values: torch.Tensor | None = None            # (H_kv, N, D)  fp32
        self.buffer: torch.Tensor | None = None            # (H_kv, B, D)  fp32
        self.values_buffer: torch.Tensor | None = None     # (H_kv, B, D)  fp32
        self._steps_since_update = 0

    # ── Build ─────────────────────────────────────────────────────────

    def build(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "SubspaceKCenterIndexCPU":
        self.keys = keys.contiguous()
        self.values = values.contiguous()
        self.state = _build_v1_0(
            keys=self.keys,
            bf=self.cfg.bf,
            n_subspaces=self.cfg.n_subspaces,
            refine_iter=self.cfg.refine_iter,
            values=self.values,
        )
        self.buffer = _empty_like_dim1(self.keys)
        self.values_buffer = _empty_like_dim1(self.values)
        self._steps_since_update = 0
        return self

    # ── Decoding-time key/value handling ──────────────────────────────

    def append_decoding_kv(
        self, new_key: torch.Tensor, new_value: torch.Tensor
    ) -> None:
        """Append one (k, v) pair to the fp32 buffer."""
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        if new_value.dim() == 2:
            new_value = new_value.unsqueeze(1)
        self.buffer = torch.cat([self.buffer, new_key], dim=1)
        self.values_buffer = torch.cat([self.values_buffer, new_value], dim=1)
        self._steps_since_update += 1

    def needs_update(self, update_interval: int) -> bool:
        return self._steps_since_update >= update_interval

    # ── Synchronous update ────────────────────────────────────────────

    def update(self) -> None:
        if self.buffer.shape[1] == 0:
            self._steps_since_update = 0
            return

        if self.cfg.update_mode == "full":
            new_keys = torch.cat([self.keys, self.buffer], dim=1).contiguous()
            new_values = torch.cat(
                [self.values, self.values_buffer], dim=1
            ).contiguous()
            self.state = _build_v1_0(
                keys=new_keys,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                values=new_values,
            )
            self.keys = new_keys
            self.values = new_values
        elif self.cfg.update_mode == "inc":
            new_state, new_keys, new_values, _ = _update_v1_0(
                state=self.state,
                old_keys=self.keys,
                buffer_keys=self.buffer,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.update_refine_iter,
                old_values=self.values,
                buffer_values=self.values_buffer,
                return_merged=True,
            )
            self.state = new_state
            if new_keys is not None:
                self.keys = new_keys
            if new_values is not None:
                self.values = new_values
        else:
            raise ValueError(f"Unknown update_mode: {self.cfg.update_mode!r}")

        self.buffer = _empty_like_dim1(self.keys)
        self.values_buffer = _empty_like_dim1(self.values)
        self._steps_since_update = 0

    # ── Threshold preparation compatibility helper ───────────────────

    def pack_threshold(
        self,
        q: torch.Tensor,                 # (H_q, D) fp32
        th_per_subspace: torch.Tensor,   # (S, H_q) fp32
    ) -> torch.Tensor:
        """Return thresholds in the fp32 layout v4.3 consumes.

        Kept as a compatibility shim for callers that still route threshold
        preparation through the index wrapper.
        """
        return th_per_subspace.contiguous().to(torch.float32)

    # ── Attention (v4.3, full-AND bitmask) — pure kernel call ────────

    def attend(
        self,
        q: torch.Tensor,                # (H_q, D) fp32
        th_per_subspace: torch.Tensor,  # (S, H_q) fp32
        q_head_to_kv: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Return (H_q, D_v) fp32 attention output.

        Inputs should already be contiguous fp32 tensors. Buffer keys/values
        are sourced from the fp32 buffer maintained by append_decoding_kv().
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        buf_keys = self.buffer if self.buffer.shape[1] > 0 else None
        buf_values = (
            self.values_buffer if self.values_buffer.shape[1] > 0 else None
        )
        return _attend_v4_3(
            q, th_per_subspace, self.state,
            buffer_keys=buf_keys,
            buffer_values=buf_values,
            keys_children=self.keys,
            q_head_to_kv=q_head_to_kv,
            scale=float(scale),
        )

    # ── Introspection ─────────────────────────────────────────────────

    @property
    def n_children(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    @property
    def n_buffered(self) -> int:
        return 0 if self.buffer is None else int(self.buffer.shape[1])

    def memory_bytes(self) -> int:
        total = 0
        for t in (
            self.keys, self.values, self.buffer, self.values_buffer,
        ):
            if t is not None:
                total += t.element_size() * t.numel()
        if self.state is not None:
            for v in self.state.values():
                if isinstance(v, torch.Tensor):
                    total += v.element_size() * v.numel()
                elif isinstance(v, list):
                    for t in v:
                        if isinstance(t, torch.Tensor):
                            total += t.element_size() * t.numel()
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.Tensor):
                            total += vv.element_size() * vv.numel()
                        elif isinstance(vv, list):
                            for t in vv:
                                if isinstance(t, torch.Tensor):
                                    total += t.element_size() * t.numel()
        return total


# ── CPU baselines (fp32 / SDPA) ───────────────────────────────────────

def baseline_attention(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Brute-force dense attention with native GQA (no H_q expansion)."""
    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)

    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return torch.einsum("hn,hnd->hd", probs, values)

    if h_q % h_kv != 0:
        raise ValueError(f"H_q={h_q} must be a multiple of H_kv={h_kv}")
    groups = h_q // h_kv
    q_hg = q.view(h_kv, groups, d)
    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    out = torch.einsum("hgn,hnd->hgd", probs, values)
    return out.reshape(h_q, d_v)


def baseline_sdpa(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """torch.nn.functional.scaled_dot_product_attention with native GQA.

    Dispatches to oneDNN's fused single-query attention on x86 CPU. Honors
    the dtype of the inputs (so callers can pass fp32 or bf16 K/V).
    """
    h_q, d = q.shape
    h_kv, n, _ = keys.shape
    d_v = values.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)

    q4 = q.view(1, h_q, 1, d)
    k4 = keys.view(1, h_kv, n, d)
    v4 = values.view(1, h_kv, n, d_v)
    out = torch.nn.functional.scaled_dot_product_attention(
        q4, k4, v4, is_causal=False, scale=scale,
        enable_gqa=(h_q != h_kv),
    )
    return out.view(h_q, d_v)
