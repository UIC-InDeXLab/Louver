"""TA-filter index — fused build + filter + sparse-attn pipeline.

Hard-coded: ``S=4``, ``bf=4``, ``BUFFER_SIZE=256``.

Each decode step:
    1. ``attend(q, threshold)`` — runs ``ta_filter_v8`` on the index, then
       ``sdpa_cuda_sparse_v2_5_fp16`` (buffer-aware) over (filter survivors
       ∪ active buffer) in a single launch.
    2. ``append_decoding_kv(k, v)`` — appends to the buffer.
    3. Every ``BUFFER_SIZE`` steps: ``update()`` (sync) or ``update_async()``
       + ``wait_for_update()`` (parallel) — re-clusters the buffer's 256
       keys into 64 new parents and appends them to the arena tail
       (.cu kernel ``update_v1_1``).

Buffer integration: sparse_attn v2.5 takes the active buffer K/V as
separate inputs.  The filter writes ``live_idx_filter (h_q, K_cap·BF)``
directly into v2.5; no intermediate ``live_idx_attn`` copy / buffer
scatter is needed.  Empirically ``attend ≈ filter + sparse_attn`` (see
``kernel_bench/bench_attend_breakdown.py``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from .kernels import TA_build
from .kernels.filtering.ta_filter_v_8_0 import (
    _build_packed_assigns,
    _load_ext as _load_filter_ext,
    GRID_BLOCKS_BOUNDARY,
    TILE_N_V710,
)
from .kernels.sparse_attn._sdpa_cuda_sparse_v2_5_fp16 import (
    sdpa_cuda_sparse_v2_5_fp16,
)
from .kernels.update.update_v1_1 import (
    BF, BUFFER_SIZE, K_BUF, S as S_FIXED,
    apply_publish, update_v1_1,
)


HALF_NEG_LARGE = -65504.0
_FILT_L = 256


@dataclass
class TAIndexConfig:
    n_growth: int = 4096
    refine_iter: int = 5
    parallel_update: bool = False
    update_stream_priority: int = -1


@dataclass
class _UpdateMetrics:
    kernel_ms: float = 0.0
    host_wait_ms: float = 0.0
    inflight_at_fire: bool = False
    fire_step: int = -1


def _expand_arena(state: dict[str, Any], k_cap: int, n_cap: int) -> None:
    """Extend every K-/N-major arena tensor in-place to (K_cap, N_cap)."""

    def _grow_first(t: torch.Tensor, *, dim: int, new: int, fill) -> torch.Tensor:
        old = t.shape[dim]
        if old >= new:
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = new - old
        pad = torch.full(pad_shape, fill, device=t.device, dtype=t.dtype)
        return torch.cat([t, pad], dim=dim).contiguous()

    state["centers_padded_f16"] = _grow_first(
        state["centers_padded_f16"], dim=2, new=k_cap, fill=HALF_NEG_LARGE
    )
    state["assigns_padded"] = _grow_first(
        state["assigns_padded"], dim=2, new=n_cap, fill=0
    )
    state["keys_padded_f16"] = _grow_first(
        state["keys_padded_f16"], dim=1, new=n_cap, fill=0
    )
    if "values_padded_f16" in state:
        state["values_padded_f16"] = _grow_first(
            state["values_padded_f16"], dim=1, new=n_cap, fill=0
        )
    inv = state["invalid_mask"]
    if inv.shape[1] < n_cap:
        pad = torch.ones(
            inv.shape[0], n_cap - inv.shape[1], dtype=torch.bool, device=inv.device
        )
        state["invalid_mask"] = torch.cat([inv, pad], dim=1).contiguous()

    state.pop("_assigns_packed_u64_v34", None)
    _build_packed_assigns(state)

    state["K_cap"] = k_cap
    state["N_pad"] = n_cap
    state["K_used"] = int(state["K"])
    state["N_used"] = int(state["N"])


def _filter_with_workspace(
    *,
    q: torch.Tensor,
    threshold: torch.Tensor,
    state: dict[str, Any],
    q_head_to_kv: torch.Tensor | None,
    live_idx: torch.Tensor,
    live_count: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
    depth: torch.Tensor,
) -> None:
    if q.dtype != torch.float16:
        raise TypeError("filter expects q fp16")
    centers = state["centers_padded_f16"].contiguous()
    dim_offsets = state["dim_offsets"].contiguous()
    dim_widths = state["dim_widths"].contiguous()
    assigns_packed = _build_packed_assigns(state)
    h_q = int(q.shape[0])
    K_stride = int(centers.shape[2])
    K_used = int(state.get("K_used", K_stride))
    k_clusters = min(K_used, K_stride) if K_used > 0 else K_stride
    n_pad_filter = int(state["N_pad"])
    if q_head_to_kv is None:
        q_head_to_kv_t = torch.empty(0, device=q.device, dtype=torch.long)
    else:
        q_head_to_kv_t = q_head_to_kv.contiguous()

    n_tiles = (n_pad_filter + TILE_N_V710 - 1) // TILE_N_V710
    grid_blocks = h_q * max(4, n_tiles)
    tile_n = 4096 if grid_blocks > GRID_BLOCKS_BOUNDARY else 2048

    ext = _load_filter_ext()
    n_used_int = int(state.get("N_used", n_pad_filter))
    ext.fused_pipeline(
        q, centers, dim_offsets, dim_widths, q_head_to_kv_t,
        threshold.float().contiguous(), assigns_packed,
        top_scores, top_indices, depth, live_idx, live_count,
        int(k_clusters), int(K_stride), int(n_used_int), tile_n,
    )


class TAIndex:
    """Subspace TA-filter index with fixed S=4, bf=4, buffer=256."""

    def __init__(self, cfg: TAIndexConfig | None = None):
        self.cfg = cfg or TAIndexConfig()
        self.state: dict[str, Any] | None = None

        # Active-buffer K/V (separate tensors fed directly to sparse_attn v2.5).
        self._buf_keys_arena: torch.Tensor | None = None
        self._buf_values_arena: torch.Tensor | None = None
        self._l_buf: int = 0
        self._steps_since_update = 0

        # Filter workspace (sized lazily on first attend).
        self._ws: dict[str, torch.Tensor] | None = None

        # Async-update plumbing.
        self._update_stream: torch.cuda.Stream | None = None
        self._update_start_event: torch.cuda.Event | None = None
        self._update_done_event: torch.cuda.Event | None = None
        self._pending_update: bool = False
        self._pending_publish: dict | None = None
        self._buffer_inflight_keys: torch.Tensor | None = None
        self._buffer_inflight_values: torch.Tensor | None = None
        self._last_metrics: _UpdateMetrics | None = None
        self.update_metrics_log: list[_UpdateMetrics] = []
        self.n_overlap_misses: int = 0

    # ── Build ──

    def build(self, keys: torch.Tensor, values: torch.Tensor) -> "TAIndex":
        if keys.dtype != torch.float16 or values.dtype != torch.float16:
            raise TypeError("TAIndex requires fp16 keys/values")
        keys = keys.contiguous()
        values = values.contiguous()
        h_kv, n_init, d = keys.shape
        d_v = int(values.shape[-1])

        state = TA_build.build(
            keys=keys, bf=BF, n_subspaces=S_FIXED,
            refine_iter=self.cfg.refine_iter, values=values,
        )

        n_cap = int(math.ceil((n_init + self.cfg.n_growth) / BF) * BF)
        k_cap = n_cap // BF
        _expand_arena(state, k_cap, n_cap)

        device = keys.device
        self._buf_keys_arena = torch.zeros(
            h_kv, BUFFER_SIZE, d, device=device, dtype=torch.float16
        )
        self._buf_values_arena = torch.zeros(
            h_kv, BUFFER_SIZE, d_v, device=device, dtype=torch.float16
        )

        self.state = state
        self._l_buf = 0
        self._steps_since_update = 0

        if self.cfg.parallel_update:
            self._update_stream = torch.cuda.Stream(
                priority=self.cfg.update_stream_priority
            )
        return self

    # ── Decoding-time ──

    def append_decoding_kv(self, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        if new_value.dim() == 2:
            new_value = new_value.unsqueeze(1)
        if self._l_buf >= BUFFER_SIZE:
            raise RuntimeError(
                f"buffer overflow: l_buf={self._l_buf} >= {BUFFER_SIZE}; "
                "call update()/update_async() before appending more keys"
            )
        self._buf_keys_arena[:, self._l_buf:self._l_buf + 1, :].copy_(new_key)
        self._buf_values_arena[:, self._l_buf:self._l_buf + 1, :].copy_(new_value)
        self._l_buf += 1
        self._steps_since_update += 1

    def needs_update(self, update_interval: int = BUFFER_SIZE) -> bool:
        return self._steps_since_update >= update_interval

    # ── Sync update ──

    def update(self) -> None:
        if self._l_buf == 0:
            self._steps_since_update = 0
            return
        if self._l_buf != BUFFER_SIZE:
            raise RuntimeError(
                f"update() requires a full {BUFFER_SIZE}-key buffer; got l_buf={self._l_buf}"
            )
        buf_keys = self._buf_keys_arena.contiguous()
        buf_values = self._buf_values_arena.contiguous()
        pending = update_v1_1(self.state, buf_keys, buf_values)
        apply_publish(pending)
        self._reset_buffer_after_publish()

    # ── Async update ──

    def update_async(self, fire_step: int = -1) -> None:
        if self._l_buf == 0:
            self._steps_since_update = 0
            return
        if self._l_buf != BUFFER_SIZE:
            raise RuntimeError(
                f"update_async requires a full {BUFFER_SIZE}-key buffer; got l_buf={self._l_buf}"
            )
        if self._update_stream is None:
            raise RuntimeError("update_async requires cfg.parallel_update=True")

        attn_stream = torch.cuda.current_stream()
        metrics = _UpdateMetrics(fire_step=fire_step)

        if self._pending_update:
            metrics.inflight_at_fire = True
            self.n_overlap_misses += 1
            self._wait_and_publish(attn_stream)

        self._buffer_inflight_keys = self._buf_keys_arena.detach().clone()
        self._buffer_inflight_values = self._buf_values_arena.detach().clone()
        self._buffer_inflight_keys.record_stream(self._update_stream)
        self._buffer_inflight_values.record_stream(self._update_stream)

        self._buf_keys_arena.zero_()
        self._buf_values_arena.zero_()

        self._update_stream.wait_stream(attn_stream)

        if self._update_start_event is None:
            self._update_start_event = torch.cuda.Event(enable_timing=True)
            self._update_done_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self._update_stream):
            self._update_start_event.record(self._update_stream)
            pending = update_v1_1(
                self.state,
                self._buffer_inflight_keys,
                self._buffer_inflight_values,
            )
            self._update_done_event.record(self._update_stream)

        self._pending_publish = pending
        self._pending_update = True
        self._last_metrics = metrics
        self._l_buf = 0
        self._steps_since_update = 0

    def wait_for_update(self) -> None:
        if not self._pending_update:
            return
        attn_stream = torch.cuda.current_stream()
        self._wait_and_publish(attn_stream)

    def try_publish(self) -> bool:
        if not self._pending_update:
            return False
        if self._update_done_event is None:
            return False
        if not self._update_done_event.query():
            return False
        attn_stream = torch.cuda.current_stream()
        self._wait_and_publish(attn_stream)
        return True

    def _wait_and_publish(self, attn_stream: torch.cuda.Stream) -> None:
        metrics = self._last_metrics or _UpdateMetrics()
        already_done = self._update_done_event.query()
        host_wait_start = time.perf_counter() if not already_done else None

        attn_stream.wait_event(self._update_done_event)
        if self._pending_publish is not None:
            apply_publish(self._pending_publish)

        if host_wait_start is not None:
            metrics.host_wait_ms = (time.perf_counter() - host_wait_start) * 1000.0
        try:
            metrics.kernel_ms = self._update_start_event.elapsed_time(
                self._update_done_event
            )
        except Exception:
            metrics.kernel_ms = 0.0

        self._pending_update = False
        self._pending_publish = None
        self._buffer_inflight_keys = None
        self._buffer_inflight_values = None
        self.update_metrics_log.append(metrics)
        self._last_metrics = None

    @property
    def has_pending_update(self) -> bool:
        return self._pending_update

    def _reset_buffer_after_publish(self) -> None:
        self._buf_keys_arena.zero_()
        self._buf_values_arena.zero_()
        self._l_buf = 0
        self._steps_since_update = 0

    # ── Attention ──

    def _ensure_workspace(self, h_q: int) -> dict[str, torch.Tensor]:
        device = self.state["centers_padded_f16"].device
        n_pad_filter = int(self.state["N_pad"])
        ws = self._ws
        if ws is not None and ws["live_idx_filter"].shape == (h_q, n_pad_filter):
            return ws
        ws = {
            "top_scores":   torch.empty(h_q, S_FIXED, _FILT_L, device=device, dtype=torch.float32),
            "top_indices":  torch.empty(h_q, S_FIXED, _FILT_L, device=device, dtype=torch.int32),
            "depth":        torch.empty(h_q, device=device, dtype=torch.int32),
            "live_idx_filter": torch.empty(h_q, n_pad_filter, device=device, dtype=torch.int32),
            "live_count":   torch.zeros(h_q, device=device, dtype=torch.int32),
        }
        self._ws = ws
        return ws

    def attend(
        self,
        q: torch.Tensor,
        threshold: torch.Tensor,
        q_head_to_kv: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("call build() first")
        h_q, d = q.shape
        ws = self._ensure_workspace(h_q)

        _filter_with_workspace(
            q=q, threshold=threshold, state=self.state,
            q_head_to_kv=q_head_to_kv,
            live_idx=ws["live_idx_filter"], live_count=ws["live_count"],
            top_scores=ws["top_scores"], top_indices=ws["top_indices"],
            depth=ws["depth"],
        )

        if scale is None:
            scale = d ** -0.5

        return sdpa_cuda_sparse_v2_5_fp16(
            q=q,
            keys_f16=self.state["keys_padded_f16"],
            values_f16=self.state["values_padded_f16"],
            buffer_keys_f16=self._buf_keys_arena,
            buffer_values_f16=self._buf_values_arena,
            live_idx=ws["live_idx_filter"],
            live_count=ws["live_count"],
            l_buf=self._l_buf,
            q_head_to_kv=q_head_to_kv,
            scale=scale,
        )

    # ── Introspection ──

    @property
    def n_indexed(self) -> int:
        return int(self.state["N_used"]) if self.state is not None else 0

    @property
    def n_buffered(self) -> int:
        return self._l_buf

    def memory_bytes(self) -> int:
        total = 0
        for t in (self._buf_keys_arena, self._buf_values_arena):
            if t is not None:
                total += t.element_size() * t.numel()
        if self.state is None:
            return total
        for v in self.state.values():
            if isinstance(v, torch.Tensor):
                total += v.element_size() * v.numel()
        return total


def baseline_attention(
    q, keys, values, q_head_to_kv=None, scale=None,
):
    h_q, d = q.shape
    h_kv = keys.shape[0]
    scale = 1.0 / math.sqrt(d) if scale is None else float(scale)
    if h_q == h_kv:
        scores = torch.einsum("hd,hnd->hn", q, keys) * scale
        probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return torch.einsum("hn,hnd->hd", probs, values)
    groups = h_q // h_kv
    q_hg = q.view(h_kv, groups, d)
    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    out = torch.einsum("hgn,hnd->hgd", probs, values)
    return out.reshape(h_q, values.shape[-1])


def baseline_sdpa(
    q, keys, values, q_head_to_kv=None, scale=None,
):
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
