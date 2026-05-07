"""Subspace k-center + ball_centroid index — fused-attention path only.

Per subspace the index stores parents (centers/radii) and a parent-major,
block-packed child layout produced by build_v2_7. At attention time a fused
Triton pipeline gates parents against per-subspace thresholds, evaluates the
online softmax over survivors and the decoding buffer, and merges the result.

Update can run synchronously (``update()``) or in parallel with attention
(``update_async()`` + ``wait_for_update()``). The async path requires an
overlap-aware update kernel — currently ``update_v4_0`` — which writes the
buffer's clustered data into the arena's unused tail without publishing the
invalid flags. ``wait_for_update`` then publishes those flags on the
attention stream after waiting on the update_done event, so attention only
ever observes the new range as either fully invalid or fully published.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .kernels import get_attention, get_build, get_update
from .kernels._update_v3_utils import apply_pending_publish


@dataclass
class IndexConfig:
    n_subspaces: int = 8
    bf: int = 4
    refine_iter: int = 5
    update_mode: str = "inc"                        # "full" | "inc"
    build_kernel: str = "build_v2_7"
    update_kernel: str = "update_v4_0"
    attention_kernel: str = "attention_v5_14"
    # Parallel-update toggles. Async path requires an update kernel that
    # returns a 4-tuple (state, keys, values, pending_publish).
    parallel_update: bool = False
    update_stream_priority: int = -1


@dataclass
class _UpdateMetrics:
    """Per-fire telemetry for the async update path.

    All times are in milliseconds. ``kernel_ms`` is GPU-side (event delta).
    ``host_wait_ms`` is the host-clock time spent inside ``wait_for_update``
    when the previous update hadn't finished by then. ``inflight_at_fire`` is
    set if a prior update was still in flight when this one was fired.
    """

    kernel_ms: float = 0.0
    host_wait_ms: float = 0.0
    inflight_at_fire: bool = False
    fire_step: int = -1
    publish_step: int = -1


class SubspaceKCenterIndex:
    """Holds the index state + key/value history + decoding buffer."""

    def __init__(self, cfg: IndexConfig):
        self.cfg = cfg
        self._build = get_build(cfg.build_kernel)
        self._update = get_update(cfg.update_kernel)
        self._attention = get_attention(cfg.attention_kernel)

        self.state: dict | None = None
        self.keys: torch.Tensor | None = None          # (H_kv, N, D)
        self.values: torch.Tensor | None = None        # (H_kv, N, D_v)
        self.buffer: torch.Tensor | None = None        # (H_kv, B_active, D)
        self.values_buffer: torch.Tensor | None = None # (H_kv, B_active, D_v)
        self._steps_since_update = 0

        # Parallel-update plumbing.
        self._update_stream: torch.cuda.Stream | None = None
        self._update_start_event: torch.cuda.Event | None = None
        self._update_done_event: torch.cuda.Event | None = None
        self._pending_update: bool = False
        self._pending_publish: dict | None = None
        self._buffer_inflight: torch.Tensor | None = None
        self._values_buffer_inflight: torch.Tensor | None = None
        # Reusable cat-buffers for attend() during the overlap window.
        self._buffer_concat_cache: torch.Tensor | None = None
        self._values_buffer_concat_cache: torch.Tensor | None = None

        # Bookkeeping for telemetry, populated by update_async / wait_for_update.
        self.update_metrics_log: list[_UpdateMetrics] = []
        self._last_metrics: _UpdateMetrics | None = None
        self.n_overlap_misses: int = 0

    # ── Build ─────────────────────────────────────────────────────────

    def build(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> "SubspaceKCenterIndex":
        self.keys = keys.contiguous()
        self.values = values.contiguous()
        self.state = self._build(
            keys=self.keys,
            bf=self.cfg.bf,
            n_subspaces=self.cfg.n_subspaces,
            refine_iter=self.cfg.refine_iter,
            values=self.values,
        )
        self.buffer = torch.empty(
            self.keys.shape[0], 0, self.keys.shape[2],
            device=self.keys.device, dtype=self.keys.dtype,
        )
        self.values_buffer = torch.empty(
            self.values.shape[0], 0, self.values.shape[2],
            device=self.values.device, dtype=self.values.dtype,
        )
        self._steps_since_update = 0

        if self.cfg.parallel_update:
            self._update_stream = torch.cuda.Stream(
                priority=self.cfg.update_stream_priority
            )
        return self

    # ── Decoding-time key/value handling ──────────────────────────────

    def append_decoding_kv(
        self, new_key: torch.Tensor, new_value: torch.Tensor
    ) -> None:
        """Append a matching (k, v) pair to the active decoding buffer."""
        if new_key.dim() == 2:
            new_key = new_key.unsqueeze(1)
        if new_value.dim() == 2:
            new_value = new_value.unsqueeze(1)
        self.buffer = torch.cat([self.buffer, new_key], dim=1)
        self.values_buffer = torch.cat([self.values_buffer, new_value], dim=1)
        self._steps_since_update += 1

    def needs_update(self, update_interval: int) -> bool:
        return self._steps_since_update >= update_interval

    # ── Synchronous update (legacy path) ──────────────────────────────

    def update(self) -> None:
        if self.buffer.shape[1] == 0:
            self._steps_since_update = 0
            return

        if self.cfg.update_mode == "full":
            new_keys = torch.cat([self.keys, self.buffer], dim=1).contiguous()
            new_values = torch.cat(
                [self.values, self.values_buffer], dim=1
            ).contiguous()
            self.state = self._build(
                keys=new_keys,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                values=new_values,
            )
            self.keys = new_keys
            self.values = new_values
        elif self.cfg.update_mode == "inc":
            ret = self._update(
                state=self.state,
                old_keys=self.keys,
                buffer_keys=self.buffer,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                old_values=self.values,
                buffer_values=self.values_buffer,
            )
            # update_v4 returns 4-tuple; sync-update over v4 publishes inline.
            if len(ret) == 4:
                new_state, new_keys, new_values, pending = ret
                if pending is not None:
                    apply_pending_publish(pending)
                self.state = new_state
                if new_keys is not None:
                    self.keys = new_keys
                if new_values is not None:
                    self.values = new_values
            else:
                self.state, new_keys, new_values = ret
                if new_keys is not None:
                    self.keys = new_keys
                if new_values is not None:
                    self.values = new_values
        else:
            raise ValueError(f"Unknown update_mode: {self.cfg.update_mode!r}")

        self.buffer = torch.empty(
            self.keys.shape[0], 0, self.keys.shape[2],
            device=self.keys.device, dtype=self.keys.dtype,
        )
        self.values_buffer = torch.empty(
            self.values.shape[0], 0, self.values.shape[2],
            device=self.values.device, dtype=self.values.dtype,
        )
        self._steps_since_update = 0

    # ── Asynchronous update (parallel path) ───────────────────────────

    def update_async(self, fire_step: int = -1) -> None:
        """Launch an update on the side stream without blocking the host.

        The current ``buffer`` becomes ``buffer_inflight`` and remains
        readable by ``attend()`` until ``wait_for_update`` is called and
        the publish step runs. A fresh empty ``buffer`` starts collecting
        new tokens immediately.
        """
        if self.buffer.shape[1] == 0:
            self._steps_since_update = 0
            return
        if self.cfg.update_mode != "inc":
            raise RuntimeError(
                "update_async requires update_mode='inc' (overlap kernel"
                " must be the v4 family)."
            )
        if self._update_stream is None:
            raise RuntimeError(
                "update_async requires cfg.parallel_update=True (no side stream)."
            )

        attn_stream = torch.cuda.current_stream()
        metrics = _UpdateMetrics(fire_step=fire_step)

        if self._pending_update:
            # Previous update still in flight — wait now and bump the miss
            # counter. This serializes updates against each other but does
            # not touch attention's stream more than necessary.
            metrics.inflight_at_fire = True
            self.n_overlap_misses += 1
            self._wait_and_publish(attn_stream)

        # Hand off the buffer; the inflight tensors must outlive the kernel
        # call on update_stream. record_stream tells the caching allocator
        # to keep them alive until update_stream is done with them.
        self._buffer_inflight = self.buffer
        self._values_buffer_inflight = self.values_buffer
        self._buffer_inflight.record_stream(self._update_stream)
        self._values_buffer_inflight.record_stream(self._update_stream)

        # Fresh empty buffers for the next tokens.
        self.buffer = torch.empty(
            self._buffer_inflight.shape[0], 0, self._buffer_inflight.shape[2],
            device=self._buffer_inflight.device, dtype=self._buffer_inflight.dtype,
        )
        self.values_buffer = torch.empty(
            self._values_buffer_inflight.shape[0], 0, self._values_buffer_inflight.shape[2],
            device=self._values_buffer_inflight.device,
            dtype=self._values_buffer_inflight.dtype,
        )

        # Make sure update_stream sees the producing stream's writes
        # (the appends/cats that built buffer_inflight all ran on attn_stream).
        self._update_stream.wait_stream(attn_stream)

        if self._update_start_event is None:
            self._update_start_event = torch.cuda.Event(enable_timing=True)
            self._update_done_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self._update_stream):
            self._update_start_event.record(self._update_stream)
            ret = self._update(
                state=self.state,
                old_keys=self.keys,
                buffer_keys=self._buffer_inflight,
                bf=self.cfg.bf,
                n_subspaces=self.cfg.n_subspaces,
                refine_iter=self.cfg.refine_iter,
                old_values=self.values,
                buffer_values=self._values_buffer_inflight,
            )
            if len(ret) != 4:
                raise RuntimeError(
                    "update_async requires an overlap-aware update kernel "
                    "returning (state, keys, values, pending_publish). "
                    f"Got tuple of len {len(ret)} from {self.cfg.update_kernel}."
                )
            new_state, new_keys, new_values, pending = ret
            self._update_done_event.record(self._update_stream)

        self.state = new_state
        if new_keys is not None:
            self.keys = new_keys
        if new_values is not None:
            self.values = new_values
        self._pending_publish = pending
        self._pending_update = True
        self._last_metrics = metrics
        self._steps_since_update = 0

    def wait_for_update(self) -> None:
        """If an update is pending, wait for it on the current stream and publish.

        Safe to call when no update is in flight (no-op). Also safe to call
        once per step regardless — the host check is cheap.
        """
        if not self._pending_update:
            return
        attn_stream = torch.cuda.current_stream()
        self._wait_and_publish(attn_stream)

    def try_publish(self) -> bool:
        """Non-blocking: if an update is pending AND already done on the GPU,
        publish it now. Otherwise leave it in flight. Returns True if a publish
        happened.

        Call this once per decode step (cheap host probe). It keeps the "miss"
        accounting in update_async honest — without it, every fire after the
        first observes ``_pending_update=True`` and counts as a miss even when
        the GPU work finished long ago.
        """
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
        # Cheap host probe — if the GPU already finished, skip the host stall.
        already_done = self._update_done_event.query()
        host_wait_start = time.perf_counter() if not already_done else None

        # Make the new state visible to attn_stream, then publish on it.
        attn_stream.wait_event(self._update_done_event)
        if self._pending_publish is not None:
            apply_pending_publish(self._pending_publish)

        if host_wait_start is not None:
            metrics.host_wait_ms = (time.perf_counter() - host_wait_start) * 1000.0

        # Kernel time (event delta). Both events are on update_stream and we've
        # already done a stream-level wait, so elapsed_time is safe to query.
        try:
            metrics.kernel_ms = self._update_start_event.elapsed_time(
                self._update_done_event
            )
        except Exception:
            metrics.kernel_ms = 0.0

        self._pending_update = False
        self._pending_publish = None
        # Drop inflight refs; allocator can recycle once update_stream is done
        # (record_stream above ensures correct lifetime).
        self._buffer_inflight = None
        self._values_buffer_inflight = None
        self.update_metrics_log.append(metrics)
        self._last_metrics = None

    @property
    def has_pending_update(self) -> bool:
        return self._pending_update

    def consume_last_metrics(self) -> _UpdateMetrics | None:
        """Return the metrics for the most recent published update, if any."""
        if not self.update_metrics_log:
            return None
        return self.update_metrics_log[-1]

    # ── Fused attention ───────────────────────────────────────────────

    def _attend_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine inflight + active buffers for the attention call.

        During the overlap window both must be visible — inflight holds the
        tokens being merged into the index; active holds tokens appended since
        the fire. Outside that window, only the active buffer is present.
        """
        if self._buffer_inflight is None or self._buffer_inflight.shape[1] == 0:
            return self.buffer, self.values_buffer
        if self.buffer.shape[1] == 0:
            return self._buffer_inflight, self._values_buffer_inflight
        keys_cat = torch.cat([self._buffer_inflight, self.buffer], dim=1)
        values_cat = torch.cat(
            [self._values_buffer_inflight, self.values_buffer], dim=1
        )
        return keys_cat, values_cat

    def attend(
        self,
        q: torch.Tensor,                               # (H_q, D)
        th_per_subspace: torch.Tensor,                 # (S, H_q)
        q_head_to_kv: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """Return (H_q, D_v) attention output over (index + buffer) K/V."""
        buffer_keys, buffer_values = self._attend_buffers()
        return self._attention(
            q=q,
            th_per_subspace=th_per_subspace,
            state=self.state,
            buffer_keys=buffer_keys,
            buffer_values=buffer_values,
            keys_children=self.keys,
            q_head_to_kv=q_head_to_kv,
            scale=scale,
        )

    # ── Introspection ─────────────────────────────────────────────────

    def last_cluster_pass(self) -> torch.Tensor | None:
        """The most recent cluster-pass mask produced by attend().

        Shape: (S, H_q, K) int8, 1 = parent survives threshold in subspace s
        for query head h_q. AND across S = parents actually scanned by the
        index kernel. Returns None if the active attention kernel doesn't
        expose this or attend() hasn't run yet.

        Safe to read after `attend()` returns — it syncs via _time_gpu. The
        tensor gets overwritten on the next attend() call.
        """
        if self.state is None:
            return None
        for cache_name in (
            "_attn_v1_16_fixed",
            "_attn_v1_17_fixed",
            "_attn_v1_18_fixed",
            "_attn_v1_20_fixed",
            "_attn_v1_22_fixed",
            "_attn_v1_23_fixed",
            "_attn_v1_24_fixed",
            "_attn_v2_6_fixed",
            "_attn_v2_15_fixed",
        ):
            wrap = self.state.get(cache_name)
            if not wrap:
                continue
            fixed = wrap.get("fixed")
            if fixed is None:
                continue
            cluster_pass = fixed["shared"].get("cluster_pass")
            if cluster_pass is not None:
                return cluster_pass
        return None

    @property
    def n_children(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[1])

    @property
    def n_buffered(self) -> int:
        active = 0 if self.buffer is None else int(self.buffer.shape[1])
        inflight = (
            0 if self._buffer_inflight is None
            else int(self._buffer_inflight.shape[1])
        )
        return active + inflight

    def memory_bytes(self) -> int:
        """Approximate GPU memory held by the index (keys + values + state)."""
        total = 0
        for t in (
            self.keys, self.values, self.buffer, self.values_buffer,
            self._buffer_inflight, self._values_buffer_inflight,
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
        return total


def baseline_attention(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,         # kept for API parity
    scale: float | None = None,
) -> torch.Tensor:
    """Brute-force dense attention: softmax(scale * q @ K^T) @ V → (H_q, D_v).

    GQA-aware: when H_q != H_kv we reshape Q into (groups, H_kv, D) and
    broadcast against the original (H_kv, N, D) keys rather than materializing
    an (H_q, N, D) expansion — same work as `enable_gqa=True` for SDPA.
    """
    import math

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
    # Q heads are kv-major: q_head i -> kv_head i // groups (matches
    # _q_to_kv_map and SDPA's enable_gqa convention).
    q_hg = q.view(h_kv, groups, d)
    scores = torch.einsum("hgd,hnd->hgn", q_hg, keys) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    out = torch.einsum("hgn,hnd->hgd", probs, values)
    return out.reshape(h_q, d_v)


def baseline_sdpa(
    q: torch.Tensor,                                  # (H_q, D)
    keys: torch.Tensor,                               # (H_kv, N, D)
    values: torch.Tensor,                             # (H_kv, N, D_v)
    q_head_to_kv: torch.Tensor | None = None,         # kept for API parity
    scale: float | None = None,
) -> torch.Tensor:
    """torch.nn.functional.scaled_dot_product_attention with native GQA.

    Uses `enable_gqa=True` so K/V stay at H_kv — no (H_q, N, D) expansion
    inside the timed region. Requires PyTorch >= 2.5.
    """
    import math

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
