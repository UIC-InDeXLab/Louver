"""
LouverCache: HuggingFace Cache interface wrapping kernel_impl indices.

Pattern mirrors hira_attention_v2 / HiraCache:
  - update() returns (LouverCacheOutput, None)
  - LouverCacheOutput is received as `key` in the attention forward function
  - prefill_keys/prefill_values non-None only on the prefill call

Supports both variants:
  - variant="full"  → SubspaceKCenterIndex (attention_v5_14)
  - variant="ta"    → TAIndex (ta_filter_v8 + sparse_sdpa_v2_5)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig

HIRA_ROOT = Path(__file__).resolve().parents[4]
BENCH_ROOT = HIRA_ROOT / "benchmark_area"
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from kernel_impl.index import SubspaceKCenterIndex, IndexConfig                         # noqa: E402
from kernel_impl.TA_filter_alg.index import TAIndex, TAIndexConfig                      # noqa: E402
from kernel_impl.TA_filter_alg.kernels.update.update_v1_1 import BUFFER_SIZE as TA_BUF # noqa: E402

from .threshold import LouverThreshold


@dataclass
class LouverCacheOutput:
    """What the custom attention forward receives as `key`."""
    variant: str                          # "full" | "ta"
    index: SubspaceKCenterIndex | TAIndex
    threshold: LouverThreshold
    q_head_to_kv: torch.Tensor            # (H_q,) int64
    scale: float

    # non-None only during prefill
    prefill_keys: torch.Tensor | None = None   # (B, H_kv, N, D)
    prefill_values: torch.Tensor | None = None


class LouverCacheLayer(CacheLayerMixin):
    """Manages one transformer layer's Louver index."""

    def __init__(
        self,
        variant: str,
        threshold_kwargs: dict,
        full_index_cfg: IndexConfig | None,
        ta_index_cfg: TAIndexConfig | None,
        update_interval: int,
    ):
        super().__init__()
        self.variant = variant
        self.threshold_kwargs = threshold_kwargs
        self.full_index_cfg = full_index_cfg or IndexConfig()
        self.ta_index_cfg = ta_index_cfg or TAIndexConfig()
        self.update_interval = update_interval

        self.index: SubspaceKCenterIndex | TAIndex | None = None
        self.threshold: LouverThreshold | None = None
        self.q_head_to_kv: torch.Tensor | None = None
        self.scale: float = 1.0
        self.is_initialized: bool = False
        self._seq_len: int = 0
        self._steps: int = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass  # handled inside update() on first call

    def _make_index(self):
        if self.variant == "full":
            return SubspaceKCenterIndex(self.full_index_cfg)
        return TAIndex(self.ta_index_cfg)

    def _build_q_head_to_kv(self, H_q: int, H_kv: int, device) -> torch.Tensor:
        g = H_q // H_kv
        return (torch.arange(H_q, device=device, dtype=torch.int64) // g)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        return self._seq_len + query_length, 0

    def update(
        self,
        key_states: torch.Tensor,   # (B, H_kv, L, D) from model
        value_states: torch.Tensor,
        cache_kwargs: dict | None = None,
    ):
        # key_states/value_states: (1, H_kv, L, D) — always batch=1 for inference
        B, H_kv, L, D = key_states.shape
        device = key_states.device

        if not self.is_initialized:
            # ── Prefill ──────────────────────────────────────────────────
            self.index = self._make_index()
            self.threshold = LouverThreshold(**self.threshold_kwargs)

            keys_f16 = key_states.squeeze(0).to(torch.float16).contiguous()  # (H_kv, N, D)
            vals_f16 = value_states.squeeze(0).to(torch.float16).contiguous()

            self.index.build(keys_f16, vals_f16)
            self.threshold.prefill_prep(keys_f16)
            self._seq_len = L
            self.scale = float(cache_kwargs.get("scaling", D ** -0.5)) if cache_kwargs else D ** -0.5
            self.is_initialized = True

            out = LouverCacheOutput(
                variant=self.variant,
                index=self.index,
                threshold=self.threshold,
                q_head_to_kv=self._build_q_head_to_kv(H_kv, H_kv, device),  # placeholder
                scale=self.scale,
                prefill_keys=key_states,
                prefill_values=value_states,
            )
            return out, None

        # ── Decode step ───────────────────────────────────────────────────
        assert L == 1, f"Expected decode step (L=1), got L={L}"
        new_key_f16 = key_states.squeeze(0).to(torch.float16)   # (H_kv, 1, D)
        new_val_f16 = value_states.squeeze(0).to(torch.float16)

        self.index.append_decoding_kv(new_key_f16, new_val_f16)
        self._seq_len += 1
        self._steps += 1

        self.threshold.update(new_key_f16, self._seq_len)

        if self.index.needs_update(self.update_interval):
            self.index.update()

        H_q_approx = H_kv  # corrected in attention forward via module attr
        out = LouverCacheOutput(
            variant=self.variant,
            index=self.index,
            threshold=self.threshold,
            q_head_to_kv=self._build_q_head_to_kv(H_q_approx, H_kv, device),
            scale=self.scale,
        )
        return out, None

    def get_seq_length(self) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def reset(self):
        self.index = None
        self.threshold = None
        self.q_head_to_kv = None
        self.is_initialized = False
        self._seq_len = 0
        self._steps = 0


class LouverCache(Cache):
    """HuggingFace Cache implementation wrapping per-layer LouverCacheLayer."""

    def __init__(
        self,
        model_config: PreTrainedConfig,
        variant: str = "ta",
        threshold_mode: str = "oracle",
        oracle: str = "sample_max",
        budget_fraction: float = 0.1,
        sample_size: int = 256,
        update_interval: int | None = None,
        full_index_cfg: IndexConfig | None = None,
        ta_index_cfg: TAIndexConfig | None = None,
    ):
        assert variant in ("full", "ta"), f"variant must be 'full' or 'ta', got {variant!r}"

        threshold_kwargs = dict(
            mode=threshold_mode,
            oracle=oracle,
            budget_fraction=budget_fraction,
            sample_size=sample_size,
        )

        _update_interval = update_interval or TA_BUF  # default = 256

        config = model_config.get_text_config(decoder=True)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            n = config.num_hidden_layers
            if getattr(config, "sliding_window", None) is not None:
                layer_types = ["sliding_attention"] * n
            else:
                layer_types = ["full_attention"] * n
        if hasattr(config, "num_kv_shared_layers"):
            layer_types = layer_types[: -config.num_kv_shared_layers]

        layers = [
            LouverCacheLayer(
                variant=variant,
                threshold_kwargs=threshold_kwargs,
                full_index_cfg=full_index_cfg,
                ta_index_cfg=ta_index_cfg,
                update_interval=_update_interval,
            )
            for _ in layer_types
        ]

        super().__init__(layers=layers, offloading=False, offload_only_non_sliding=None)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ):
        out, _ = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)
        return out, None
