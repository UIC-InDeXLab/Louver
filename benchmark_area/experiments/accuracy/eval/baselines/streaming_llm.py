"""
StreamingLLM KV cache via AttentionInterface.

Keeps first `sink` tokens (attention sinks) + `recent_size` most-recent tokens.
Budget = sink + recent_size, fixed regardless of sequence length.

Registers: "streaming_llm" attention forward.

Ref: Xiao et al., "Efficient Streaming Language Models with Attention Sinks"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS


# ── Cache layer ───────────────────────────────────────────────────────────────

class StreamingLLMCacheLayer(CacheLayerMixin):
    """Keeps sink tokens + recent window; evicts middle tokens."""

    def __init__(self, sink_size: int, recent_size: int):
        super().__init__()
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.budget = sink_size + recent_size

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """key_states/value_states: (H_kv, T, D)."""
        is_first = self.keys is None
        if is_first:
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)

        # Only evict during decode (T=1), not during prefill (T>1) to keep mask valid
        if not is_first and key_states.shape[1] == 1:
            N = self.keys.shape[1]
            if N > self.budget:
                sink_k = self.keys[:, :self.sink_size, :]
                sink_v = self.values[:, :self.sink_size, :]
                recent_k = self.keys[:, N - self.recent_size:, :]
                recent_v = self.values[:, N - self.recent_size:, :]
                self.keys = torch.cat([sink_k, recent_k], dim=1)
                self.values = torch.cat([sink_v, recent_v], dim=1)

        return self.keys, self.values

    def get_mask_sizes(self, query_length) -> tuple[int, int]:
        if not isinstance(query_length, int):
            query_length = query_length.shape[0]
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[1]

    def get_max_cache_shape(self) -> int:
        return self.budget


# ── Cache (multi-layer) ───────────────────────────────────────────────────────

@dataclass
class StreamingLLMCacheOutput:
    keys: torch.Tensor
    values: torch.Tensor
    layer_cache: StreamingLLMCacheLayer
    is_prefill: bool


class StreamingLLMCache(Cache):
    def __init__(self, sink_size: int, recent_size: int, num_layers: int):
        layers = [StreamingLLMCacheLayer(sink_size, recent_size) for _ in range(num_layers)]
        super().__init__(layers=layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        layer: StreamingLLMCacheLayer = self.layers[layer_idx]
        is_prefill = not layer.is_initialized
        layer.update(key_states, value_states)
        return StreamingLLMCacheOutput(
            keys=layer.keys,
            values=layer.values,
            layer_cache=layer,
            is_prefill=is_prefill,
        ), None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self):
        return None


# ── Attention forward ─────────────────────────────────────────────────────────

def streaming_llm_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: StreamingLLMCacheOutput,
    value: None,
    attention_mask,
    dropout: float,
    scaling: float,
    **kwargs,
):
    cache_out = key
    keys = cache_out.keys.unsqueeze(0)
    values = cache_out.values.unsqueeze(0)

    B, H_q, T, D = query.shape
    H_kv = keys.shape[1]
    group_size = H_q // H_kv

    # Prefill: SDPA (no OOM)
    if T > 1:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, keys, values, attention_mask, dropout, scaling=scaling, **kwargs
        )

    # Decode: standard eager attention on reduced KV
    if group_size > 1:
        keys_exp = keys.repeat_interleave(group_size, dim=1)
        vals_exp = values.repeat_interleave(group_size, dim=1)
    else:
        keys_exp = keys
        vals_exp = values

    N = keys_exp.shape[2]
    scores = torch.matmul(query, keys_exp.transpose(-2, -1)) * scaling  # (1, H_q, 1, N)
    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)
    return torch.matmul(attn_weights, vals_exp), None


# ── Registration ──────────────────────────────────────────────────────────────

AttentionInterface.register("streaming_llm", streaming_llm_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("streaming_llm", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
