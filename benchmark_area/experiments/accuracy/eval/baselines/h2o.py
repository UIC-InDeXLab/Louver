"""
H2O (Heavy Hitter Oracle) KV cache eviction via AttentionInterface.

Keeps top-K tokens by cumulative attention score (heavy hitters) + R most-recent
tokens. Budget = heavy_ratio + recent_ratio of context length.

Registers: "h2o" attention forward.
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

class H2OCacheLayer(CacheLayerMixin):
    """Per-layer KV store with heavy-hitter eviction."""

    def __init__(self, heavy_budget: int, recent_budget: int):
        super().__init__()
        self.heavy_budget = heavy_budget
        self.recent_budget = recent_budget
        self.cum_scores: Optional[torch.Tensor] = None  # (H_kv, N)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append key_states/value_states; shape (H_kv, T, D)."""
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self.cum_scores = torch.zeros(
                key_states.shape[0], key_states.shape[1],
                device=key_states.device, dtype=torch.float32,
            )
            self.is_initialized = True
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)
            new_scores = torch.zeros(
                key_states.shape[0], key_states.shape[1],
                device=key_states.device, dtype=torch.float32,
            )
            self.cum_scores = torch.cat([self.cum_scores, new_scores], dim=1)
        return self.keys, self.values

    def update_scores_and_evict(self, attn_weights: torch.Tensor, group_size: int) -> None:
        """
        attn_weights: (H_q, N) softmax weights from decode step.
        Accumulate into kv-head scores (average over query group), then evict.
        """
        N = self.keys.shape[1]
        H_kv = self.keys.shape[0]

        if group_size > 1:
            kv_weights = attn_weights.view(H_kv, group_size, N).mean(dim=1)
        else:
            kv_weights = attn_weights

        self.cum_scores += kv_weights.to(torch.float32)

        total_budget = self.heavy_budget + self.recent_budget
        if N <= total_budget:
            return

        n_old = N - self.recent_budget
        old_scores = self.cum_scores[:, :n_old]  # (H_kv, n_old)
        n_heavy = min(self.heavy_budget, n_old)
        mean_scores = old_scores.mean(dim=0)
        _, keep_idx = torch.topk(mean_scores, n_heavy, dim=0, sorted=False)
        keep_idx, _ = torch.sort(keep_idx)

        recent_idx = torch.arange(n_old, N, device=self.keys.device)
        all_idx = torch.cat([keep_idx, recent_idx])

        self.keys = self.keys[:, all_idx, :]
        self.values = self.values[:, all_idx, :]
        self.cum_scores = self.cum_scores[:, all_idx]

    def get_mask_sizes(self, query_length) -> tuple[int, int]:
        if not isinstance(query_length, int):
            query_length = query_length.shape[0]
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[1]

    def get_max_cache_shape(self) -> int:
        return -1


# ── Cache (multi-layer) ───────────────────────────────────────────────────────

@dataclass
class H2OCacheOutput:
    keys: torch.Tensor
    values: torch.Tensor
    layer_cache: H2OCacheLayer
    is_prefill: bool


class H2OCache(Cache):
    def __init__(self, heavy_budget: int, recent_budget: int, num_layers: int):
        layers = [H2OCacheLayer(heavy_budget, recent_budget) for _ in range(num_layers)]
        super().__init__(layers=layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = key_states.squeeze(0)    # (H_kv, T, D)
        value_states = value_states.squeeze(0)
        layer: H2OCacheLayer = self.layers[layer_idx]
        is_prefill = not layer.is_initialized
        layer.update(key_states, value_states)
        return H2OCacheOutput(
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

def h2o_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,       # (B, H_q, T, D)
    key: H2OCacheOutput,
    value: None,
    attention_mask,
    dropout: float,
    scaling: float,
    **kwargs,
):
    cache_out = key
    keys = cache_out.keys.unsqueeze(0)     # (1, H_kv, N, D)
    values = cache_out.values.unsqueeze(0)

    B, H_q, T, D = query.shape
    H_kv = keys.shape[1]
    N = keys.shape[2]
    group_size = H_q // H_kv

    # Prefill: SDPA (fast); cum_scores start at zero, built up from decode steps
    if T > 1:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, keys, values, attention_mask, dropout, scaling=scaling, **kwargs
        )

    # Decode: eager attention (T=1, N can be large but not N^2)
    if group_size > 1:
        keys_exp = keys.repeat_interleave(group_size, dim=1)
        vals_exp = values.repeat_interleave(group_size, dim=1)
    else:
        keys_exp = keys
        vals_exp = values

    scores = torch.matmul(query, keys_exp.transpose(-2, -1)) * scaling  # (1, H_q, 1, N)
    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)

    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    out = torch.matmul(attn_weights, vals_exp)  # (1, H_q, 1, D)

    w_2d = attn_weights[0, :, 0, :]  # (H_q, N)
    cache_out.layer_cache.update_scores_and_evict(w_2d, group_size)

    return out, None


# ── Registration ──────────────────────────────────────────────────────────────

AttentionInterface.register("h2o", h2o_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("h2o", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
