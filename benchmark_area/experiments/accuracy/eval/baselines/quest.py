"""
Quest page-level KV cache retrieval via AttentionInterface.

During decode: score each chunk via sign(q) · max(k_chunk), select top-K chunks,
mask out the rest with -inf before softmax.

Registers: "quest" attention forward.

Ref: Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"
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

class QuestCacheLayer(CacheLayerMixin):
    """Per-layer KV store; maintains per-chunk min/max for fast scoring."""

    def __init__(self, chunk_size: int, token_budget: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.token_budget = token_budget
        self.chunk_max: Optional[torch.Tensor] = None   # (H_kv, n_chunks, D)
        self.chunk_min: Optional[torch.Tensor] = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """key_states/value_states: (H_kv, T, D)."""
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self.is_initialized = True
        else:
            self.keys = torch.cat([self.keys, key_states], dim=1)
            self.values = torch.cat([self.values, value_states], dim=1)
        self._rebuild_chunks()
        return self.keys, self.values

    def _rebuild_chunks(self):
        H_kv, N, D = self.keys.shape
        C = self.chunk_size
        n_full = N // C
        if n_full == 0:
            self.chunk_max = None
            self.chunk_min = None
            return
        k_trunc = self.keys[:, :n_full * C, :].view(H_kv, n_full, C, D)
        self.chunk_max = k_trunc.max(dim=2).values
        self.chunk_min = k_trunc.min(dim=2).values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.shape[1]

    def get_max_cache_shape(self) -> int:
        return -1


# ── Cache (multi-layer) ───────────────────────────────────────────────────────

@dataclass
class QuestCacheOutput:
    keys: torch.Tensor
    values: torch.Tensor
    layer_cache: QuestCacheLayer
    is_prefill: bool


class QuestCache(Cache):
    def __init__(self, chunk_size: int, token_budget: int, num_layers: int):
        layers = [QuestCacheLayer(chunk_size, token_budget) for _ in range(num_layers)]
        super().__init__(layers=layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        layer: QuestCacheLayer = self.layers[layer_idx]
        is_prefill = not layer.is_initialized
        layer.update(key_states, value_states)
        return QuestCacheOutput(
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

def quest_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,       # (B, H_q, T, D)
    key: QuestCacheOutput,
    value: None,
    attention_mask,
    dropout: float,
    scaling: float,
    **kwargs,
):
    cache_out = key
    layer = cache_out.layer_cache
    keys = cache_out.keys.unsqueeze(0)
    values = cache_out.values.unsqueeze(0)

    B, H_q, T, D = query.shape
    H_kv = keys.shape[1]
    N = keys.shape[2]
    group_size = H_q // H_kv

    if group_size > 1:
        keys_exp = keys.repeat_interleave(group_size, dim=1)
        vals_exp = values.repeat_interleave(group_size, dim=1)
    else:
        keys_exp = keys
        vals_exp = values

    # Prefill or no chunks yet: use SDPA to avoid materializing O(N^2) weights
    if T > 1 or layer.chunk_max is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(
            module, query, keys, values, attention_mask, dropout, scaling=scaling, **kwargs
        )

    # ── Decode: Quest chunk scoring ──────────────────────────────────────
    chunk_max = layer.chunk_max  # (H_kv, n_chunks, D)
    chunk_min = layer.chunk_min
    n_chunks = chunk_max.shape[1]
    C = layer.chunk_size

    q_2d = query[0, :, 0, :].float()  # (H_q, D)

    if group_size > 1:
        q_kv = q_2d.view(H_kv, group_size, D).mean(dim=1)  # (H_kv, D)
    else:
        q_kv = q_2d

    chunk_max_f = chunk_max.float()
    chunk_min_f = chunk_min.float()
    q_kv_f = q_kv.float()

    score_max = (q_kv_f.unsqueeze(1) * chunk_max_f).sum(-1)  # (H_kv, n_chunks)
    score_min = (q_kv_f.unsqueeze(1) * chunk_min_f).sum(-1)
    chunk_scores = torch.maximum(score_max, score_min)         # (H_kv, n_chunks)

    if group_size > 1:
        chunk_scores_q = chunk_scores.repeat_interleave(group_size, dim=0)  # (H_q, n_chunks)
    else:
        chunk_scores_q = chunk_scores

    n_budget_chunks = max(1, layer.token_budget // C)
    n_sel = min(n_budget_chunks, n_chunks)

    # Build token keep-mask vectorized: (H_q, N)
    if n_sel < n_chunks:
        _, top_chunk_idx = torch.topk(chunk_scores_q, n_sel, dim=1)  # (H_q, n_sel)
        # Convert chunk indices to token ranges: offset = chunk_idx * C
        token_offsets = top_chunk_idx * C  # (H_q, n_sel)
        # Build per-token index: (H_q, n_sel, C) → flatten → scatter into mask
        chunk_range = torch.arange(C, device=query.device)  # (C,)
        token_idx = (token_offsets.unsqueeze(-1) + chunk_range).clamp(max=N - 1)  # (H_q, n_sel, C)
        keep_mask = torch.zeros(H_q, N, device=query.device, dtype=torch.bool)
        keep_mask.scatter_(1, token_idx.view(H_q, -1), True)
        # Always keep remainder (tokens past last full chunk)
        remainder_start = n_chunks * C
        if remainder_start < N:
            keep_mask[:, remainder_start:] = True
    else:
        keep_mask = torch.ones(H_q, N, device=query.device, dtype=torch.bool)

    scores = torch.matmul(query, keys_exp.transpose(-2, -1)) * scaling  # (1, H_q, 1, N)
    fill_mask = ~keep_mask.unsqueeze(0).unsqueeze(2)                    # (1, H_q, 1, N)
    scores = scores.masked_fill(fill_mask, float("-inf"))

    attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = F.dropout(attn_weights, p=dropout)

    return torch.matmul(attn_weights, vals_exp), None


# ── Registration ──────────────────────────────────────────────────────────────

AttentionInterface.register("quest", quest_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("quest", ALL_MASK_ATTENTION_FUNCTIONS["eager"])
