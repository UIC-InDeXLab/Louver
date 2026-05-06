"""
Twilight top-p sparse attention (accuracy-mode, pure PyTorch).

Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning (NeurIPS 2025).
https://arxiv.org/abs/2502.02770

At each decode step: compute full attention weights, softmax-normalize, keep the
minimum set of tokens whose cumulative probability >= top_p, zero out the rest.

No eviction — full KV cache retained (standard DynamicCache).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import AttentionInterface

# Module-level config set by configure_twilight()
_top_p: float = 0.85
_skip_first_layers: int = 2


def _top_p_mask(attn_weights: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    attn_weights: (B, H_q, 1, N) unnormalized
    Returns bool mask (B, H_q, 1, N): True = keep.
    Keeps minimum tokens whose softmax mass sums to >= top_p.
    """
    probs = F.softmax(attn_weights.float(), dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # remove token i if cumulative mass *before* it already >= top_p
    remove = (cumsum - sorted_probs) >= top_p
    mask = torch.zeros_like(remove)
    mask.scatter_(dim=-1, index=sorted_idx, src=~remove)
    return mask


def twilight_attention_forward(
    module, query, key, value, attention_mask, dropout=0.0, **kwargs
):
    is_prefill = query.shape[2] > 1
    layer_idx = getattr(module, "layer_idx", None)
    skip = (layer_idx is not None) and (layer_idx < _skip_first_layers)

    if is_prefill or skip:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, **kwargs
        )

    scale = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    mask = _top_p_mask(attn_weights, _top_p)
    attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
    attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


AttentionInterface.register("twilight", twilight_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "twilight", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
)


def configure_twilight(top_p: float = 0.9, skip_first_layers: int = 2) -> None:
    global _top_p, _skip_first_layers
    _top_p = top_p
    _skip_first_layers = skip_first_layers
