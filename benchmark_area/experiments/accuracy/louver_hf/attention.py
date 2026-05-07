"""
Custom HuggingFace AttentionInterface implementations for Louver.

Registers:
  - "louver_full"  → SubspaceKCenterIndex attend()
  - "louver_ta"    → TAIndex attend()

Both fall back to eager attention during prefill.
"""
from __future__ import annotations

import math

import torch
from transformers.modeling_utils import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from .cache import LouverCacheOutput


def _sdpa_prefill(module, query, key_cache_out: LouverCacheOutput, attention_mask, scaling, dropout, **kwargs):
    """SDPA (flash-attention) over prefill keys/values — avoids materializing O(N^2) weights."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module,
        query,
        key_cache_out.prefill_keys,
        key_cache_out.prefill_values,
        attention_mask,
        dropout,
        scaling=scaling,
        **kwargs,
    )


def louver_full_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: LouverCacheOutput,
    value: None,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
):
    # ── Prefill ──────────────────────────────────────────────────────────
    if query.shape[-2] > 1:
        return _sdpa_prefill(module, query, key, attention_mask, scaling, dropout, **kwargs)

    # ── Decode ───────────────────────────────────────────────────────────
    index = key.index
    threshold = key.threshold

    H_q = query.shape[1]
    H_kv = index.keys.shape[0]
    device = query.device

    q_2d = query.squeeze(0).squeeze(-2).to(torch.float16).contiguous()  # (H_q, D)

    if not hasattr(module, "_louver_q_head_to_kv") or module._louver_q_head_to_kv.shape[0] != H_q:
        g = H_q // H_kv
        module._louver_q_head_to_kv = (
            torch.arange(H_q, device=device, dtype=torch.int64) // g
        )
    q_head_to_kv = module._louver_q_head_to_kv

    dim_slices = index.state["dim_slices"]
    th_packed = threshold.get_subspace_threshold(q_2d, dim_slices)  # (2*S, H_q) fp16

    out_2d = index.attend(
        q=q_2d,
        th_per_subspace=th_packed,
        q_head_to_kv=q_head_to_kv,
        scale=scaling,
    )  # (H_q, D_v)

    return out_2d.unsqueeze(0).unsqueeze(0).to(query.dtype), None


def louver_ta_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: LouverCacheOutput,
    value: None,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
):
    # ── Prefill ──────────────────────────────────────────────────────────
    if query.shape[-2] > 1:
        return _sdpa_prefill(module, query, key, attention_mask, scaling, dropout, **kwargs)

    # ── Decode ───────────────────────────────────────────────────────────
    index = key.index
    threshold = key.threshold

    H_q = query.shape[1]
    H_kv = index.state["H_kv"] if "H_kv" in index.state else (
        index._buf_keys_arena.shape[0] if index._buf_keys_arena is not None else H_q
    )
    device = query.device

    q_2d = query.squeeze(0).squeeze(-2).to(torch.float16).contiguous()  # (H_q, D)

    if not hasattr(module, "_louver_q_head_to_kv") or module._louver_q_head_to_kv.shape[0] != H_q:
        g = H_q // H_kv
        module._louver_q_head_to_kv = (
            torch.arange(H_q, device=device, dtype=torch.int64) // g
        )
    q_head_to_kv = module._louver_q_head_to_kv

    th = threshold.get_threshold_ta(q_2d)  # (H_q,) float32

    out_2d = index.attend(
        q=q_2d,
        threshold=th,
        q_head_to_kv=q_head_to_kv,
        scale=scaling,
    )  # (H_q, D_v)

    return out_2d.unsqueeze(0).unsqueeze(0).to(query.dtype), None


# ── Registration ─────────────────────────────────────────────────────────────

AttentionInterface.register("louver_full", louver_full_attention_forward)
AttentionInterface.register("louver_ta", louver_ta_attention_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register("louver_full", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
ALL_MASK_ATTENTION_FUNCTIONS.register("louver_ta", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
