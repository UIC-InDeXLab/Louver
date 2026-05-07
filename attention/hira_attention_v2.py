"""
HIRA Attention v2 — Fused Triton decoding kernel
=================================================

Replaces v1's post-search masked_fill + cat + repeat_kv + two-pass softmax
with a single-pass fused Triton kernel that:

  • Streams over pre-computed index weights (from searcher), treating 0 as pruned
  • Computes queued QK dot-products on-the-fly
  • Uses online softmax to avoid materialising the full weight tensor
  • Handles GQA internally via q_head_to_kv mapping (no repeat_kv copy)

Prefilling still delegates to eager_attention_forward (unchanged).
"""

import torch
from transformers.modeling_utils import AttentionInterface
from transformers.models.llama.modeling_llama import eager_attention_forward, repeat_kv
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from hira.cache.hira_cache import CacheOutput
from hira.attention.kernels.attn_triton_kernels import fused_hira_attention


def hira_attention_v2_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: CacheOutput,
    value: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
):
    """
    Parameters match the HF ``AttentionInterface`` signature.

    During prefilling (query seq_len > 1) → eager attention.
    During decoding  (query seq_len == 1) → fused Triton kernel.
    """

    indexer = key.indexer
    searcher = key.searcher
    queued_keys = key.queued_keys  # (1, H_kv, Q, D)
    queued_values = key.queued_values  # (1, H_kv, Q, D)

    # ------------------------------------------------------------------
    # Prefilling — use standard eager attention
    # ------------------------------------------------------------------
    if query.shape[-2] > 1:
        # indices = torch.randperm(key.prefill_keys.shape[2])[:100]
        # samples = key.prefill_keys[:, :, indices, :]
        # indexer.samples = repeat_kv(samples, module.num_key_value_groups)
        return eager_attention_forward(
            module,
            query,
            key.prefill_keys,
            key.prefill_values,
            attention_mask,
            scaling,
            dropout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Decoding — fused kernel path
    # ------------------------------------------------------------------
    # query: (1, H_q, 1, D)
    H_q = query.shape[1]
    H_kv = queued_keys.shape[1]

    # ── Search indexed keys ──────────────────────────────────────────
    query_norm = torch.norm(query, dim=-1, keepdim=True)  # (1, H_q, 1, 1)
    q_n = query / query_norm
    # threshold = (
    #     (torch.matmul(q_n, indexer.samples.transpose(2, 3)).squeeze(0).squeeze(-2))
    #     .max(dim=1)
    #     .values
    # )  # (H_q, num_samples)
    threshold = torch.full(
        (H_q,), -float("inf"), device=query.device, dtype=query.dtype
    )
    index_weights = searcher.search(
        q_n,
        threshold,
        indexer,
        scaling=scaling * query_norm.view(-1),  # (H_q,)
    )  # (H_q, N)

    # ── Prepare kernel inputs (squeeze batch & seq dims) ─────────────
    q = query.squeeze(0).squeeze(-2).contiguous()  # (H_q, D)
    iv = indexer.values.squeeze(0).contiguous()  # (H_kv, N, D)
    qk = queued_keys.squeeze(0).contiguous()  # (H_kv, Q, D)
    qv = queued_values.squeeze(0).contiguous()  # (H_kv, Q, D)

    # ── GQA head mapping ─────────────────────────────────────────────
    num_kv_groups = H_q // H_kv
    q_head_to_kv = (
        torch.arange(H_q, device=query.device, dtype=torch.int64) // num_kv_groups
    )  # (H_q,)

    # ── Launch fused kernel ──────────────────────────────────────────
    attn_output = fused_hira_attention(
        index_weights=index_weights,
        index_values=iv,
        query=q,
        queued_keys=qk,
        queued_values=qv,
        q_head_to_kv=q_head_to_kv,
        scaling=scaling,
    )  # (H_q, D)

    # ── Reshape to HF expected format (B, L, H, D) ──────────────────
    attn_output = attn_output.unsqueeze(0).unsqueeze(0)  # (1, 1, H_q, D)

    return attn_output, None


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
AttentionInterface.register("hira_attention_v2", hira_attention_v2_forward)
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "hira_attention_v2", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
)
