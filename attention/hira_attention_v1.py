import torch
from transformers.modeling_utils import AttentionInterface
from transformers.models.llama.modeling_llama import repeat_kv, eager_attention_forward
import math

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from hira.cache.hira_cache import CacheOutput

"""
Tasks:
    - [ ] Threshold finding
"""


def hira_attention_v1_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: CacheOutput,
    value: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    **kwargs,
):
    indexer = key.indexer
    searcher = key.searcher
    queued_keys = key.queued_keys
    queued_values = key.queued_values

    indexer_values = indexer.values

    if query.shape[-2] > 1:
        # prefilling
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

    values = torch.cat([indexer_values, queued_values], dim=-2)
    values = repeat_kv(values, module.num_key_value_groups)

    # INDEX
    threshold = torch.full(
        (query.shape[1],), -float("inf"), device=query.device, dtype=query.dtype
    )
    query_norm = torch.norm(query, dim=-1, keepdim=True)
    index_weights = searcher.search(
        query / query_norm,
        threshold,
        indexer,
        scaling=scaling * query_norm.squeeze(),
    )  # (head_dim, num_keys)
    
    index_weights = index_weights.masked_fill(
        index_weights == 0, float("-inf")
    )  # Remove padded keys
    index_weights = index_weights.unsqueeze(1).unsqueeze(0)

    # QUEUE
    queued_keys = repeat_kv(queued_keys, module.num_key_value_groups)
    queued_weights = torch.matmul(query, queued_keys.transpose(2, 3)) * scaling

    # MERGE
    attn_weights = torch.cat([index_weights, queued_weights], dim=-1)

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    # print(attn_weights.shape, value.shape)
    attn_output = torch.matmul(attn_weights, values)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


AttentionInterface.register("hira_attention_v1", hira_attention_v1_forward)
# Custom attention keys are not mask-aware by default; map to eager mask creation.
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "hira_attention_v1", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
)
