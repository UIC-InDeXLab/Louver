import math
import torch
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from transformers.models.llama.modeling_llama import repeat_kv


def sdp_attention_ref(
    module, query, key, value, attention_mask, scaling, dropout, **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)

    attn_weights = torch.nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


AttentionInterface.register("sdpa_attention_ref", sdp_attention_ref)
# Custom attention keys are not mask-aware by default; map to eager mask creation.
ALL_MASK_ATTENTION_FUNCTIONS.register(
    "sdpa_attention_ref", ALL_MASK_ATTENTION_FUNCTIONS["eager"]
)
