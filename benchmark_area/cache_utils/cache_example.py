from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import torch
from torch import nn


model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True,
)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):  
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    # print(f"[AFTER] shapes: query={query.shape}, key={key_states.shape}, value={value_states.shape}")

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# counter = 0


def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    # global counter
    # counter += 1
    # print(f"Here in the foward... {counter}")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (1, 24, 1, 128)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (1, 8, 1, 128)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (1, 8, 1, 128)

    # print("===========")
    # print(
    #     f"shapes: query_states={query_states.shape}, key_states={key_states.shape}, value_states={value_states.shape}"
    # )

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    breakpoint()
    # print("is past_key_values not None:", past_key_values is not None)
    if past_key_values is not None:
        # print("caching...")
        # print(key_states.shape)
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    # print(f"key_states shape: {key_states.shape}, value_states shape: {value_states.shape}")

    attention_interface: Callable = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


for layer in model.model.layers:
    layer.self_attn.forward = forward.__get__(
        layer.self_attn, layer.self_attn.__class__
    )

prompt = "Explain what multi-head attention is in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=10, temperature=0.7, do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
