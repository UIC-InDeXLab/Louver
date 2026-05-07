"""Shared utilities for false-negativity experiments.

Sparse attention is simulated by dropping selected KEY positions (i.e. removing
KV entries from the cache). We do this with a 4-D additive attention mask:
columns of dropped positions get -inf, so no query attends to them.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("EXP_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


IRRELEVANT_WORDS = {
    "the", "of", "a", "an", "and", "with", "to", "in", "on", "just",
    "one", "nothing", "else", "this", ",", ".", ":",
}


@dataclass
class PromptInfo:
    prompt: str
    input_ids: torch.Tensor          # (1, S)
    number_token_positions: List[List[int]]  # per-number list of token indices
    answer_str: str                  # ground-truth sum as string
    answer_first_token_id: int       # first token id of " <sum>"


def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="eager",
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


def build_prompt(numbers: Sequence[int]) -> str:
    list_str = ", ".join(str(n) for n in numbers)
    return (
        f"Consider the list of numbers: {list_str}. "
        f"What is the sum of the numbers? "
        f"Answer with just one integer and nothing else. The answer is"
    )


def _find_number_positions(
    tok, prompt: str, numbers: Sequence[int]
) -> List[List[int]]:
    """Map each number in `numbers` to its token indices in the tokenized prompt."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    out: List[List[int]] = []
    cursor = 0
    for n in numbers:
        s = str(n)
        idx = prompt.find(s, cursor)
        assert idx >= 0, f"number {n} not found from cursor {cursor}"
        end = idx + len(s)
        toks = [i for i, (a, b) in enumerate(offsets) if a < end and b > idx and b > a]
        assert toks, f"no token for number {n}"
        out.append(toks)
        cursor = end
    return out


def make_prompt_info(tok, numbers: Sequence[int], device: str = "cuda") -> PromptInfo:
    prompt = build_prompt(numbers)
    enc = tok(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    positions = _find_number_positions(tok, prompt, numbers)
    answer = str(sum(numbers))
    # First token in " <num>" that actually carries a digit (skip leading-space token)
    ans_ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
    digit_id = None
    for tid in ans_ids:
        s = tok.decode([tid])
        if any(c.isdigit() for c in s):
            digit_id = tid
            break
    if digit_id is None:
        digit_id = ans_ids[-1]
    return PromptInfo(
        prompt=prompt,
        input_ids=input_ids,
        number_token_positions=positions,
        answer_str=answer,
        answer_first_token_id=digit_id,
    )


def greedy_answer(model, tok, input_ids: torch.Tensor, drop_positions: Sequence[int],
                  max_new_tokens: int = 4) -> str:
    """Greedy-decode `max_new_tokens` with drop applied at every step. Returns decoded string."""
    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = forward_with_drop(model, ids, drop_positions)
        nxt = int(logits.argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return tok.decode(ids[0, input_ids.shape[1]:])


def forward_with_drop(model, input_ids: torch.Tensor, drop_positions: Sequence[int]):
    """Run a single forward pass dropping given KEY positions. Returns logits at last token."""
    B, S = input_ids.shape
    device = input_ids.device
    dtype = next(model.parameters()).dtype

    # Causal additive mask
    neg = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, S, S, device=device, dtype=dtype)
    causal = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
    mask.masked_fill_(causal, neg)
    if drop_positions:
        col = torch.zeros(S, device=device, dtype=torch.bool)
        col[list(drop_positions)] = True
        mask.masked_fill_(col[None, None, None, :], neg)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=mask, use_cache=False)
    return out.logits[0, -1].float()  # (V,)


def dist_metrics(baseline_logits: torch.Tensor, dropped_logits: torch.Tensor):
    """Distribution-level metrics that do not depend on answer tokenization."""
    pb = F.softmax(baseline_logits, dim=-1)
    pd = F.softmax(dropped_logits, dim=-1)
    kl = F.kl_div(pd.log(), pb, reduction="sum").item()
    return {
        "kl": kl,
        "top1_agrees_with_dense": int(int(baseline_logits.argmax()) == int(dropped_logits.argmax())),
    }


def answer_match(model, tok, input_ids: torch.Tensor, drop_positions: Sequence[int],
                 answer_str: str, max_new_tokens: int = 4) -> int:
    """1 iff greedy decode with the given drop produces a string containing the answer."""
    gen = greedy_answer(model, tok, input_ids, drop_positions, max_new_tokens)
    return int(answer_str in gen)


def full_metrics(model, tok, info, drop_positions, base_logits, drop_logits):
    m = dist_metrics(base_logits, drop_logits)
    m["answer_correct_drop"] = answer_match(
        model, tok, info.input_ids, drop_positions, info.answer_str
    )
    return m


def random_lists(rng: random.Random, n: int, low: int = 1, high: int = 3) -> List[int]:
    return [rng.randint(low, high) for _ in range(n)]


def greedy_string(model, tok, input_ids, drop_positions, max_new_tokens=4):
    return greedy_answer(model, tok, input_ids, drop_positions, max_new_tokens)


def answer_changed(dense_gen: str, drop_gen: str) -> int:
    return int(dense_gen.strip() != drop_gen.strip())


def dense_attention_last_query(model, input_ids: torch.Tensor, layer: int = -1) -> torch.Tensor:
    """Attention probabilities at the last query position.
    layer = -1 → max over heads then max over layers (any head/layer attends strongly);
                 else specific layer index, mean over heads.
    Returns (S,) summing to 1 after renormalization.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, output_attentions=True)
    if layer == -1:
        # max across heads (per-layer), then max across layers
        per_layer = [a[0, :, -1, :].max(0).values.float() for a in out.attentions]
        x = torch.stack(per_layer, 0).max(0).values
    else:
        x = out.attentions[layer][0, :, -1, :].mean(0).float()
    return (x / x.sum().clamp_min(1e-12)).cpu()


def irrelevant_positions(tok, prompt_info: PromptInfo) -> List[int]:
    """Positions of true filler tokens ('the', ',', '.', etc.) — NOT numbers, NOT key task words."""
    ids = prompt_info.input_ids[0].tolist()
    out = []
    num_set = {p for grp in prompt_info.number_token_positions for p in grp}
    for i, tid in enumerate(ids):
        if i == 0 or i in num_set:
            continue
        s = tok.decode([tid]).strip().lower()
        if s in IRRELEVANT_WORDS:
            out.append(i)
    return out
