"""
MATH-500 evaluation runner.

Downloads HuggingFaceH4/MATH-500 from HuggingFace, runs inference with a
reasoning model (DeepSeek-R1-Distill-Llama-8B), scores by parsing \\boxed{}.

Usage:
    python eval/math500.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --method louver_ta --max_new_tokens 4096
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "HuggingFaceH4/MATH-500"

DEEPSEEK_PROMPT_TEMPLATE = "<｜begin▁of▁sentence｜><｜User｜>{problem}<｜Assistant｜><think>\n"
GENERIC_PROMPT_TEMPLATE = "Solve the following math problem step by step.\n\nProblem: {problem}\n\nSolution:"


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.strip()
    # Strip sizing/display commands: \left, \right, \big*, \dfrac→\frac
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\s*", "", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    # Remove all whitespace
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    # Strip leading zeros on pure integers
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s or "0"


def extract_boxed(text: str) -> str | None:
    """Extract last \\boxed{...} content, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r"\boxed{", i)
        if idx == -1:
            break
        start = idx + len(r"\boxed{")
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1].strip())
        i = idx + 1
    if results:
        return results[-1]
    # Fallback: last number
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def score_answer(pred_text: str, gold: str) -> float:
    pred = extract_boxed(pred_text)
    if pred is None:
        return 0.0
    return float(_normalize(pred) == _normalize(gold))


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str, attn_impl: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_prompt(problem: str, model_name: str) -> str:
    if "deepseek" in model_name.lower() or "r1" in model_name.lower():
        return DEEPSEEK_PROMPT_TEMPLATE.format(problem=problem)
    return GENERIC_PROMPT_TEMPLATE.format(problem=problem)


def make_louver_cache(model_config, args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from louver_hf import LouverCache
    return LouverCache(
        model_config=model_config,
        variant=args.louver_variant,
        threshold_mode=args.threshold_mode,
        oracle=args.oracle,
        budget_fraction=args.budget_fraction,
        sample_size=args.sample_size,
        top_p=args.louver_top_p,
    )


def make_baseline_cache(args, num_layers: int):
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    budget = args.budget_tokens
    if args.method == "h2o":
        from baselines.h2o import H2OCache
        heavy = max(1, int(budget * args.h2o_heavy_ratio))
        return H2OCache(heavy_budget=heavy, recent_budget=budget - heavy, num_layers=num_layers)
    elif args.method == "quest":
        from baselines.quest import QuestCache
        return QuestCache(chunk_size=args.quest_chunk_size, token_budget=budget, num_layers=num_layers)
    elif args.method == "streaming_llm":
        from baselines.streaming_llm import StreamingLLMCache
        return StreamingLLMCache(sink_size=4, recent_size=budget - 4, num_layers=num_layers)
    elif args.method == "twilight":
        return None  # uses standard DynamicCache
    raise ValueError(f"Unknown method: {args.method}")


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, past_key_values=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
    generated = output[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

BASELINE_METHODS = {"h2o", "quest", "streaming_llm", "clusterkv", "twilight"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--method", default="dense_sdpa",
                        choices=["dense_sdpa", "dense_eager",
                                 "louver_ta",
                                 "h2o", "quest", "streaming_llm", "twilight"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--output_dir", default="results/math500")
    # Louver
    parser.add_argument("--louver_variant", default="ta", choices=["full", "ta"])
    parser.add_argument("--threshold_mode", default="oracle", choices=["oracle", "budget"])
    parser.add_argument("--oracle", default="sample_top_p",
                        choices=["sample_max", "sample_mean_max", "sample_top_p"])
    parser.add_argument("--budget_fraction", type=float, default=0.1)
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--louver_top_p", type=float, default=0.85)
    # Baselines
    parser.add_argument("--budget_tokens", type=int, default=512,
                        help="Fixed KV token budget for baseline methods")
    parser.add_argument("--h2o_heavy_ratio", type=float, default=0.5)
    parser.add_argument("--quest_chunk_size", type=int, default=16)
    # Twilight
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--twilight_skip_layers", type=int, default=2)
    args = parser.parse_args()

    # Register attention implementations
    if args.method in BASELINE_METHODS:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if args.method == "h2o":
            import baselines.h2o
        elif args.method == "quest":
            import baselines.quest
        elif args.method == "streaming_llm":
            import baselines.streaming_llm
        elif args.method == "twilight":
            import baselines.twilight
        elif args.method == "clusterkv":
            import baselines.clusterkv
        attn_impl = args.method
    elif args.method == "dense_sdpa":
        attn_impl = "sdpa"
    elif args.method == "dense_eager":
        attn_impl = "eager"
    else:
        attn_impl = f"louver_{args.louver_variant}"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import louver_hf.attention

    model, tokenizer = load_model(args.model, attn_impl)
    num_layers = model.config.num_hidden_layers

    if args.method == "twilight":
        from baselines.twilight import configure_twilight
        configure_twilight(top_p=args.top_p, skip_first_layers=args.twilight_skip_layers)

    ds = load_dataset(DATASET_NAME, split="test")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    if args.method.startswith("louver"):
        if args.oracle == "sample_top_p":
            tag = f"louver_ta_top_p{args.louver_top_p}"
        else:
            tag = f"louver_ta_{args.threshold_mode}_f{args.budget_fraction}"
    elif args.method == "twilight":
        tag = f"twilight_p{args.top_p}"
    elif args.method in BASELINE_METHODS:
        tag = f"{args.method}_b{args.budget_tokens}"
    else:
        tag = args.method

    scores, records = [], []
    for i, row in enumerate(tqdm(ds, desc=f"MATH-500/{args.method}")):
        problem = row["problem"]
        gold = str(row["answer"]).strip()
        prompt = make_prompt(problem, args.model)

        past_kv = None
        if args.method.startswith("louver"):
            past_kv = make_louver_cache(model.config, args)
        elif args.method in BASELINE_METHODS:
            past_kv = make_baseline_cache(args, num_layers)

        pred_text = generate_one(model, tokenizer, prompt, args.max_new_tokens, past_kv)
        sc = score_answer(pred_text, gold)
        scores.append(sc)
        extracted = extract_boxed(pred_text)
        records.append({
            "idx": i, "problem": problem[:200], "gold": gold,
            "extracted": extracted, "score": sc,
            "output_len": len(pred_text),
        })
        print(f"  [{i+1}/{len(ds)}] gold={gold} pred={extracted} score={sc:.0f}", flush=True)

    acc = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "model": args.model, "method": args.method,
        "accuracy": acc, "n": len(scores), "correct": int(sum(scores)),
        "avg_output_len": sum(r["output_len"] for r in records) / max(len(records), 1),
    }
    print(f"\nMATH-500 accuracy: {acc:.4f} ({int(sum(scores))}/{len(scores)})", flush=True)

    with open(output_dir / f"{model_tag}_{tag}_math500.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
