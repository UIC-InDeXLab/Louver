"""
AIME evaluation runner.

Downloads AIME 2024 problems from HuggingFace datasets (Maxwell-Jia/AIME_2024)
and runs with DeepSeek-R1-Distill-Llama-8B (or any reasoning model).

Scoring: parse integer answer from generation, compare to ground truth.
Answer is a 3-digit integer (000–999).

Usage:
    python eval/aime.py --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --method louver_ta --max_new_tokens 8192
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are a helpful math competition assistant. "
    "Solve the problem step by step, then state your final answer as a single integer."
)

AIME_PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    "{problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
)

DEEPSEEK_PROMPT_TEMPLATE = (
    "<｜begin▁of▁sentence｜><｜User｜>{problem}<｜Assistant｜><think>\n"
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_aime_problems(year: int = 2024):
    """Load AIME problems. Falls back to hardcoded examples if dataset unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        problems = [{"problem": row["Problem"], "answer": str(row["Answer"])} for row in ds]
        return problems
    except Exception as e:
        print(f"Warning: could not load AIME dataset ({e}). Using fallback problems.", flush=True)
        # Minimal fallback for testing
        return [
            {
                "problem": "Find the number of positive integers n ≤ 1000 such that n^2 + n + 41 is divisible by 41.",
                "answer": "41",
            },
            {
                "problem": "What is the sum of digits of 10^2024 - 2024?",
                "answer": "18",
            },
        ]


# ── Scoring ──────────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract the final integer answer from model output."""
    text = text.strip()
    # Try: "The answer is X" / "= X" / boxed answer / last number
    patterns = [
        r"(?:answer|Answer)(?:\s+is)?\s*[:=]?\s*(\d{1,4})",
        r"\\boxed\{(\d{1,4})\}",
        r"(?:therefore|Thus|So)\s*,?\s*(?:the\s+answer\s+is\s*)?(\d{1,4})",
        r"(?:\*\*|__)(\d{1,4})(?:\*\*|__)\s*$",
        r"(\d{1,4})\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 <= val <= 999:
                return str(val)
    # Last number in text
    nums = re.findall(r"\b(\d{1,3})\b", text)
    if nums:
        return nums[-1]
    return None


def score_answer(pred_text: str, gold: str) -> float:
    pred = extract_answer(pred_text)
    return float(pred is not None and pred.lstrip("0") == gold.lstrip("0"))


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
    return AIME_PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, problem=problem)


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
        gap_search_frac=args.gap_search_frac,
        gap_topk=args.gap_topk,
    )


def make_baseline_cache(args, num_layers: int):
    if args.method == "twilight":
        return None  # uses standard DynamicCache
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    budget = args.budget_tokens
    if args.method == "h2o":
        from baselines.h2o import H2OCache
        heavy = max(1, int(budget * args.h2o_heavy_ratio))
        return H2OCache(heavy_budget=heavy, recent_budget=budget - heavy, num_layers=num_layers)
    raise ValueError(f"Unknown baseline: {args.method}")


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--method", default="louver_ta",
                        choices=["dense_sdpa", "dense_eager", "louver_ta", "h2o", "twilight"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max_problems", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--output_dir", default="results/aime")
    # Louver
    parser.add_argument("--louver_variant", default="ta", choices=["full", "ta"])
    parser.add_argument("--threshold_mode", default="oracle", choices=["oracle", "budget"])
    parser.add_argument("--oracle", default="sample_gap",
                        choices=["sample_max", "sample_mean_max", "sample_gap"])
    parser.add_argument("--budget_fraction", type=float, default=0.1)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--gap_search_frac", type=float, default=1.0)
    parser.add_argument("--gap_topk", type=int, default=3)
    # Baselines
    parser.add_argument("--budget_tokens", type=int, default=512)
    parser.add_argument("--h2o_heavy_ratio", type=float, default=0.5)
    # Twilight
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--twilight_skip_layers", type=int, default=2)
    args = parser.parse_args()

    BASELINE_METHODS = {"h2o", "twilight"}

    if args.method in BASELINE_METHODS:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import baselines.h2o
        import baselines.twilight
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

    problems = load_aime_problems(args.year)
    if args.max_problems is not None:
        problems = problems[:args.max_problems]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method.startswith("louver"):
        if args.oracle == "sample_gap":
            tag = f"louver_ta_gap_f{args.gap_search_frac}_k{args.gap_topk}"
        else:
            tag = f"louver_ta_{args.threshold_mode}_f{args.budget_fraction}"
    elif args.method == "twilight":
        tag = f"twilight_p{args.top_p}"
    elif args.method == "h2o":
        tag = f"h2o_b{args.budget_tokens}"
    else:
        tag = args.method
    model_tag = args.model.split("/")[-1]

    scores, records = [], []
    for i, prob in enumerate(tqdm(problems, desc="AIME")):
        prompt = make_prompt(prob["problem"], args.model)
        past_kv = None
        if args.method.startswith("louver"):
            past_kv = make_louver_cache(model.config, args)
        elif args.method in BASELINE_METHODS:
            past_kv = make_baseline_cache(args, num_layers)

        pred_text = generate_one(model, tokenizer, prompt,
                                 args.max_new_tokens, past_key_values=past_kv)
        sc = score_answer(pred_text, prob["answer"])
        scores.append(sc)
        extracted = extract_answer(pred_text)
        records.append({
            "idx": i, "problem": prob["problem"][:200],
            "gold": prob["answer"], "extracted": extracted,
            "score": sc, "output_len": len(pred_text),
            "generation": pred_text[:500],
        })
        print(f"  [{i+1}/{len(problems)}] gold={prob['answer']} pred={extracted} score={sc:.0f}", flush=True)

    acc = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "model": args.model, "method": args.method, "year": args.year,
        "accuracy": acc, "n": len(scores), "correct": int(sum(scores)),
        "avg_output_len": sum(r["output_len"] for r in records) / max(len(records), 1),
    }
    print(f"\nAIME {args.year} accuracy: {acc:.4f} ({int(sum(scores))}/{len(scores)})", flush=True)

    with open(output_dir / f"{model_tag}_{tag}_aime{args.year}.json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
