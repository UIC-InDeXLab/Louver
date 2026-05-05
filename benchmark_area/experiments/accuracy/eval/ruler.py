"""
RULER evaluation runner.

Generates synthetic RULER tasks (NIAH single/multi, VT, CWE, QA) and runs inference.
Re-uses the existing ruler_loader.py from benchmark_area/kv_sampling if available,
otherwise generates tasks from scratch.

Usage:
    python eval/ruler.py --model meta-llama/Llama-3.1-8B-Instruct \
        --method louver_ta --seq_len 32768 --tasks niah_single,niah_multi,vt
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HIRA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HIRA_ROOT))

RULER_LOADER = HIRA_ROOT / "benchmark_area" / "kv_sampling" / "ruler_loader.py"

ALL_TASKS = ["niah_single", "niah_multi", "vt", "cwe", "qa"]


# ── Task generators ──────────────────────────────────────────────────────────

_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "The water is clear. The moon is bright. "
)


def _make_niah_single(seq_len: int, tokenizer, n_samples: int = 50):
    needle = "The special magic number is: {n}"
    question = "What is the special magic number?"
    samples = []
    for _ in range(n_samples):
        n = random.randint(10000, 99999)
        needle_str = needle.format(n=n)
        filler_tokens = tokenizer.encode(_FILLER * 200, add_special_tokens=False)
        needle_tok = tokenizer.encode(" " + needle_str, add_special_tokens=False)
        budget = seq_len - len(needle_tok) - 100
        filler_tok = (filler_tokens * 20)[:budget]
        mid = len(filler_tok) // 2
        ctx_tok = filler_tok[:mid] + needle_tok + filler_tok[mid:]
        ctx = tokenizer.decode(ctx_tok)
        prompt = f"{ctx}\n\nQuestion: {question}\nAnswer:"
        samples.append({"prompt": prompt, "answer": str(n)})
    return samples


def _make_niah_multi(seq_len: int, tokenizer, n_samples: int = 50, n_needles: int = 3):
    question = "List all special magic numbers you found."
    samples = []
    for _ in range(n_samples):
        numbers = [random.randint(10000, 99999) for _ in range(n_needles)]
        needles = [f"Magic number {i+1}: {n}" for i, n in enumerate(numbers)]
        filler_tokens = tokenizer.encode(_FILLER * 200, add_special_tokens=False)
        needle_toks = [tokenizer.encode(" " + nd, add_special_tokens=False) for nd in needles]
        total_needle = sum(len(t) for t in needle_toks)
        budget = seq_len - total_needle - 100
        filler_tok = (filler_tokens * 20)[:budget]
        positions = sorted(random.sample(range(len(filler_tok)), n_needles))
        ctx_tok = []
        prev = 0
        for pos, nd_tok in zip(positions, needle_toks):
            ctx_tok.extend(filler_tok[prev:pos])
            ctx_tok.extend(nd_tok)
            prev = pos
        ctx_tok.extend(filler_tok[prev:])
        ctx = tokenizer.decode(ctx_tok)
        answer = ", ".join(str(n) for n in numbers)
        prompt = f"{ctx}\n\nQuestion: {question}\nAnswer:"
        samples.append({"prompt": prompt, "answer": answer, "numbers": numbers})
    return samples


def _make_vt(seq_len: int, tokenizer, n_samples: int = 50):
    """Variable tracking: follow a chain of variable assignments."""
    samples = []
    for _ in range(n_samples):
        depth = 5
        vars_ = [f"var_{random.randint(1000,9999)}" for _ in range(depth + 1)]
        assignments = [f"{vars_[i+1]} = {vars_[i]}" for i in range(depth)]
        init_val = random.randint(1000, 9999)
        assignments = [f"{vars_[0]} = {init_val}"] + assignments
        filler_tokens = tokenizer.encode(_FILLER * 200, add_special_tokens=False)
        assign_tok = tokenizer.encode(" ".join(assignments), add_special_tokens=False)
        budget = seq_len - len(assign_tok) - 100
        filler_tok = (filler_tokens * 20)[:budget]
        mid = len(filler_tok) // 2
        ctx_tok = filler_tok[:mid] + assign_tok + filler_tok[mid:]
        ctx = tokenizer.decode(ctx_tok)
        prompt = f"{ctx}\n\nWhat is the value of {vars_[-1]}?\nAnswer:"
        samples.append({"prompt": prompt, "answer": str(init_val)})
    return samples


def _make_cwe(seq_len: int, tokenizer, n_samples: int = 50):
    """Common words extraction: find most frequent word in long text."""
    samples = []
    words = ["apple", "banana", "cherry", "dragon", "echo", "forest"]
    for _ in range(n_samples):
        target = random.choice(words)
        filler_tokens = tokenizer.encode(_FILLER * 200, add_special_tokens=False)
        target_toks = tokenizer.encode(f" {target}", add_special_tokens=False)
        n_inserts = 20
        budget = seq_len - n_inserts * len(target_toks) - 100
        filler_tok = (filler_tokens * 20)[:budget]
        positions = sorted(random.sample(range(len(filler_tok)), n_inserts))
        ctx_tok = []
        prev = 0
        for pos in positions:
            ctx_tok.extend(filler_tok[prev:pos])
            ctx_tok.extend(target_toks)
            prev = pos
        ctx_tok.extend(filler_tok[prev:])
        ctx = tokenizer.decode(ctx_tok)
        prompt = f"{ctx}\n\nWhat is the most frequently repeated special word?\nAnswer:"
        samples.append({"prompt": prompt, "answer": target})
    return samples


TASK_GENERATORS = {
    "niah_single": _make_niah_single,
    "niah_multi": _make_niah_multi,
    "vt": _make_vt,
    "cwe": _make_cwe,
}


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_ruler(pred: str, sample: dict, task: str) -> float:
    pred = pred.strip().lower()
    if task == "niah_single":
        return float(sample["answer"] in pred)
    if task == "niah_multi":
        return sum(str(n).lower() in pred for n in sample["numbers"]) / len(sample["numbers"])
    if task in ("vt", "cwe"):
        return float(sample["answer"] in pred)
    return 0.0


# ── Model & generation ────────────────────────────────────────────────────────

def load_model(model_name: str, attn_impl: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


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
    )


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 32, past_key_values=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=131072).to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            past_key_values=past_key_values, use_cache=True,
        )
    generated = output[0, input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="dense_sdpa",
                        choices=["dense_sdpa", "dense_eager", "louver_full", "louver_ta"])
    parser.add_argument("--tasks", default=",".join(ALL_TASKS))
    parser.add_argument("--seq_len", type=int, default=32768)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="results/ruler")
    parser.add_argument("--seed", type=int, default=42)
    # Louver
    parser.add_argument("--louver_variant", default="ta", choices=["full", "ta"])
    parser.add_argument("--threshold_mode", default="oracle", choices=["oracle", "budget"])
    parser.add_argument("--oracle", default="sample_max", choices=["sample_max", "sample_mean_max"])
    parser.add_argument("--budget_fraction", type=float, default=0.1)
    parser.add_argument("--sample_size", type=int, default=256)
    args = parser.parse_args()

    random.seed(args.seed)

    attn_impl = "sdpa" if args.method == "dense_sdpa" else "eager"
    if args.method.startswith("louver"):
        attn_impl = f"louver_{args.louver_variant}"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import louver_hf.attention

    model, tokenizer = load_model(args.model, attn_impl)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_{args.threshold_mode}_f{args.budget_fraction}_L{args.seq_len}"
    model_tag = args.model.split("/")[-1]

    all_results = {}
    for task_name in args.tasks.split(","):
        task_name = task_name.strip()
        if task_name not in TASK_GENERATORS:
            print(f"Skipping unknown task {task_name!r}", flush=True)
            continue

        samples = TASK_GENERATORS[task_name](args.seq_len, tokenizer, args.n_samples)
        scores = []
        for sample in tqdm(samples, desc=task_name):
            past_kv = None
            if args.method.startswith("louver"):
                past_kv = make_louver_cache(model.config, args)
            pred = generate_one(model, tokenizer, sample["prompt"],
                                max_new_tokens=32, past_key_values=past_kv)
            scores.append(score_ruler(pred, sample, task_name))

        avg = sum(scores) / len(scores)
        all_results[task_name] = avg
        print(f"  {task_name} (L={args.seq_len}): {avg:.4f}", flush=True)
        out_file = output_dir / f"{model_tag}_{tag}_{task_name}.json"
        with open(out_file, "w") as f:
            json.dump({"task": task_name, "seq_len": args.seq_len, "avg": avg,
                       "n": len(scores), "scores": scores}, f, indent=2)

    summary = {"model": args.model, "method": args.method, "seq_len": args.seq_len,
               "scores": all_results, "avg": sum(all_results.values()) / max(len(all_results), 1)}
    print(f"\nOverall avg: {summary['avg']:.4f}", flush=True)
    with open(output_dir / f"{model_tag}_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
