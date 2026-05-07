"""
LongBench accuracy + offload profiling for:
  louver_offload  — Louver parents on GPU; KV on CPU; GPU filter + CPU→GPU transfer
  hnsw_offload    — RetrievalAttention: HNSW on CPU; KV on CPU; CPU search + transfer
  ivf_offload     — InfLLM: IVF on CPU; KV on CPU; CPU search + transfer
  lsh_offload     — MagicPIG: LSH on CPU; KV on CPU; CPU hash + transfer

Metrics per method:
  - LongBench F1 / EM (same scoring as accuracy Exp 1)
  - search_ms: avg GPU filter time (Louver) or CPU search time (baselines) per decode step per layer
  - transfer_ms: avg CPU→GPU KV transfer time per decode step per layer
  - gpu_mb: persistent GPU memory (parent centers for Louver; ~0 for baselines)

Budget: 15% of tokens (same as louver_ta_budget_f15 in accuracy experiments).

Usage:
    python run_longbench_offload.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --method louver_offload \
        --tasks hotpotqa,2wikimqa,musique,qasper,narrativeqa
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
_ACC  = _HERE.parent / "accuracy"

# Patch sys.path so imports find both accuracy/ and offload/
for _p in (str(_HERE), str(_ACC), str(_HERE.parents[2])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import types as _t
_hira = _t.ModuleType("hira")
_hira.__path__ = [str(_HERE.parents[3])]
_hira.__package__ = "hira"
sys.modules["hira"] = _hira

DATASET_NAME = "THUDM/LongBench"

ALL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]

MAX_GEN_TOKENS = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32,
    "gov_report": 512, "qmsum": 512, "multi_news": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128,
    "passage_count": 32, "passage_retrieval_en": 32,
    "lcc": 64, "repobench-p": 64,
}

F1_TASKS = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
            "2wikimqa", "musique", "triviaqa", "samsum"}

SHORT_ANSWER_TASKS = {"hotpotqa", "2wikimqa", "musique", "triviaqa",
                      "multifieldqa_en", "narrativeqa", "qasper", "trec",
                      "passage_count", "passage_retrieval_en"}

DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\nStory: {context}\n\nNow, answer the question based on the story as concisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nArticle: {context}\n\nAnswer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\".\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news passages.\n\nNews Passages:\n{context}\n\nNow, write a one-page summary of all the news passages above.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read through all the paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number.\n\nThe number of unique paragraphs:",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Your task is to find which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 3\", \"Paragraph 7\", etc.\n\nThe answer is:",
    "lcc": "Please complete the code given below.\n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:\n",
}

METHODS = ["louver_offload", "hnsw_offload", "ivf_offload", "lsh_offload"]
METHOD_ATTN = {
    "louver_offload": "louver_offload",
    "hnsw_offload":   "ann_offload",
    "ivf_offload":    "ann_offload",
    "lsh_offload":    "ann_offload",
}
BUDGET_FRACTION = 0.15


# ── Scoring ───────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(p_toks); r = n / len(g_toks)
    return 2 * p * r / (p + r)


def score_prediction(pred: str, answers: list[str], task: str) -> float:
    if task in F1_TASKS:
        return max(f1_score(pred, a) for a in answers)
    norm_pred = _normalize(pred)
    return float(any(_normalize(a) in norm_pred for a in answers))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, attn_impl: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto", attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── Cache factory ─────────────────────────────────────────────────────────────

def make_cache(method: str, model_config):
    if method == "louver_offload":
        import louver_offload as _lo
        return _lo.LouverOffloadCache(model_config, budget_fraction=BUDGET_FRACTION)
    else:
        ann_method = {"hnsw_offload": "hnsw", "ivf_offload": "ivf", "lsh_offload": "lsh"}[method]
        import ann_offload as _ao
        return _ao.ANNOffloadCache(model_config, method=ann_method,
                                   budget_fraction=BUDGET_FRACTION)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_one(model, tokenizer, prompt: str, max_new_tokens: int,
                 past_key_values=None, max_input_length: int = 32768) -> str:
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=max_input_length).to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, past_key_values=past_key_values, use_cache=True,
        )
    pred = tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
    return pred.split("\n")[0].strip()


# ── Task runner ───────────────────────────────────────────────────────────────

def build_prompt(example: dict, task: str) -> str:
    template = DATASET2PROMPT.get(task, "{context}\n\n{input}")
    return template.format(context=example.get("context", ""),
                           input=example.get("input", ""))


def run_task(model, tokenizer, task: str, method: str, max_samples: int | None):
    ds = load_dataset(DATASET_NAME, task, split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    scores = []
    preds  = []
    agg_search   = []
    agg_transfer = []
    gpu_mb = 0.0

    bar = tqdm(ds, desc=task, leave=True, dynamic_ncols=True)
    for example in bar:
        prompt  = build_prompt(example, task)
        answers = example["answers"] if isinstance(example["answers"], list) else [example["answers"]]
        max_gen = MAX_GEN_TOKENS.get(task, 64)

        cache = make_cache(method, model.config)
        pred  = generate_one(model, tokenizer, prompt, max_gen,
                             past_key_values=cache)
        if task in SHORT_ANSWER_TASKS:
            pred = pred.split(". ")[0].strip()

        scores.append(score_prediction(pred, answers, task))
        preds.append({"question": example.get("input", "")[:200], "pred": pred,
                      "gold": answers, "score": scores[-1]})

        stats = cache.aggregate_stats()
        agg_search.append(stats["search_ms"])
        agg_transfer.append(stats["transfer_ms"])
        gpu_mb = stats["gpu_mb"]
        last_cache = cache

        avg_so_far = sum(scores) / len(scores)
        bar.set_postfix(
            acc=f"{avg_so_far:.3f}",
            srch=f"{stats['search_ms']:.2f}ms",
            xfer=f"{stats['transfer_ms']:.2f}ms",
        )

    n = max(len(scores), 1)
    return {
        "task": task, "n": len(scores),
        "avg": sum(scores) / n,
        "search_ms":   round(sum(agg_search) / n, 4),
        "transfer_ms": round(sum(agg_transfer) / n, 4),
        "gpu_mb":      gpu_mb,
        "preds": preds,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--method", default="louver_offload", choices=METHODS)
    p.add_argument("--tasks", default="hotpotqa,2wikimqa,musique,qasper,narrativeqa,gov_report,trec,triviaqa")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--max_input_length", type=int, default=32768)
    return p.parse_args()


def main():
    args = parse_args()

    # Register attention implementations
    attn_impl = METHOD_ATTN[args.method]
    if args.method == "louver_offload":
        import louver_offload  # noqa: F401 — registers "louver_offload" AttentionInterface
    else:
        import ann_offload     # noqa: F401 — registers "ann_offload" AttentionInterface

    model, tokenizer = load_model(args.model, attn_impl)

    output_dir = args.output_dir or (Path(__file__).parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.split("/")[-1]
    tag = f"{args.method}_budget_f{BUDGET_FRACTION}"

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ALL_TASKS]
    all_scores = {}
    all_timing = {}
    global_gpu_mb = 0.0
    last_cache = None

    task_bar = tqdm(tasks, desc=args.method, unit="task")
    for task in task_bar:
        task_bar.set_postfix(current=task)
        result = run_task(model, tokenizer, task, args.method, args.max_samples)
        all_scores[task] = result["avg"]
        all_timing[task] = {"search_ms": result["search_ms"],
                            "transfer_ms": result["transfer_ms"]}
        global_gpu_mb = result["gpu_mb"]
        print(f"  {task}: acc={result['avg']:.4f}  "
              f"search={result['search_ms']:.3f}ms  "
              f"transfer={result['transfer_ms']:.3f}ms", flush=True)
        with open(output_dir / f"{model_tag}_{tag}_{task}.json", "w") as f:
            json.dump(result, f, indent=2)

    n_tasks = max(len(all_scores), 1)
    summary = {
        "model":        args.model,
        "method":       args.method,
        "budget_frac":  BUDGET_FRACTION,
        "scores":       all_scores,
        "avg_accuracy": round(sum(all_scores.values()) / n_tasks, 4),
        "timing":       all_timing,
        "avg_search_ms":   round(sum(r["search_ms"]   for r in all_timing.values()) / n_tasks, 4),
        "avg_transfer_ms": round(sum(r["transfer_ms"] for r in all_timing.values()) / n_tasks, 4),
        "gpu_mb":       global_gpu_mb,
    }
    # Estimate full KV memory for reference (same model, same context length)
    try:
        cfg = model.config.get_text_config(decoder=True)
        n_layers = cfg.num_hidden_layers
        h_kv     = cfg.num_key_value_heads
        d_head   = cfg.hidden_size // cfg.num_attention_heads
        # Use observed avg n_stored from first layer as proxy for N
        sample_layer = next(
            (l for l in getattr(last_cache, "layers", []) if hasattr(l, "_n_stored") and l._n_stored > 0),
            None,
        )
        n_ctx = sample_layer._n_stored if sample_layer else args.max_input_length
        full_kv_mb = n_layers * h_kv * n_ctx * d_head * 2 * 2 / 1e6  # fp16 K+V
        summary["full_kv_mb_estimate"] = round(full_kv_mb, 1)
    except Exception:
        full_kv_mb = 0.0
        summary["full_kv_mb_estimate"] = 0.0

    print(f"\n── Summary: {args.method} ──")
    print(f"  Accuracy:     {summary['avg_accuracy']:.4f}")
    print(f"  Search:       {summary['avg_search_ms']:.3f} ms/step")
    print(f"  Transfer:     {summary['avg_transfer_ms']:.3f} ms/step")
    print(f"  GPU memory:   {summary['gpu_mb']:.1f} MB  (persistent; full KV ≈ {full_kv_mb:.0f} MB)")
    with open(output_dir / f"{model_tag}_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results → {output_dir}")


if __name__ == "__main__":
    main()
