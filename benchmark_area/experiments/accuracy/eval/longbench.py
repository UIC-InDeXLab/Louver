"""
LongBench v1 evaluation runner.

Downloads dataset from HuggingFace if not cached, runs generation on each task,
scores with task-specific metric. Outputs a JSON results file.

Usage:
    python eval/longbench.py --model meta-llama/Llama-3.1-8B-Instruct \
        --method louver_ta --variant oracle --tasks hotpotqa,2wikimqa
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Task config ──────────────────────────────────────────────────────────────

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

# Tasks that use F1 scoring; others use EM or ROUGE
F1_TASKS = {"narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
            "2wikimqa", "musique", "triviaqa", "samsum"}

# Task-specific prompt templates (from LongBench official repo)
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


# ── Scoring ──────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    common = Counter(p_toks) & Counter(g_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(p_toks)
    recall = n_common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_prediction(pred: str, answers: list[str], task: str) -> float:
    if task in F1_TASKS:
        return max(f1_score(pred, a) for a in answers)
    # EM for the rest
    norm_pred = _normalize(pred)
    return float(any(_normalize(a) in norm_pred for a in answers))


# ── Generation ───────────────────────────────────────────────────────────────

def load_model(model_name: str, attn_impl: str, dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def make_louver_cache(model_config, args):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from louver_hf import LouverCache
    return LouverCache(
        model_config=model_config,
        variant=args.louver_variant,
        threshold_mode=args.threshold_mode,
        oracle=args.oracle,
        budget_fraction=args.budget_fraction,
        sample_size=args.sample_size,
        update_interval=args.update_interval,
        gap_search_frac=args.gap_search_frac,
        gap_topk=args.gap_topk,
    )


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, past_key_values=None, max_input_length: int = 32768):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=past_key_values,
            use_cache=True,
        )
    generated = output[0, input_len:]
    pred = tokenizer.decode(generated, skip_special_tokens=True).strip()
    pred = pred.split("\n")[0].strip()
    return pred


SHORT_ANSWER_TASKS = {"hotpotqa", "2wikimqa", "musique", "triviaqa",
                      "multifieldqa_en", "narrativeqa", "qasper", "trec",
                      "passage_count", "passage_retrieval_en"}


# ── Main ─────────────────────────────────────────────────────────────────────

def build_prompt(example: dict, task: str) -> str:
    template = DATASET2PROMPT.get(task, "{context}\n\n{input}")
    return template.format(
        context=example.get("context", ""),
        input=example.get("input", ""),
    )


def run_task(model, tokenizer, task: str, args, max_samples: int = None):
    ds = load_dataset(DATASET_NAME, task, split="test")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    scores, preds = [], []
    for example in tqdm(ds, desc=task):
        prompt = build_prompt(example, task)
        answers = example["answers"] if isinstance(example["answers"], list) else [example["answers"]]
        max_gen = MAX_GEN_TOKENS.get(task, 64)

        past_kv = None
        if args.method.startswith("louver"):
            past_kv = make_louver_cache(model.config, args)

        pred = generate_one(model, tokenizer, prompt, max_gen, past_key_values=past_kv,
                            max_input_length=args.max_input_length)
        if task in SHORT_ANSWER_TASKS:
            pred = pred.split(". ")[0].strip()
        score = score_prediction(pred, answers, task)
        scores.append(score)
        preds.append({"question": example.get("input", "")[:200], "pred": pred, "gold": answers, "score": score})

    return {"task": task, "n": len(scores), "avg": sum(scores) / len(scores), "preds": preds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", default="dense_sdpa",
                        choices=["dense_sdpa", "dense_eager", "louver_full", "louver_ta"])
    parser.add_argument("--tasks", default=",".join(ALL_TASKS))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="results/longbench")
    # Louver options
    parser.add_argument("--louver_variant", default="ta", choices=["full", "ta"])
    parser.add_argument("--threshold_mode", default="oracle", choices=["oracle", "budget"])
    parser.add_argument("--oracle", default="sample_max", choices=["sample_max", "sample_mean_max", "sample_gap"])
    parser.add_argument("--gap_search_frac", type=float, default=1.0)
    parser.add_argument("--gap_topk", type=int, default=3)
    parser.add_argument("--budget_fraction", type=float, default=0.1)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--update_interval", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=32768)
    args = parser.parse_args()

    attn_impl = "eager"
    if args.method == "dense_sdpa":
        attn_impl = "sdpa"
    elif args.method.startswith("louver"):
        attn_impl = f"louver_{args.louver_variant}"
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import louver_hf.attention  # register AttentionInterface

    model, tokenizer = load_model(args.model, attn_impl)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.threshold_mode == "oracle":
        oracle_tag = args.oracle if args.oracle != "sample_gap" else f"gap_f{args.gap_search_frac}_k{args.gap_topk}"
        tag = f"{args.method}_oracle_{oracle_tag}"
    else:
        tag = f"{args.method}_{args.threshold_mode}_f{args.budget_fraction}"
    model_tag = args.model.split("/")[-1]

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in ALL_TASKS]
    all_results = {}
    task_bar = tqdm(tasks, desc=f"{args.method}/{args.threshold_mode}", unit="task", position=0)
    for task in task_bar:
        task_bar.set_postfix(current=task)
        result = run_task(model, tokenizer, task, args, args.max_samples)
        all_results[task] = result["avg"]
        print(f"  {task}: {result['avg']:.4f}", flush=True)
        out_file = output_dir / f"{model_tag}_{tag}_{task}.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

    summary = {"model": args.model, "method": args.method, "scores": all_results,
               "avg": sum(all_results.values()) / len(all_results) if all_results else 0}
    print(f"\nAvg: {summary['avg']:.4f}", flush=True)
    with open(output_dir / f"{model_tag}_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    from pathlib import Path
    main()
