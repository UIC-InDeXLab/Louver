"""Probe candidate models for dense-baseline arithmetic accuracy on the sum task across N.

Reports correctness fraction at each N, with a small set of trials.
Goal: find a model with >=80% correct up to N=20.
"""

import argparse
import random
import sys
from statistics import mean

import torch

from common import greedy_string, load_model, make_prompt_info, random_lists


def eval_model(model_name, Ns, trials, low, high, max_new=6):
    print(f"\n=== {model_name} ===", flush=True)
    try:
        tok, model = load_model(model_name, "cuda")
    except Exception as e:
        print(f"  load failed: {e}", flush=True)
        return
    rng = random.Random(0)
    for N in Ns:
        correct = 0
        wrong_examples = []
        for _ in range(trials):
            nums = random_lists(rng, N, low, high)
            info = make_prompt_info(tok, nums, "cuda")
            gen = greedy_string(model, tok, info.input_ids, [], max_new)
            ok = info.answer_str in gen
            correct += int(ok)
            if not ok and len(wrong_examples) < 1:
                wrong_examples.append((nums, info.answer_str, gen.strip()[:30]))
        rate = correct / trials
        ex = wrong_examples[0] if wrong_examples else None
        ex_s = f"  ex_wrong: nums={ex[0][:8]}{'...' if len(ex[0])>8 else ''} sum={ex[1]} got={ex[2]!r}" if ex else ""
        print(f"  N={N:3d}  acc={rate:.2f}{ex_s}", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--low", type=int, default=1)
    ap.add_argument("--high", type=int, default=3)
    args = ap.parse_args()
    for m in args.models:
        eval_model(m, args.Ns, args.trials, args.low, args.high)
