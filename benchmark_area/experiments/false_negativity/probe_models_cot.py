"""Probe with CoT prompting + longer max_new_tokens."""

import argparse
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import random_lists


def cot_prompt(numbers):
    s = ", ".join(str(n) for n in numbers)
    return (
        f"Consider the list of numbers: {s}. "
        f"What is the sum of the numbers? "
        f"Compute step by step, then write 'Final answer: <integer>' at the end."
    )


def chat_prompt(tok, numbers):
    msgs = [{"role": "user", "content": cot_prompt(numbers)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def find_final(text, target):
    m = re.search(r"final answer[^\d-]*(-?\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1) == target)
    # fallback: last integer in text
    nums = re.findall(r"-?\d+", text)
    if nums:
        return int(nums[-1] == target)
    return 0


def eval_model(model_name, Ns, trials, low, high, max_new=200):
    print(f"\n=== {model_name} ===", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
            low_cpu_mem_usage=True
        ).eval()
    except Exception as e:
        print(f"  load failed: {e}", flush=True)
        return
    rng = random.Random(0)
    for N in Ns:
        correct = 0
        ex = None
        for _ in range(trials):
            nums = random_lists(rng, N, low, high)
            target = str(sum(nums))
            text = chat_prompt(tok, nums)
            ids = tok(text, return_tensors="pt").to("cuda").input_ids
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ok = find_final(gen, target)
            correct += ok
            if not ok and ex is None:
                ex = (nums, target, gen.strip().replace("\n", " ")[:120])
        rate = correct / trials
        ex_s = f"  ex: nums={ex[0][:6]}{'...' if len(ex[0])>6 else ''} tgt={ex[1]} got={ex[2]!r}" if ex else ""
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
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()
    for m in args.models:
        eval_model(m, args.Ns, args.trials, args.low, args.high, args.max_new)
