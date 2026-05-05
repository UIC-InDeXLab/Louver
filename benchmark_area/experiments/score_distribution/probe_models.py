"""Probe candidate models with the reasoning prompt. Save decoded text only.

Usage: python probe_models.py
Outputs: snapshots/probe_<short_name>.txt
"""

import gc
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from capture import REASONING_PROMPT  # noqa: E402

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
]

OUT = ROOT / "snapshots"
OUT.mkdir(parents=True, exist_ok=True)


def short(name): return name.split("/")[-1].replace(".", "_")


def run(model_id, max_new=500):
    print(f"\n=== {model_id} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    # Use chat template if available
    msgs = [{"role": "user", "content": REASONING_PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = REASONING_PROMPT
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=False)
    path = OUT / f"probe_{short(model_id)}.txt"
    path.write_text(decoded)
    print(f"saved {path}")

    del model, tok, out, inputs
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for m in MODELS:
        try:
            run(m)
        except Exception as e:
            print(f"FAILED {m}: {e}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
