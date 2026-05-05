"""
Smoke test: verify LouverCache + attention patch works end-to-end.
Uses Llama-3.2-1B-Instruct (smallest available) for speed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = (
    "Below is a long passage followed by a question.\n\n"
    + ("The quick brown fox jumps over the lazy dog. " * 200)
    + "\n\nQuestion: What animal jumps over the dog?\nAnswer:"
)


def run_dense(model_name, prompt):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return text


def run_louver(model_name, prompt, variant, threshold_mode, budget_fraction=0.1):
    import louver_hf.attention  # register AttentionInterface
    from louver_hf import LouverCache

    attn_impl = f"louver_{variant}"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation=attn_impl,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    cache = LouverCache(
        model_config=model.config,
        variant=variant,
        threshold_mode=threshold_mode,
        oracle="sample_max",
        budget_fraction=budget_fraction,
        sample_size=128,
        update_interval=64,
    )

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            past_key_values=cache, use_cache=True,
        )
    text = tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return text


def main():
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL}")
    print(f"Prompt length: ~{len(PROMPT.split())} words\n")

    print("── Dense SDPA ─────────────────────────────")
    dense_out = run_dense(MODEL, PROMPT)
    print(f"Output: {dense_out!r}\n")

    configs = [
        ("ta",   "oracle",  0.0),
        ("ta",   "budget",  0.2),
        ("full", "oracle",  0.0),
        ("full", "budget",  0.2),
    ]

    for variant, mode, frac in configs:
        label = f"louver_{variant} / {mode}" + (f" f={frac}" if mode == "budget" else "")
        print(f"── {label} ─────────────────────────────")
        try:
            out = run_louver(MODEL, PROMPT, variant, mode, frac)
            match = "fox" in out.lower() or "quick" in out.lower() or out.strip()
            print(f"Output: {out!r}")
            print(f"Status: {'OK' if out.strip() else 'EMPTY'}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
