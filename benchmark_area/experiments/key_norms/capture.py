"""Capture post-RoPE keys for a short prompt and save snapshot.

Reuses ObserveAttentionHelper from fixed_k_chal. Default model is
Qwen2.5-7B-Instruct (cheaper than DSR-14B); the experiment is purely
observational so a smaller model is fine.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_K_CHAL = ROOT.parent.parent / "fixed_k_chal"
sys.path.insert(0, str(FIXED_K_CHAL))

from helpers import ObserveAttentionHelper  # noqa: E402


PROMPT = """You are a careful technical writer.

Task: Explain the difference between dense attention and sparse attention in
2-3 short paragraphs. Mention that sparse attention drops keys/values from
the KV cache, that the choice of which keys to drop matters, and that
fixed-budget methods can miss relevant tokens. Keep the answer concise.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "snapshots" / "snap.pt"))
    args = ap.parse_args()

    helper = ObserveAttentionHelper(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        load_model=True,
    )
    text = helper.run_model(PROMPT)
    print("\n=== generated ===\n" + text + "\n=== /generated ===\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    helper.save(args.out)

    side = Path(args.out).with_suffix(".tokens.json")
    side.write_text(json.dumps({
        "model_name": args.model,
        "prompt_length": helper.prompt_length,
        "generated_tokens": helper.generated_tokens,
        "all_token_ids": helper.all_token_ids,
    }))
    print(f"snapshot -> {args.out}")
    print(f"side-car -> {side}")
