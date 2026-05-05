"""Print decoded windows around top-N highest cov50_weight_mean steps.

Excludes prefill and special/EOS tokens.

Usage:
  python report_windows.py                    # top-10 widest, ±20 token window
  python report_windows.py --top_n 5 --window 30
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_K_CHAL = ROOT.parent.parent / "fixed_k_chal"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FIXED_K_CHAL))


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k == "token_str":
                    continue
                try:
                    fv = float(v) if "." in v or "e" in v.lower() or v.lower() in ("nan", "inf", "-inf") else int(v)
                    if isinstance(fv, float) and (math.isnan(fv) or math.isinf(fv)):
                        r[k] = None
                    else:
                        r[k] = fv
                except ValueError:
                    pass
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=str(ROOT / "snapshots" / "snap_qwen2k.pt"))
    ap.add_argument("--metrics", default=str(ROOT / "reports" / "tail_metrics.csv"))
    ap.add_argument("--metric", default="cov50_weight_mean")
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "reports" / "windows.md"))
    args = ap.parse_args()

    rows = load_csv(args.metrics)
    if not rows:
        print("no rows"); return

    r0 = rows[0]
    prompt_len = int(r0["T"]) - int(r0["token_index"])

    filt = [r for r in rows
            if isinstance(r.get(args.metric), (int, float))
            and int(r["token_index"]) >= prompt_len
            and int(r.get("is_special", 0)) == 0]
    filt.sort(key=lambda r: -r[args.metric])
    wide = filt[: args.top_n]
    narrow = filt[-args.top_n :][::-1]  # smallest cov50, ascending → reversed for printing
    narrow = list(reversed(narrow))
    print(f"top-{args.top_n} WIDE by {args.metric} (prefill+EOS excluded):")
    for r in wide:
        print(f"  t={int(r['token_index']):5d}  {args.metric}={r[args.metric]:.4f}  T={int(r['T'])}")
    print(f"\ntop-{args.top_n} NARROW by {args.metric}:")
    for r in narrow:
        print(f"  t={int(r['token_index']):5d}  {args.metric}={r[args.metric]:.4f}  T={int(r['T'])}")

    # Fast path: side-car JSON if available; else fallback to full snapshot.
    import json
    from transformers import AutoTokenizer
    side = Path(args.snap).with_suffix(".tokens.json")
    if side.exists():
        d = json.loads(side.read_text())
        gen_tokens = d["generated_tokens"]
        model_name = d["model_name"]
    else:
        import torch
        data = torch.load(args.snap, map_location="cpu", weights_only=False)
        gen_tokens = data["generated_tokens"]
        model_name = data["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name)

    def context(t, w):
        parts = []
        for idx in range(max(0, t - w), min(len(gen_tokens), t + w + 1)):
            s = tok.decode([gen_tokens[idx]])
            parts.append(f"[[{s}]]" if idx == t else s)
        return "".join(parts)

    lines = [f"# Decoding-step windows  (metric: `{args.metric}`)\n\n",
             f"Prefill region (t < {prompt_len}) and special/EOS tokens excluded.\n",
             f"Window: ±{args.window} tokens. `[[token]]` marks the decoding step.\n"]

    def section(title, items):
        lines.append(f"\n## {title}\n")
        for r in items:
            ti = int(r["token_index"])
            lines.append(f"\n### t={ti}  ·  `{args.metric}={r[args.metric]:.4f}`  ·  T={int(r['T'])}\n\n")
            lines.append("```text\n")
            lines.append(context(ti, args.window))
            lines.append("\n```\n")

    section(f"Top-{args.top_n} WIDE (largest cov50 — broad attention)", wide)
    section(f"Top-{args.top_n} NARROW (smallest cov50 — focused / local attention)", narrow)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
