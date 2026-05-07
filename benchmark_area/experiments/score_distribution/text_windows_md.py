"""Emit a markdown file with the decoded windows for the same set used by
plot_text_windows.py (W2, W3 wide; N1, N2 narrow). Each entry:

  ## W2 — Transition / synthesis (wide tail)
  - t = 1107
  - cov50_weight_mean = 0.0595
  ```text
  ... before tokens ...   [[target]]   ... after tokens ...
  ```
  Highlights: Per plan step ...   (claim-relevant phrases bolded.)

Uses side-car tokens.json for fast load.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WINDOWS = {
    "wide": [
        (1365, "W1", "Constraint check: \"Constraint (f) says at most 2 per day\""),
        (1460, "W2", "Pivot: \"Wait, no — each day can have up to 2 cities\""),
        (1648, "W3", "Recap: \"A, B, C, D, E, F, G — All are assigned…\""),
        (1767, "W4", "Pivot: \"Wait, no, Fri can have zero cities…\""),
    ],
    "narrow": [
        (1050, "N1", "\"Let me consider each subcase. **Subcase 1a:**…\""),
        (1265, "N2", "\"Let's try A on Mon. Then B can be on…\""),
        (1557, "N3", "\"Let's explore both. **Subsubcase 1a1:**…\""),
        (1822, "N4", "\"So this subsubcase is invalid. Wait, no, let me recount…\""),
    ],
}

CLAIM_PATTERNS = [
    r"Wait,?\s", r"Actually,?\s", r"Hmm", r"reconsider", r"let me\s",
    r"on second thought", r"recount", r"going back",
    r"\bSubcase\b", r"\bSubsubcase\b",
    r"\d+\s*[\+\-\*/=]\s*\d+",
]


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
    ap.add_argument("--snap", default=str(ROOT / "snapshots" / "snap_dsr14b_2k.pt"))
    ap.add_argument("--metrics", default=str(ROOT / "reports" / "tail_metrics_dsr.csv"))
    ap.add_argument("--metric", default="cov50_weight_mean")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "reports" / "text_windows.md"))
    args = ap.parse_args()

    rows = load_csv(args.metrics)
    by_t = {int(r["token_index"]): r for r in rows
            if isinstance(r.get(args.metric), (int, float))}

    side = Path(args.snap).with_suffix(".tokens.json")
    d = json.loads(side.read_text())
    gen = d["generated_tokens"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(d["model_name"])

    def context(t, w):
        before = "".join(tok.decode([gen[i]]) for i in range(max(0, t - w), t))
        target = tok.decode([gen[t]])
        after = "".join(tok.decode([gen[i]])
                        for i in range(t + 1, min(len(gen), t + w + 1)))
        return before, target, after

    def find_claims(s):
        hits = []
        for pat in CLAIM_PATTERNS:
            for m in re.finditer(pat, s, re.IGNORECASE):
                hits.append(m.group(0))
        return sorted(set(hits))

    lines = ["# Decoded windows for W1 / W2 / N1 / N2\n",
             f"\nMetric: `{args.metric}`. Window: ±{args.window} tokens. ",
             "Target token marked `[[…]]`. Highlights = claim-relevant phrases.\n"]

    for kind, label in [("wide", "Transition / Synthesis (wide tail)"),
                        ("narrow", "Local / Template (narrow tail)")]:
        lines.append(f"\n## {label}\n")
        for ti, lab, hint in WINDOWS[kind]:
            r = by_t.get(ti, {})
            v = r.get(args.metric, float("nan"))
            before, target, after = context(ti, args.window)
            full = before + target + after
            hits = find_claims(full)
            tag = target.replace("\n", "⏎").strip() or repr(target)
            block = before.replace("\n", "\n") + f"[[{target}]]" + after.replace("\n", "\n")

            lines.append(f"\n### {lab} — {hint}\n")
            lines.append(f"- `t = {ti}`,  `{args.metric} = {v:.4f}`,  target = `{tag}`\n")
            if hits:
                lines.append(f"- claim-phrases nearby: " +
                             ", ".join(f"**{h}**" for h in hits) + "\n")
            lines.append("\n```text\n")
            lines.append(block)
            lines.append("\n```\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(lines))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
