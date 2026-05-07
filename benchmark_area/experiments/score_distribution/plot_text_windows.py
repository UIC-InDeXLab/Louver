"""Render decoded windows as a paper-ready figure.

2x3 grid (top row = wide W1/W2/W3, bottom row = narrow N1/N2/N3) by default.
Per panel: title bar with id + cov50 value, then ±W decoded text with
the target token highlighted (yellow box) and claim-relevant phrases
emphasised (red bold).

Use side-car JSON `<snap>.tokens.json` for fast load (no .pt deserialize).
"""

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220,
    "font.family": "DejaVu Sans",
    "font.size": 12,
})


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


# default windows to render — 3 wide (W1-W3) + 3 narrow (N1-N3)
DEFAULT = {
    "wide": [
        (1460, "W1", "Pivot: \"Wait, no — at most 2 per day\""),
        (1900, "W2", "Pivot: \"Wait — re-read constraint\""),
    ],
    "narrow": [
        (1050, "N1", "\"So, four possibilities for G and F. Let me consider each…\""),
        (1822, "N2", "\"So this subsubcase is invalid…\""),
    ],
}

WIDE_COLOR = "#7c3aed"
NARROW_COLOR = "#0d9488"

# regexes that highlight CLAIM-relevant tokens — model behaviors that justify
# wide vs narrow attention.
CLAIM_PATTERNS = [
    r"Wait,?\s",
    r"Actually,?\s",
    r"Hmm",
    r"reconsider",
    r"let me\s",
    r"on second thought",
    r"recount",
    r"going back",
    r"\bSubcase\b", r"\bSubsubcase\b",
    r"\d+\s*[\+\-\*/=]\s*\d+",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=str(ROOT / "snapshots" / "snap_dsr14b_2k.pt"))
    ap.add_argument("--metrics", default=str(ROOT / "reports" / "tail_metrics_dsr.csv"))
    ap.add_argument("--metric", default="cov50_weight_mean")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--wrap", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "figs" / "text_windows.png"))
    args = ap.parse_args()

    rows = load_csv(args.metrics)
    by_t = {int(r["token_index"]): r for r in rows
            if isinstance(r.get(args.metric), (int, float))}

    side = Path(args.snap).with_suffix(".tokens.json")
    d = json.loads(side.read_text())
    gen = d["generated_tokens"]
    tok = AutoTokenizer.from_pretrained(d["model_name"])

    def context(t, w):
        before = "".join(tok.decode([gen[i]]) for i in range(max(0, t - w), t))
        target = tok.decode([gen[t]])
        after = "".join(tok.decode([gen[i]])
                        for i in range(t + 1, min(len(gen), t + w + 1)))
        return before, target, after

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    rows_def = [("wide", WIDE_COLOR), ("narrow", NARROW_COLOR)]
    for row_i, (kind, color) in enumerate(rows_def):
        for col_i, (ti, lab, hint) in enumerate(DEFAULT[kind]):
            ax = axes[row_i, col_i]
            render_panel(ax, ti, lab, hint, color, by_t, args, context)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"  -> {args.out}")


def render_panel(ax, ti, lab, hint, color, by_t, args, context_fn):
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.005, 0.005), 0.99, 0.99,
                                transform=ax.transAxes,
                                fill=False, edgecolor=color, linewidth=2.4))
    ax.add_patch(plt.Rectangle((0.005, 0.84), 0.99, 0.155,
                                transform=ax.transAxes,
                                facecolor=color, edgecolor=color))
    r = by_t.get(ti, {})
    v = r.get(args.metric, float("nan"))
    ax.text(0.5, 0.92, f"{lab} · {hint}", transform=ax.transAxes,
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="white")
    ax.text(0.5, 0.86, f"t={ti}    {args.metric}={v:.4f}",
            transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="white")

    before, target, after = context_fn(ti, args.window)

    def wrap(s):
        # Replace newlines with visible markers, then wrap to width
        s = s.replace("\n", " ⏎ ")
        return "\n".join(textwrap.wrap(s, width=args.wrap,
                                        drop_whitespace=False,
                                        replace_whitespace=False)) or " "

    before_w = wrap(before)
    after_w = wrap(after)

    # render with simple emphasis: underlay claim-pattern matches in red bold
    # via two passes — base text in dark, then overplot bold red markers
    body_top = 0.79

    # before text (gray) + claim highlights (red bold) using tokenised approach:
    # split text by claim regex
    def render_with_highlights(ax, text, x, y, fs=10, color_base="#1f2937",
                                color_hi="#dc2626"):
        # use simple word-by-word rendering with highlights — split into segments
        segments = _segment(text, CLAIM_PATTERNS)
        # render multi-line via single text with mathtext-free approach: stack
        # we do per-line: scan segments mapped onto wrapped text via simple
        # fallback — render whole block and overlay highlights using axhspan-
        # style is hard; instead use plain monospace and emit each line as text
        # with parts in different colors using `ax.text` per chunk on same line.
        # Simpler: render whole text first, then place red text on top of any
        # claim phrase positions found by re.finditer in the wrapped text.
        ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=fs, family="monospace", color=color_base)
        # overlay red bold on claim phrases by drawing the substring at the
        # same character position. Approximate by line-scanning.
        lines = text.split("\n")
        line_h = 0.040
        for li, line in enumerate(lines):
            for pat in CLAIM_PATTERNS:
                for m in re.finditer(pat, line, re.IGNORECASE):
                    # approximate horizontal offset by char count (monospace)
                    char_w = 0.0118  # works for our wrap width / figsize
                    px = x + m.start() * char_w
                    py = y - li * line_h
                    ax.text(px, py, m.group(0),
                            transform=ax.transAxes, ha="left", va="top",
                            fontsize=fs, family="monospace",
                            color=color_hi, fontweight="bold")

    render_with_highlights(ax, before_w, 0.04, body_top, fs=10)

    # target token (boxed yellow, bold, in panel color)
    target_disp = target.replace("\n", "⏎")
    if not target_disp.strip():
        target_disp = repr(target)
    ax.text(0.5, body_top - max(0.04, 0.04 * before_w.count("\n")) - 0.04,
            f"⟶  {target_disp.strip() or target_disp}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=14, family="monospace",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.30", fc="#fff7c2",
                      ec=color, lw=1.6))
    after_y = body_top - max(0.04, 0.04 * before_w.count("\n")) - 0.16
    render_with_highlights(ax, after_w, 0.04, after_y, fs=10)


def _segment(text, patterns):
    """Return list of (substring, is_highlight) pairs."""
    if not patterns:
        return [(text, False)]
    pat = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    out = []
    last = 0
    for m in pat.finditer(text):
        if m.start() > last:
            out.append((text[last:m.start()], False))
        out.append((m.group(0), True))
        last = m.end()
    if last < len(text):
        out.append((text[last:], False))
    return out


if __name__ == "__main__":
    main()
