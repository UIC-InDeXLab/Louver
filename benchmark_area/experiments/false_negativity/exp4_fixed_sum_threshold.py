"""Experiment 4: a fixed sum-of-scores threshold T also fails as list size N grows.

For each prompt with N numbers:
  * Run dense forward; collect mean head-attention at the last query position
    (last layer by default).
  * Sort token positions by attention score, descending.
  * For each candidate T in {0.3,0.5,0.7,0.9,0.95}, find the smallest K(T)
    such that cumsum reaches T. Measure coverage = #number-tokens kept / N.
  * Also compute T_needed = minimum T whose top-K(T) covers ALL number tokens.

Show: coverage at fixed T drops as N grows, and T_needed approaches 1.0
(i.e. the threshold has to grow with N → fixed T is fundamentally wrong).
"""

import argparse
import csv
import random
from pathlib import Path

import torch

from common import (
    DEFAULT_MODEL,
    dense_attention_last_query,
    load_model,
    make_prompt_info,
    random_lists,
)

T_LIST = [0.3, 0.5, 0.7, 0.9, 0.95]


def run(args):
    rng = random.Random(args.seed)
    tok, model = load_model(args.model, args.device)

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp4_fixed_sum_threshold.csv"

    fields = ["trial", "N", "S", "layer", "T", "K_at_T", "coverage", "T_needed_for_full"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for trial in range(args.trials):
            for N in args.N_list:
                numbers = random_lists(rng, N, 1, 3)
                info = make_prompt_info(tok, numbers, args.device)
                S = info.input_ids.shape[1]
                num_pos = {p for grp in info.number_token_positions for p in grp}

                attn = dense_attention_last_query(model, info.input_ids, args.layer)
                # Exclude attention sinks (BOS + last query position itself); always keep them.
                sink_mask = torch.zeros_like(attn, dtype=torch.bool)
                sink_mask[0] = True
                sink_mask[-1] = True
                content = attn.clone()
                content[sink_mask] = 0.0
                content = content / content.sum().clamp_min(1e-12)
                # Sort positions by content score desc
                order = torch.argsort(content, descending=True).tolist()
                sorted_scores = content[order]
                cum = torch.cumsum(sorted_scores, dim=0)

                # T_needed: smallest cumsum such that all num_pos are in prefix
                T_needed = None
                seen = set()
                for rank, pos in enumerate(order):
                    if pos in num_pos:
                        seen.add(pos)
                    if len(seen) == len(num_pos):
                        T_needed = float(cum[rank].item())
                        break
                if T_needed is None:
                    T_needed = 1.0

                for T in T_LIST:
                    # smallest K such that cum[K-1] >= T
                    idx = int((cum >= T).nonzero(as_tuple=True)[0][0])
                    K = idx + 1
                    kept = set(order[:K])
                    cov = len(kept & num_pos) / max(len(num_pos), 1)
                    w.writerow(dict(
                        trial=trial, N=N, S=S, layer=args.layer,
                        T=T, K_at_T=K, coverage=cov,
                        T_needed_for_full=T_needed,
                    ))

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{args.trials}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--N_list", type=int, nargs="+",
                    default=[3, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    run(ap.parse_args())
