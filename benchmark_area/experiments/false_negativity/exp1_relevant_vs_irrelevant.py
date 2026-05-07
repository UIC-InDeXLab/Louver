"""Experiment 1: false negatives on RELEVANT tokens hurt much more than on irrelevant ones.

Setup: list of N=8 numbers.
For each M in {1..N}, drop M positions:
  * relevant   — M chosen number tokens (one rep per number)
  * irrelevant — M random non-number prompt tokens
Aggregate over many random lists.
"""

import argparse
import csv
import random
from pathlib import Path

from common import (
    DEFAULT_MODEL,
    answer_changed,
    answer_match,
    dist_metrics,
    forward_with_drop,
    greedy_string,
    irrelevant_positions,
    load_model,
    make_prompt_info,
    random_lists,
)


def run(args):
    rng = random.Random(args.seed)
    tok, model = load_model(args.model, args.device)

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp1_relevant_vs_irrelevant.csv"

    fields = [
        "trial", "N", "M", "drop_kind",
        "kl", "top1_agrees_with_dense",
        "answer_correct_base", "answer_correct_drop", "answer_changed_vs_dense",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for trial in range(args.trials):
            numbers = random_lists(rng, args.N, 1, 3)
            info = make_prompt_info(tok, numbers, args.device)
            base = forward_with_drop(model, info.input_ids, [])
            dense_gen = greedy_string(model, tok, info.input_ids, [])
            base_correct = int(info.answer_str in dense_gen)

            irr_pool = irrelevant_positions(tok, info)
            num_rep = [grp[-1] for grp in info.number_token_positions]
            if len(irr_pool) < args.N:
                print(f"  trial {trial}: only {len(irr_pool)} irrelevant tokens, skipping high-M")

            M_max = min(args.N, len(irr_pool))
            for M in range(1, M_max + 1):
                for kind, drop in [
                    ("relevant", rng.sample(num_rep, M)),
                    ("irrelevant", rng.sample(irr_pool, M)),
                ]:
                    dlogits = forward_with_drop(model, info.input_ids, drop)
                    drop_gen = greedy_string(model, tok, info.input_ids, drop)
                    row = dist_metrics(base, dlogits)
                    row["answer_correct_base"] = base_correct
                    row["answer_correct_drop"] = int(info.answer_str in drop_gen)
                    row["answer_changed_vs_dense"] = answer_changed(dense_gen, drop_gen)
                    row.update(trial=trial, N=args.N, M=M, drop_kind=kind)
                    w.writerow(row)

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{args.trials}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
