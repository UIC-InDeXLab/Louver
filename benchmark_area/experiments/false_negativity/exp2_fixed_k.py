"""Experiment 2: fixed-K sliding-window sparse attention fails when K < list span."""

import argparse
import csv
import random
from pathlib import Path

from common import (
    DEFAULT_MODEL,
    answer_changed,
    dist_metrics,
    forward_with_drop,
    greedy_string,
    load_model,
    make_prompt_info,
    random_lists,
)


def run(args):
    rng = random.Random(args.seed)
    tok, model = load_model(args.model, args.device)

    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp2_fixed_k.csv"

    fields = [
        "trial", "N", "S", "K", "kept_numbers",
        "kl", "top1_agrees_with_dense",
        "answer_correct_base", "answer_correct_drop", "answer_changed_vs_dense",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for trial in range(args.trials):
            numbers = random_lists(rng, args.N, 1, 3)
            info = make_prompt_info(tok, numbers, args.device)
            S = info.input_ids.shape[1]
            base = forward_with_drop(model, info.input_ids, [])
            dense_gen = greedy_string(model, tok, info.input_ids, [])
            base_correct = int(info.answer_str in dense_gen)

            for K in args.K_list:
                keep = set(range(max(1, S - K), S)) | {0}
                drop = [i for i in range(S) if i not in keep]
                kept_nums = sum(
                    1 for grp in info.number_token_positions
                    if all(p in keep for p in grp)
                )

                dlogits = forward_with_drop(model, info.input_ids, drop)
                drop_gen = greedy_string(model, tok, info.input_ids, drop)
                row = dist_metrics(base, dlogits)
                row["answer_correct_base"] = base_correct
                row["answer_correct_drop"] = int(info.answer_str in drop_gen)
                row["answer_changed_vs_dense"] = answer_changed(dense_gen, drop_gen)
                row.update(trial=trial, N=args.N, S=S, K=K, kept_numbers=kept_nums)
                w.writerow(row)

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{args.trials}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--K_list", type=int, nargs="+",
                    default=[6, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    run(ap.parse_args())
