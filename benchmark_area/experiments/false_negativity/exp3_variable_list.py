"""Experiment 3: list size varies → fixed-K is wrong; need dynamic, recall-oriented sparse attn."""

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
    out_path = out_dir / "exp3_variable_list.csv"

    fields = [
        "trial", "N", "S", "method", "K_eff",
        "kl", "top1_agrees_with_dense",
        "answer_correct_base", "answer_correct_drop", "answer_changed_vs_dense",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for trial in range(args.trials):
            for N in args.N_list:
                numbers = random_lists(rng, N, 1, 3)
                info = make_prompt_info(tok, numbers, args.device)
                S = info.input_ids.shape[1]
                base = forward_with_drop(model, info.input_ids, [])
                dense_gen = greedy_string(model, tok, info.input_ids, [])
                base_correct = int(info.answer_str in dense_gen)

                num_groups = info.number_token_positions
                num_pos_all = {p for grp in num_groups for p in grp}
                first_num = min(num_pos_all)
                last_num = max(num_pos_all)
                # structural context: BOS + prefix "Consider the list of numbers:" + question span
                prefix = set(range(0, first_num))
                question_span = set(range(last_num + 1, S))
                always_keep = prefix | question_span | {0}

                methods = []
                # 1) dense
                methods.append(("dense", [], S))
                # 2) fixed-K budget on RELEVANT (number) tokens — sparse-attention picks K of N
                for K in args.K_list:
                    if K >= len(num_groups):
                        kept_groups = num_groups
                    else:
                        kept_groups = rng.sample(num_groups, K)
                    kept_num = {p for grp in kept_groups for p in grp}
                    keep = always_keep | kept_num
                    methods.append((
                        f"fixed_K{K}",
                        [i for i in range(S) if i not in keep],
                        len(keep),
                    ))
                # 3) keep_all_relevant (oracle) — every number token + structural context
                keep = always_keep | num_pos_all
                methods.append((
                    "keep_all_relevant",
                    [i for i in range(S) if i not in keep],
                    len(keep),
                ))

                for name, drop, K_eff in methods:
                    dlogits = forward_with_drop(model, info.input_ids, drop) if drop else base
                    drop_gen = greedy_string(model, tok, info.input_ids, drop) if drop else dense_gen
                    row = dist_metrics(base, dlogits)
                    row["answer_correct_base"] = base_correct
                    row["answer_correct_drop"] = int(info.answer_str in drop_gen)
                    row["answer_changed_vs_dense"] = answer_changed(dense_gen, drop_gen)
                    row.update(trial=trial, N=N, S=S, method=name, K_eff=K_eff)
                    w.writerow(row)

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{args.trials}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--N_list", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 8, 10, 12, 16])
    ap.add_argument("--K_list", type=int, nargs="+",
                    default=[2, 3, 4, 6])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2)
    run(ap.parse_args())
