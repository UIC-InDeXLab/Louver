"""Capture Q/K snapshot for one long reasoning generation. Reuse via from_file."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXED_K_CHAL = ROOT.parent.parent / "fixed_k_chal"
sys.path.insert(0, str(FIXED_K_CHAL))

from helpers import ObserveAttentionHelper  # noqa: E402


REASONING_PROMPT = """You are an extremely careful long-form reasoner who **thinks aloud and revises**.

Two style requirements you MUST follow throughout:

(A) PLANNING + PIVOTING. First, write an explicit numbered plan ("Plan step 1: ...", etc). When executing each step, refer back to the plan ("Per plan step N:"). At least four times during the response, **write a pivot block** that begins with one of: "Wait, " / "Actually, " / "Hmm, let me reconsider — " / "On second thought, ". In each pivot block, explicitly point out what you missed or want to revise, and update the plan if needed (using a "Plan revision:" header). Pivot blocks must be substantive, not cosmetic.

(B) EXPLICIT ARITHMETIC. Whenever a numeric quantity appears, compute it on its own line in the form `LHS = a OP b = result`, fully expanded, e.g. `cost = 3 * 5 = 15`, `total = 4 + 7 + 2 = 13`. Do not skip the LHS. For sums of more than three terms, show the running partial sums explicitly: `1 + 2 = 3`, `3 + 4 = 7`, `7 + 5 = 12`, etc. Re-derive each total at least once for verification.

Problem (six parts, all required):

Part 1. A logistics company schedules deliveries for 7 cities (A, B, C, D, E, F, G) over 5 days (Mon-Fri).
Constraints:
  (a) Each city has exactly one delivery on a unique (city, day) pair.
  (b) City A is strictly earlier in the week than City B.
  (c) City C and City D cannot share a day, and both must be on Tue or Thu.
  (d) City E must be on Wed or later.
  (e) City F must be on the day immediately after City G.
  (f) At most 2 cities per day.
  (g) City B is not on Friday.
Enumerate all valid schedules and count them. Show the case analysis. Verify the count by an independent method.

Part 2. For each valid schedule from Part 1, assign delivery cost = day_index * city_letter_index where Mon=1..Fri=5, A=1..G=7. Find the cost-minimizing schedule(s). Show every cost computation, then verify by a second computation.

Part 3. Generalize. Suppose we have N cities and D days, with the analogue of constraints (b)–(g) where applicable. Derive a counting recurrence T(N, D). Compute T(7, 5), T(8, 5), T(8, 6) using the recurrence and cross-check against a brute-force argument.

Part 4. Symmetry and equivalence classes. Identify which valid schedules from Part 1 are related by relabeling-of-day permutations that respect the constraints. Group them into equivalence classes and count the classes.

Part 5. Algorithmic. Write Python-style pseudocode for a backtracking solver for Part 1 with explicit pruning rules. Trace its execution on the first three branches it explores. Estimate its time complexity in the worst case and on the actual instance.

Part 6. Critique. Suppose constraint (e) were dropped. How does the count change? Suppose instead constraint (f) were strengthened to "F is exactly two days after G". Recompute the count.

Do all six parts. Reason aloud. Correct yourself when wrong. Cite earlier steps by number. Conclude only after part 6.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "snapshots" / "snap.pt"))
    ap.add_argument("--prompt_file", default=None)
    args = ap.parse_args()

    prompt = REASONING_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text()

    helper = ObserveAttentionHelper(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        load_model=True,
    )
    text = helper.run_model(prompt)
    print("\n=== generated text ===\n")
    print(text)
    print("\n=== /generated ===\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    helper.save(args.out)
    # tiny side-car: tokens + model_name (for fast windows reports)
    import json
    sidecar = Path(args.out).with_suffix(".tokens.json")
    sidecar.write_text(json.dumps({
        "model_name": args.model,
        "prompt_length": helper.prompt_length,
        "generated_tokens": helper.generated_tokens,
        "all_token_ids": helper.all_token_ids,
    }))
    print(f"side-car saved to {sidecar}")
    # also save plain decoded text for analysis
    txt_path = Path(args.out).with_suffix(".txt")
    txt_path.write_text(text)
    print(f"text saved to {txt_path}")
