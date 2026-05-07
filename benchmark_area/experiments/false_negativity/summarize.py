"""Summarize per-model CSVs into printable tables."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

REP = Path(__file__).parent / "reports"


def load(p):
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                try: r[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                except ValueError: pass
            rows.append(r)
    return rows


def avg(rows, keys, m):
    b = defaultdict(list)
    for r in rows: b[tuple(r[k] for k in keys)].append(r[m])
    return {k: mean(v) for k, v in b.items()}


def fmt(x): return f"{x:.3f}" if isinstance(x, float) else str(x)


def banner(s): print("\n" + "=" * 60 + f"\n{s}\n" + "=" * 60)


def models_for(prefix):
    return sorted({p.name.split("__", 1)[1].replace(".csv", "")
                   for p in REP.glob(f"{prefix}__*.csv")})


def show_exp1():
    banner("EXP1: relevant vs irrelevant drops")
    for tag in models_for("exp1_relevant_vs_irrelevant"):
        rows = load(REP / f"exp1_relevant_vs_irrelevant__{tag}.csv")
        a_kl = avg(rows, ["drop_kind", "M"], "kl")
        a_ch = avg(rows, ["drop_kind", "M"], "answer_changed_vs_dense")
        a_ac = avg(rows, ["drop_kind", "M"], "answer_correct_drop")
        ms = sorted({m for (_, m) in a_kl})
        print(f"\n{tag}  (dense correct rate = {mean(r['answer_correct_base'] for r in rows):.2f})")
        print("M       " + "  ".join(f"{m:>6}" for m in ms))
        for label, src in [("KL rel ", a_kl), ("KL irr ", a_kl),
                           ("chg rel", a_ch), ("chg irr", a_ch),
                           ("acc rel", a_ac), ("acc irr", a_ac)]:
            kind = "relevant" if "rel" in label else "irrelevant"
            print(f"{label:<10}" + "  ".join(f"{src[(kind,m)]:6.3f}" for m in ms))


def show_exp2():
    banner("EXP2: fixed-K sliding window")
    for tag in models_for("exp2_fixed_k"):
        rows = load(REP / f"exp2_fixed_k__{tag}.csv")
        kl = avg(rows, ["K"], "kl")
        ac = avg(rows, ["K"], "answer_correct_drop")
        ch = avg(rows, ["K"], "answer_changed_vs_dense")
        kn = avg(rows, ["K"], "kept_numbers")
        Ks = sorted({k for (k,) in kl})
        print(f"\n{tag}  (dense correct = {mean(r['answer_correct_base'] for r in rows):.2f})")
        print("K        " + "  ".join(f"{k:>6}" for k in Ks))
        print("KL       " + "  ".join(f"{kl[(k,)]:6.3f}" for k in Ks))
        print("acc      " + "  ".join(f"{ac[(k,)]:6.3f}" for k in Ks))
        print("changed  " + "  ".join(f"{ch[(k,)]:6.3f}" for k in Ks))
        print("kept_n   " + "  ".join(f"{kn[(k,)]:6.2f}" for k in Ks))


def show_exp3():
    banner("EXP3: variable list size — dense vs fixed-K vs oracle dynamic")
    for tag in models_for("exp3_variable_list"):
        rows = load(REP / f"exp3_variable_list__{tag}.csv")
        a_kl = avg(rows, ["method", "N"], "kl")
        a_ac = avg(rows, ["method", "N"], "answer_correct_drop")
        a_ch = avg(rows, ["method", "N"], "answer_changed_vs_dense")
        Ns = sorted({n for (_, n) in a_kl})
        methods = sorted({m for (m, _) in a_kl})
        print(f"\n{tag}")
        print("              N=" + "  ".join(f"{n:>5}" for n in Ns))
        for m in methods:
            print(f"KL      {m:<13}" + "  ".join(f"{a_kl[(m,n)]:5.3f}" for n in Ns))
        for m in methods:
            print(f"acc     {m:<13}" + "  ".join(f"{a_ac[(m,n)]:5.3f}" for n in Ns))
        for m in methods:
            print(f"changed {m:<13}" + "  ".join(f"{a_ch[(m,n)]:5.3f}" for n in Ns))


def show_exp4():
    banner("EXP4: fixed cumsum threshold T — coverage of relevant tokens vs N")
    for tag in models_for("exp4_fixed_sum_threshold"):
        rows = load(REP / f"exp4_fixed_sum_threshold__{tag}.csv")
        a_cov = avg(rows, ["T", "N"], "coverage")
        a_K = avg(rows, ["T", "N"], "K_at_T")
        a_T = avg(rows, ["N"], "T_needed_for_full")
        Ns = sorted({n for (_, n) in a_cov})
        Ts = sorted({t for (t, _) in a_cov})
        print(f"\n{tag}")
        print("coverage   N=" + "  ".join(f"{n:>5}" for n in Ns))
        for T in Ts:
            print(f" T={T:.2f}    " + "  ".join(f"{a_cov[(T,n)]:5.2f}" for n in Ns))
        print("K_at_T     N=" + "  ".join(f"{n:>5}" for n in Ns))
        for T in Ts:
            print(f" T={T:.2f}    " + "  ".join(f"{a_K[(T,n)]:5.1f}" for n in Ns))
        print("T_needed   " + "  ".join(f"{a_T[(n,)]:.3f}" for n in Ns))


if __name__ == "__main__":
    show_exp1()
    show_exp2()
    show_exp3()
    show_exp4()
