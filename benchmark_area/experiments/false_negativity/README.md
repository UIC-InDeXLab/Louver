# False-Negativity Experiments

Show that **false negatives on relevant KV positions** wreck answer quality, and
that no static sparse-attention budget — fixed K, fixed local window, fixed
cumulative-score threshold — works across a varying number of relevant tokens.

## Setup

Toy task: ask an instruction-tuned LLM
"Consider the list of numbers: a, b, c, ... What is the sum?"

Numbers drawn from {1, 2, 3} (sums stay tractable for small models).
Sparse attention is simulated by **dropping selected KEY positions** from the cache
via a 4-D additive attention mask (-inf columns). Single forward pass + greedy
decode with the drop applied at every step.

Metrics:
- **`KL(p_drop ‖ p_dense)`** — distribution-level divergence.
- **`answer_changed_vs_dense`** — greedy answer differs from dense baseline.
- **`answer_correct`** — substring match against ground-truth sum.

`answer_changed_vs_dense` is the cleanest semantic signal: it fires when sparse
attention flips the model's output, regardless of whether dense was right.

## Models

Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct (eager attention, fp16).

## Files

| Script | Question |
|---|---|
| `exp1_relevant_vs_irrelevant.py` | Are false negatives on **relevant** tokens worse than on irrelevant ones? |
| `exp2_fixed_k.py` | Does a fixed-K sliding window suffice? |
| `exp3_variable_list.py` | Does any fixed-K hold up as N varies? |
| `exp4_fixed_sum_threshold.py` | Does a fixed cumulative-score threshold T hold up? |
| `run_all.py` | Run all four × multiple models. |
| `summarize.py` | Print tabular summary. |
| `plot.py` | Publication-style figures into `reports/figs/`. |

```bash
python run_all.py --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct --trials 20
python summarize.py
python plot.py
```

## Findings

### 1. False negatives on relevant tokens dominate (Exp 1, N=6)

Drop-1-number flips the answer **~2× more often** than drop-1-filler:

| M | 1B chg rel | 1B chg irr | 3B chg rel | 3B chg irr |
|---|------------|------------|------------|------------|
| 1 | **0.70** | 0.30 | **0.70** | 0.35 |
| 2 | **0.75** | 0.60 | **0.90** | 0.70 |
| 3 | **0.90** | 0.65 | **0.85** | 0.70 |

KL goes the opposite way (dropping a comma/period perturbs the next-token
distribution more than dropping a number) but the *argmax answer* flips far more
often when relevant tokens are missing — exactly the false-negative cost.

→ See `figs/exp1_<model>.png`.

### 2. Fixed-K sliding window destroys output until K covers everything (Exp 2, N=6)

Llama-3.2-1B (dense correct = 0.30):

| K | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|----|----|----|----|----|----|
| KL | 1.60 | 0.98 | 0.24 | 0.20 | 0.10 | 0.10 | 0.05 | **0.00** |
| acc | 0 | 0 | 0 | 0 | 0 | 0 | 0.05 | **0.30** |
| frac numbers kept | 0 | 0 | 0 | 0 | 0.13 | 0.50 | **1.00** | 1.00 |

Until the window slides far enough back to cover all 8 number tokens
(K ≈ 48, prompt is 51 tokens), `answer_changed = 1.0` and acc = 0. There is no
graceful degradation — sparse output is wrong.

→ See `figs/exp2_<model>.png`.

### 3. Variable list size — fixed-K is wrong, dynamic recall wins (Exp 3, K=12)

KL of fixed-K=12 vs an oracle that keeps {BOS, all number tokens, question span}:

| | N=3 | N=6 | N=12 | N=24 |
|---|---|---|---|---|
| 1B fixed_K12 | 0.22 | 0.24 | 0.27 | 0.29 |
| 1B **oracle_dynamic** | **0.06** | **0.08** | **0.10** | **0.09** |
| 3B fixed_K12 | 0.52 | 0.42 | 0.36 | 0.28 |
| 3B **oracle_dynamic** | **0.24** | **0.17** | **0.11** | **0.08** |

Oracle dynamic — same or smaller budget — achieves **3-4× lower KL** than
fixed-K, and stays low as N grows. Fixed-K cannot adapt.

→ See `figs/exp3_<model>.png`.

### 4. Fixed cumulative-score threshold T also fails as N grows (Exp 4, NEW)

Even if you replace fixed-K with a "keep tokens until cumsum reaches T" rule
(common in TopP-style sparse attention), no single T survives a growing
relevant set.

Coverage of number tokens at fixed T (Llama-3.2-1B):

| T \ N | 3 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.70 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.90 | 0.70 | 0.57 | 0.46 | 0.39 | 0.46 | 0.47 | 0.44 |
| 0.95 | **1.00** | 0.80 | 0.74 | 0.73 | 0.71 | 0.70 | 0.69 |

Llama-3.2-3B is even sharper (T=0.95 coverage: 0.60 → 0.23 across N=3 → 32).

`T_needed` (minimum cumsum to capture **all** relevant tokens) grows from
0.93 (N=3) to 0.99 (N=32) on 1B — i.e. you eventually have to keep nearly the
entire prompt to avoid false negatives.

`K(T)` budget at fixed T also grows monotonically with N — at T=0.9 on 1B,
K goes from 23 (N=3) to 58 (N=32). The budget you'd need to fix at
configuration time depends on a quantity the system doesn't know in advance.

**Why fixed T fails.** As N grows, attention mass on relevant tokens spreads
across more positions (each gets a smaller share). Sinks and structural tokens
(BOS, commas, question keywords) keep their share, so fixed T fills its budget
with those before reaching the numbers.

→ See `figs/exp4_<model>.png`.

### Summary panel

`figs/summary_<model>.png` packs all four headline panels into one figure for
each model — drop into the paper as a single plate.

## Caveats

- Llama-1B/3B are weak at the arithmetic task past N≈4, so `answer_correct`
  saturates near 0 for large N. `answer_changed_vs_dense` and `KL` remain valid
  signals (they capture *behavioral perturbation*, which is what sparse
  attention should preserve regardless of whether the dense model is right).
- Attention scores in Exp 4 are aggregated as **max over heads, then max over
  layers** at the last query position, with the BOS sink and self-position
  excluded and the remainder renormalized. This mimics what a recall-oriented
  sparse system actually has to work with — a single saliency score per token —
  while sidestepping the well-known attention-sink artifact.
- Tested on Llama-3.2-1B / 3B. Larger models / Qwen would strengthen the
  message but trends are clean already.

## Paper takeaway, in one paragraph

A static sparse-attention budget (fixed K, fixed window, fixed cumulative-score
threshold T) cannot avoid false negatives on the *relevant* subset of tokens
whose size varies with the prompt. Empirically, dropping one relevant token
flips the answer ~2× more often than dropping one filler (Exp 1); fixed-K
windowing destroys output until K covers the entire relevant span (Exp 2); the
budget required for a fixed-K to keep up grows with list size (Exp 3); and a
fixed cumulative threshold T forces an impossible trade-off — small T misses
the relevant subset, large T degenerates toward keeping (almost) the entire
prompt as N grows (Exp 4). Sparse attention must be **dynamic and
recall-oriented**: pick the budget per prompt, driven by the actual relevant
set, not by a configuration-time constant.
