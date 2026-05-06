# Experiments Plan

Ordered by priority.

## Must-Have

### 1. Accuracy vs. Baselines on Long-Context Benchmarks on same sparsity (`experiments/accuracy`)
- **Benchmarks:**
  - Long input: LongBench v1 (6 QA tasks) [DONE], RULER
  - Long output reasoning: AIME 2024 [], MATH-500 []
- **Models:**
  - Llama 3.1 8B-Instruct — LongBench, RULER
  - DeepSeek-R1-Distill-Llama-8B — AIME 2024, MATH-500 (long reasoning / CoT)
- **Baselines (implemented, same model + prompts as Louver):**
  - H2O — eviction by cumulative attention score (heavy hitters + recent)
  - StreamingLLM — eviction: sink tokens + recent window
  - Quest — retrieval: page-level sign(q)·max(k) scoring, top-K chunks
  - Twilight
- **Louver variants:** louver_ta, oracle threshold + budget (10%)
- I. All baselines at comparable KV budget/fraction (10%)
- II. Louver with threshold methods 
- All almost on the same budget ratio (output fraction)
- Key point: Louver wins because of zero false negatives


### 2. Latency vs. Sequence Length (`experiments/latency`)
- X-axis: N (8k → 128k), Y-axis: per-step decode latency (ms)
- Compare: Louver GPU, FlashAttention, Quest, H2O, Twilight
- Compare: on CPU
- Must show Louver faster than `FlashAttention` at large N
- Others to compare to `Twilight`
- **Dense baselines (both required):**
  - `dense_eager` — standard PyTorch eager attention (slowest, reference)
  - `dense_flash` — SDPA with FlashAttention backend (`SDPBackend.FLASH_ATTENTION`)
- Models: DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B
- Dataset: AIME sample, with very long output.

### 3. Recall / False Negative Rate
- Show Louver = 100% recall, baselines < 100%
- Vary threshold τ or budget
- Core theoretical claim — must be empirically confirmed

### 3.1. Offloading experiments
- make an offloading version of louver and compare with offloading baselines.
- RetrievalAttention and InfLLM.

### 3.2. Memory usage
- simple

---

## Important

### 4. Pruning Power vs. N
- Fraction of keys surviving filter across sequence lengths
- Confirms the ~90% pruning claim

### 5. Accuracy–Efficiency Trade-off Curve
- X-axis: fraction of keys retrieved (or budget), Y-axis: accuracy
- Louver's Pareto frontier vs. baselines
- Shows zero-FN recall guarantee translates to better accuracy per compute

### 6. CPU Experiments
- Louver CPU vs. SDPA-FP32 vs. Quest (if CPU version exists)
- Validates the CPU kernel claim

---

## Ablations (appendix)

### 7. Index Design Ablations
- Number of subspaces S
- Group size r
- Query 1 (full-subspace filter) vs. Query 2 (TA filter)

### 8. Threshold Oracle Ablation
- Louver-TA + Louver-Full × all oracles: sample_max, sample_meanmax, sample_gap, budget (fraction=0.1)
- Small subsample of same 6 QA tasks (10 examples/task)
- Report per variant: accuracy (avg F1) + fraction of tokens retrieved (mean ± std across layers/heads/steps/examples)
- Instrumentation: log retrieved/total tokens per decode step in LouverCacheLayer
- Goal: which oracle gives best accuracy vs sparsity trade-off? Justify oracle choice in main experiments

### 9. Buffer Size B Effect
- Update frequency vs. accuracy trade-off

---

## Other Experiments

### 11
- If time!
- Show AUC for speed vs. acc by changing the budgets.

---

## Notes
- compare to fixed budget
    - H2O, StreamingLLM, Quest
- compare to adaptive
    - Twilight
- compare to offloading
    - MagicPIG, RetrievalAttention, InfLLM
- compare to long input
    - RULER, LongBench
- compare to long output
    - MATH, AIME

### Others
- Add a bigger model: "DeepSeek-R1-Distill-Qwen-14B"