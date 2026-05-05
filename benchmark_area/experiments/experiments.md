# Experiments Plan

## Must-Have

### 1. Accuracy vs. Baselines on Long-Context Benchmarks
- **Benchmarks:** RULER, LongBench v2, NIAH variants
- **Models:** Llama 3.1 8B + one larger (70B or Qwen 2.5 72B)
- **Baselines:** H2O, StreamingLLM, MagicPIG, PQCache, RetrievalAttention, InfLLM, Quest, ClusterKV
- I. All baselines at comparable KV budget/fraction
- II. Louver with threshold methods 
- Key point: Louver wins because of zero false negatives

### 2. Latency vs. Sequence Length
- X-axis: N (8k → 128k), Y-axis: per-step decode latency (ms)
- Compare: Louver GPU, FlashAttention, Quest, ClusterKV, H2O
- Must show Louver faster than FlashAttention at large N


### 2.1. (new)
- Show AUC for speed vs. acc by changing the budgets.

### 3. Recall / False Negative Rate
- Show Louver = 100% recall, baselines < 100%
- Vary threshold τ or budget
- Core theoretical claim — must be empirically confirmed


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

### 8. Threshold Oracle Comparison
- SampleMax vs. SampleMeanMax vs. fixed τ
- Measure accuracy and latency for each

### 9. Buffer Size B Effect
- Update frequency vs. accuracy trade-off

---

## Motivating Experiment (intro / observations section)

### 10. Error Spike Demonstration
- Show one missing critical key → sharp output error
- Use a reasoning task (long chain-of-thought)
- Motivates the "zero false negatives" requirement with concrete numbers

---

## Priority Order

1 → 3 → 2 → 4 → 5 → 6 → 7 → 8 → 9 → 10
