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
- X-axis: decode step (= N growing), Y-axis: per-step latency (ms)
- Must show Louver sub-linear vs. dense O(N)
- **GPU** (`gpu_bench.py`): louver, dense_eager, dense_flash, twilight
  - Twilight = full QK (flash) + top-p sort + full V — O(N), unavoidable
- **CPU** (`cpu_bench.py`): louver, dense_eager, torch_sdpa
- **Models:** Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct
- **Dataset:** one AIME 2024 problem, up to 40k generated tokens
- **Workflow:** `capture_all.sh` → saves `.pt` per model (~400–600 MB each, ~2 GB total) → `gpu_bench.py --input-qkv` + `cpu_bench.py --input-qkv`
- Reports: `latency/reports/gpu_bench_<model>.csv`, `cpu_bench_<model>.csv`

### 3. Recall / False Negative Rate (`experiments/recall`)
- Show Louver = 100% recall@k, all baselines < 100%
- k ∈ {10, 20, 50, 100} — fixed budget (number of retrieved keys)
- Core theoretical claim — empirically confirmed

**Phase 1 — Index recall** (Louver vs ANN methods, same index these offloading papers use):
  - Louver halfspace filter, HNSW [RetrievalAttention], IVF [InfLLM], PQ [PQCache], LSH [MagicPIG]

**Phase 2 — Sparse-attention recall** (Louver vs fixed-budget sparse-attn methods):
  - Louver (oracle threshold), Quest, StreamingLLM, Twilight

**Models:** Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct

**Implementation:** `recall_bench.py`
  - Input: same `.pt` captures from Exp 2 (`latency/captures/`)
  - ANN indices built once per KV head; 100 query samples, N=8k keys
  - GPU-accelerated exact score computation for ground truth top-k
  - Output: recall table + `reports/recall_<model>.csv`

**Workflow:** `bash run.sh` (comment in/out captures per machine)

### 3.1. Offloading experiments (`experiments/offload`)
Compare Louver offload vs MagicPIG (LSH), RetrievalAttention (HNSW), InfLLM (IVF) on LongBench.

**What each method does (all KV pairs on CPU):**
- **Louver**: parents (cluster centers from TA index) stay on GPU; GPU halfspace filter → selected token indices → gather children from CPU → transfer to GPU → SDPA
- **RetrievalAttention**: HNSW index on CPU; CPU HNSW search → gather top-k KV from CPU → transfer → SDPA
- **InfLLM**: IVF clustering on CPU; CPU IVF search → gather top clusters → transfer → SDPA
- **MagicPIG**: LSH hash table on CPU; CPU hash lookup → gather matching bucket KV → transfer → SDPA

**Budget:** 15% of tokens retrieved (budget_fraction=0.15, same as accuracy Exp 1)

**Metrics measured:**
- Accuracy: LongBench F1 (same tasks as Exp 1, Llama-3.1-8B-Instruct)
- Search time: GPU filter time (Louver) or CPU index search time (baselines) — ms/step
- Transfer time: CPU→GPU data movement for retrieved KV — ms/step
- GPU memory: persistent objects on GPU per layer (parent centers for Louver; ~0 for baselines)

**Implementation:** `offload/` directory — one file per method + `run_longbench_offload.py`

**Workflow:** `bash offload/run.sh`

---

## Important

### 4. Pruning Power vs. N
- Fraction of keys surviving filter across sequence lengths
- Confirms the ~90% pruning claim

---

## Ablations (appendix)

### 7. Threshold Oracle Ablation (`experiments/threshold_oracle`)
- Oracles: sample_max, sample_topk (k=2,5,10), sample_mean_max, sample_gap, budget (fraction=0.05/0.10/0.15)
- Report per variant: fraction of tokens retrieved (mean ± std) + recall@10% vs exact top-10% (mean ± std)
- Goal: which oracle gives best accuracy vs sparsity trade-off? Justify oracle choice in main experiments

**Input:** same `.pt` captures from Exp 2 (`latency/captures/`) — no model inference needed
- `meta_llama_Llama_3.2_3B_Instruct_layer14_N40000.pt` — Llama-3.2-3B-Instruct, layer 14, N=40k
- `Qwen_Qwen2.5_7B_Instruct_layer14_N40000.pt` — Qwen2.5-7B-Instruct, layer 14, N=40k
- `Qwen_Qwen2.5_14B_Instruct_layer24_N40000.pt` — Qwen2.5-14B-Instruct, layer 24, N=40k

**Dataset:** one AIME 2024 problem (same as Exp 2 captures), up to 40k generated tokens

**Workflow:** `bash threshold_oracle/run.sh` → `results/threshold_oracle_all.json` + per-model CSVs

### 8. Index Design Ablations
- Number of subspaces S
- Group size r
- Query 1 (full-subspace filter) vs. Query 2 (TA filter)

### 9. Buffer Size B Effect
- Update frequency vs. accuracy trade-off

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