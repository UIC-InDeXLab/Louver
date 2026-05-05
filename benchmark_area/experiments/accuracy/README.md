# Accuracy Experiments

## Models

| Benchmark | Model |
|---|---|
| LongBench v1, RULER | `meta-llama/Llama-3.1-8B-Instruct` |
| AIME 2024 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |

## Louver Variants

| Variant | Index | Threshold input |
|---|---|---|
| `louver_ta` | `TAIndex` (TA filter + sparse SDPA) | `(H_q,)` float32 per-head |
| `louver_full` | `SubspaceKCenterIndex` (full-subspace) | `(2*S, H_q)` fp16 packed |

## Threshold Modes

| Mode | Description | Option |
|---|---|---|
| `oracle` | Reservoir sample → SampleMax or SampleMeanMax | `--threshold_mode oracle --oracle sample_max` |
| `budget` | Fixed fraction f of tokens retrieved | `--threshold_mode budget --budget_fraction 0.1` |

## Baselines

| Method | Category | Model support |
|---|---|---|
| Dense SDPA (FlashAttention) | dense oracle | all |
| Dense Eager | dense oracle | all |
| H2O | eviction | Llama |
| Quest | retrieval, no offload | Llama-3.1 |
| ClusterKV | retrieval, no offload | Llama-3.1 |
| PQCache | retrieval, offload | Llama-3.1, Mistral |
| MagicPIG | retrieval, offload | Llama-3.1 |

DeepSeek-R1-Distill-Llama-8B: only dense baselines (Eager, SDPA) since all baselines use LLaMA-specific patches that are compatible, but need verification per baseline.

## Quick Start

```bash
cd benchmark_area/experiments/accuracy

# Dense oracle (FlashAttention)
python run_louver.py longbench --model meta-llama/Llama-3.1-8B-Instruct --method dense_sdpa

# Louver TA, oracle threshold
python run_louver.py longbench --model meta-llama/Llama-3.1-8B-Instruct --louver_variant ta --threshold_mode oracle

# Louver TA, 10% budget
python run_louver.py longbench --model meta-llama/Llama-3.1-8B-Instruct --louver_variant ta --threshold_mode budget --budget_fraction 0.1

# Louver full-subspace, oracle
python run_louver.py longbench --model meta-llama/Llama-3.1-8B-Instruct --louver_variant full --threshold_mode oracle

# RULER at 32k
python run_louver.py ruler --model meta-llama/Llama-3.1-8B-Instruct --louver_variant ta --seq_len 32768

# AIME with DeepSeek-R1
python run_louver.py aime --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --louver_variant ta
```

## File Structure

```
accuracy/
├── louver_hf/
│   ├── attention.py   # AttentionInterface: louver_full, louver_ta
│   ├── cache.py       # LouverCache, LouverCacheLayer, LouverCacheOutput
│   └── threshold.py   # LouverThreshold (budget + oracle modes)
├── eval/
│   ├── longbench.py   # LongBench v1 runner
│   ├── ruler.py       # RULER runner (synthetic task generation)
│   └── aime.py        # AIME 2024 runner
├── baselines/         # Baseline runners (TODO: per-method wrappers)
├── run_louver.py      # Unified CLI
└── README.md
```

## Results Layout

Results written to `results/{benchmark}/{model_tag}/` as JSON files:
- Per-task: `{model}_{method}_{tag}_{task}.json`
- Summary: `{model}_{method}_{tag}_summary.json`
