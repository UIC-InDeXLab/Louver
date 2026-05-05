# Score distribution observational study

Goal: show that across a long reasoning generation, **per-step attention tail
width varies** — some decoding steps focus on a small subset of past tokens
(local computation), others spread mass across many past tokens (planning /
pivoting). Build on prior result in `benchmark_area/fixed_k_chal`.

## Pipeline

1. **Capture once** (slow, model inference):
   ```bash
   python capture.py --model Qwen/Qwen2.5-7B-Instruct --max_new_tokens 512
   ```
   Saves `snapshots/snap.pt` (Q/K tensors per generated token) and `snap.txt`
   (decoded text). Reused by every downstream step.

2. **Score metrics** (fast, no model):
   ```bash
   python tail_metrics.py
   ```
   For every generated token t, builds attention probs `(H, T)` over selected
   layers (default = last quarter), aggregates per head, writes
   `reports/tail_metrics.csv`.

   Per step:
   - `topk_mass_{8,32,128}_{min,mean}`  — fraction of mass in top-k.
   - `eff_size_{max,mean}`              — `1 / Σ p²` (participation ratio).
   - `entropy_{max,mean}`               — Shannon entropy in nats.
   - `hi_cluster_{max,mean}`            — count of points in higher cluster from
     1D 2-means on log-prob (matches the "tail size" definition from
     `fixed_k_chal/tail_size_over_time.csv`).
   - `frac_above_unif_{max,mean}`       — fraction with p > 1/T.

   `*_max` aggregates worst-head; `*_mean` aggregates head-average. Worst-head
   is the right signal for sparse-attention budgeting (a static K must cover
   the broadest head).

3. **Analyze + plot**:
   ```bash
   python analyze.py --metric hi_cluster_max --top_n 8
   ```
   Picks top-N widest and narrowest steps by `--metric`, writes:
   - `figs/dist_wide.png`  — sorted attention probs (log) for the widest step.
   - `figs/dist_narrow.png` — same for the narrowest step.
   - `figs/tail_over_time.png` — line of metric over decoding time, picks
     marked.
   - `reports/windows.md`  — decoded ±W token context around each pick, so we
     can see which kinds of tokens trigger wide vs narrow attention.

## Notes

- Prior finding (in `fixed_k_chal/DONE.md`): **planning/pivot tokens attend to
  more history; local arithmetic tokens are focused**. This study reproduces
  that with cleaner metrics + isolates plotting from inference.
- Q/K are RoPE-applied at capture time (see `helpers.py::_apply_rope`), so
  scaled dot-product = pre-softmax logits.
- Sink/BOS handling is *not* applied here (full distribution kept) since
  observational. If sinks dominate `hi_cluster_*`, switch to `eff_size_max`
  or set `--metric topk_mass_32_min` (lower = wider tail).
