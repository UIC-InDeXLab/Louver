# TA-filter index — design notes & roadmap

Pipeline at a decode step (per query-head set, GQA aware):

```
attend(q, T) = ta_filter_v8(q, T, state)  →  (live_idx, live_count)
             | sdpa_cuda_sparse_v2_5(q, K, V, buffer_K, buffer_V,
                                     live_idx, live_count, l_buf) →  out
```

Hardcoded constants: `bf=4`, `S=4`, `BUFFER_SIZE=256`. Top-k threshold
`T` is the kth-largest exact full-dim dot per head (one scalar per head).
`q_head_to_kv` is provided when GQA is active.

The index pre-allocates an arena with `K_cap` parent slots and `N_cap =
K_cap * BF` child slots. Initial build fills `[0:K_used]`; every flush
of the active 256-buffer cluster appends 64 parents → arena tail.

The active buffer K/V live in a small standalone `(H_kv, BUFFER_SIZE,
D)` tensor that is fed directly to sparse_attn v2.5.  v2.5 iterates
`live_count[hq] + l_buf` keys per head — the first `live_count[hq]`
indexed via `live_idx`, the trailing `l_buf` fetched from the buffer.

---

## Why sparse_attn v2.5 (and not v2.4 with an arena trick)

Empirical breakdown on Llama-3.2-3B, prefill=2230, l_buf=128:

| Path | filter | sparse | sum | attend | glue |
|------|-------:|-------:|----:|-------:|-----:|
| v2.5 buffer-aware (this index)        | 0.015 | 0.017 | 0.031 | **0.025** | **−0.006** |
| v2.4 with arena-tail buffer (dropped) | 0.015 | 0.009 | 0.023 | 0.058 | +0.035 |

v2.4-with-arena-tail had a leaner sparse_attn (no buffer branches) but
needed Python glue to copy `live_idx_filter → live_idx_attn` and
scatter buffer indices per head; the glue dominated.  v2.5 makes
`attend ≈ filter + sparse_attn` to within timing noise (the negative
glue on v2.5 is launch-pipelining: back-to-back launches overlap
slightly better than the two timed in isolation).

`bench_sparse_attn.py` keeps v2.4 as a microbench baseline only.

---

## Making `attend == filter_time + sparse_attn_time`

Already there for typical workloads.  Remaining glue sources:

| ID | Source                                   | Status                                |
|----|------------------------------------------|----------------------------------------|
| G1 | live_idx copy + buffer scatter           | **Eliminated** by v2.5.                |
| G2 | Workspace re-lookup per call             | Cached on instance; warm on build.     |
| G3 | Two kernel launches across separate libs | Open. (Optional) fuse filter + sparse_attn into a single coop kernel. |
| G4 | `centers_padded_f16.contiguous()` etc.   | Pre-call once at build time.           |
| G5 | `threshold.float().contiguous()`         | Cheap; can be skipped by upstream.     |
| G6 | Per-call CUDA launch latency             | Open.  CUDA-graph capture of the (filter, sparse_attn) pair, keyed by `(h_q, K_used, l_buf)`, invalidated on flush. |

After G6 the pair becomes a single replay → `attend ≈ filter +
sparse_attn` to within graph-replay overhead (~1 µs).

---

## Update kernel — design

Phase split (mirrors `kernels/update_v4_0` of the older index):

* **Phase 1 — async data scatter**: `update_v1_1.cu` clusters 256
  buffer keys into 64 parents per subspace and writes new center /
  assigns rows into the unused arena tail `[K_used : K_used + 64]`.
  Does NOT flip `invalid_mask` — filter on the attention stream still
  observes the range as invalid (sentinel-packed assigns).
* **Phase 2 — publish on attention stream**: flip invalid flags +
  publish packed assigns + bump `K_used / N_used`.

Async path: `index.update_async(fire_step)` records the cluster kernel
on a side stream; a later `wait_for_update()` (or `try_publish()`,
non-blocking) flushes phase 2 onto the attention stream after waiting
on the update_done event.

### `update_v1_1.cu` algorithm (current)

Block grid `(S=4, H_kv)`, threads=256. Per (s, h_kv):

1. Each thread `t` projects buffer key `t` onto axis = sum-of-w-dims.
2. CUB `BlockRadixSort<float, 256, 1, int>` sorts (proj, idx) ascending.
3. Cluster id = sorted_rank / BF (= /4).
4. Write 64 cluster centers `mean(4 keys per cluster)` into
   `centers_padded_f16[s, h, K_used:K_used+64, :w]`.
5. Write per-key `assigns_padded[s, h, N_used + b] = K_used + cluster_id`.

Steady-state ~0.12 ms/flush.  First call has ~22 ms JIT cost.

### Update — known limitations / future kernels

* **Cluster quality.** Single-axis sum projection clusters less tightly
  than `TA_build`'s recursive balanced PCA tree.  `bench_update.py`
  shows the incremental scan-fraction is higher than a fresh rebuild,
  especially right after the first flush:

      flush 1: scan_inc=0.65 vs scan_fresh=0.18  (Δ=+0.47)
      flush 2: scan_inc=0.44 vs scan_fresh=0.20  (Δ=+0.25)
      flush 3: scan_inc=0.34 vs scan_fresh=0.16  (Δ=+0.18)
      flush 4: scan_inc=0.18 vs scan_fresh=0.15  (Δ=+0.03)

  Follow-up: `update_v1_2.cu` could implement a 6-level PCA-tree
  (project, sort, split — recurse); cost still bounded since 256
  points / 64 clusters is small.

* **Telemetry / "was the update hidden?"**
    - `update_kernel_ms` — GPU kernel time on the side stream.
    - `update_wait_ms`   — host wall-time stalled in `wait_for_update`.
    - `update_inflight`  — was a prior update still running at step start?
    - End-of-run **hide ratio**: `1 − (Σ wait_ms) / (Σ kernel_ms)`.

  Current measurement on Llama-3.2-3B / 600 steps / 2 flushes:
  `hide_ratio = 100%`, `n_overlap_misses = 0/2`.  Update is fully
  parallel with decode — does not block.

---

## Phased roadmap

**Phase 1 — done.** Sync update, slim TA_build, end-to-end bench.

**Phase 2 — done.**
* `update_v1_1.cu` fast cluster kernel (CUB sort, 0.12 ms steady state).
* Async update path on side stream + telemetry.
* `sdpa_cuda_sparse_v2_5_fp16` (buffer-aware) — `attend ≈ filter + sparse_attn`.

**Phase 3 — open.**
* G3 / G6 — fuse filter + sparse_attn into a single launch (CUDA graph or
  cooperative kernel).
* `update_v1_2.cu` — multi-level PCA-tree clustering for tighter
  incremental clusters.
* TA_build itself: vectorise the per-`h_kv` Python loop in
  `_balanced_pca_tree_subspace`; current build is ~2.2 s on 2230 keys.
