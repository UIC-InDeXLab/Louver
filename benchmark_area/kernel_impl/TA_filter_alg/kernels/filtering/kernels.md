# Filtering kernels (short description)

Names below are the **Python extension** entry points (`ext.*`).

## v1.0

- **`centroid_scores`** — CUDA: q·centers per subspace → `scores` (all **K** parents).
- **`torch.sort`** — CPU/GPU PyTorch sort on `scores` (not this extension).
- **`depth_selected`** — CUDA: row-wise sum across subspaces vs `threshold` → `depth` + `parent_mask`.
- **`alive_mask`** — CUDA: per key, any selected parent in `assigns` → dense `alive`.

## v1.1

- Same four steps as **v1.0**; before the first call, autotune picks
  `centroid_scores` block size and `alive_mask` thread count (cached per config).

## v2.0

- **`score_top_l`** — CUDA: fused per `(hq, s)`. Streams over all `K` parents, keeps
  top-`L` using per-thread heap + CUB block sort. Outputs
  `top_scores[Hq,S,L]` + `top_indices[Hq,S,L]`.
- **`depth_selected_v2`** — CUDA: scans sorted top-`L`, finds TA stop depth, writes
  `parent_mask[Hq,4,K]`.
- **`alive_mask`** — same as v1.x.

## v3.2

- **`depth_only`** — CUDA: computes TA depth from `top_scores`.
- **`alive_bm_tiled`** — CUDA tiled kernel (`Grid=(Hq, N_TILES)`) that rebuilds a
  shared-memory bitmask from `top_indices` + `depth` and writes dense alive mask.
- Reuses **`score_top_l`** from v2.0.

## v3.3

Apply Ideas 2 + 7 to v3.2's `alive_bm_tiled`:
- TILE_N: 1024 → 2048 (each block processes 8 keys/thread).
- int4-vectorised loads on `assigns` and `invalid_mask`. Layout switch: thread t owns CONTIGUOUS keys [t*8, t*8+8) within tile (was strided).
  - int16 assigns: 1 int4 (16 B) per subspace per thread = 8 keys.
  - int32 assigns: 2 int4 per subspace per thread = 8 keys.
  - invalid_mask: 1 int2 (8 B) per thread.
- Same total bandwidth as v3.2; ~8× fewer issued loads on the dominant memory path.
- Boundary fallback (last tile tail) uses scalar loads.
- Output: dense alive[Hq, Npad] int8. depth_only reused from v3.2.

## v3.4

Apply Idea 1 (packed assigns) to v3.2's `alive_bm_tiled`:
- One-time layout build in Python wrapper (cached in state):
  `assigns_packed[Hkv, Npad]` int64 with bits [0,16):p0, [16,32):p1, [32,48):p2, [48,64):p3.
- Sentinel 0xFFFF in any lane = invalid (folds invalid_mask into the packed lane; no separate load).
- Per-key: ONE 8-byte ld.global vs v3.2's 4× ld.s16 + 1× ld.s8.
- ~4× less assigns bandwidth.
- Specialised on int16 assigns (K ≤ 32767, sentinel safely outside valid parent range).
- depth_only reused from v3.2.

## v3.5

v3.4 + retuned `score_top_l` (BLOCK/IPT_L sweep).
- L=256 → BLOCK=256, IPT_L=1 (same as v2).
- L=512 → BLOCK=**512**, IPT_L=**1** (was BLOCK=256, IPT_L=2).
- L=1024 → BLOCK=**512**, IPT_L=**2** (was BLOCK=256, IPT_L=4).
- 2× more threads stream K → halves K-iters/thread, smaller per-thread heap, smaller CUB temp_storage.
- Reuses depth_only (v3.2) + alive_bm_tiled_v34 (v3.4 packed-assigns).

## v3.6

v3.4 + GQA-reuse `score_top_l`.
- Grid: `(Hkv, 4)` was `(Hq, 4)`. Each block processes ALL queries sharing one (kvh, s) in a single K stream — centers slab loaded ONCE per group.
- centers GMEM traffic drops `gqa_factor`× (Llama 3.5×, Qwen 7×).
- Block count drops by same factor; per block does `gqa_factor`× more work + per-query CUB sort.
- `q_groups[Hkv, G_MAX=8]` + `q_count[Hkv]` precomputed once and cached in `state`.
- Reuses depth_only (v3.2) + alive_bm_tiled_v34 (v3.4 packed-assigns).

## v5.0

v3.4 + Idea A (int4 vec on packed slab) + Idea B (TILE_N=2048).
- Layout: contiguous-per-thread (thread t owns keys [t*8, t*8+8)).
- 1 int4 = 16 B = 2 packed int64 keys.
- 4 int4 loads/thread = 8 keys/thread = 4 ld.b128 (was 4 ld.b64 for 4 keys).
- Halves issue rate per key. Same total bandwidth.
- TILE_N 1024 → 2048: halves redundant smem-bitmask builds (8 → 4 at N=8000).
- Boundary fallback for last-tile tail.
- Output: dense alive[Hq, Npad] int8. depth_only reused from v3.2.

## v5.1

v3.4 + Idea C (prebuilt global bitmask).
- New `depth_and_bm_global` (Hq grid): writes `depth[Hq]` AND `bm_global[Hq, 4*K_words]` in one kernel.
- New `alive_bm_global_tiled` ((Hq, N_TILES) grid, TILE_N=2048): linear-copy `bm_global[hq]` GMEM→smem (no atomicOr build), then int4-vec packed alive lookup (same as v5.0).
- Saves N_TILES-1 redundant smem-bitmask builds per query head (7-of-8 builds at N=8000 eliminated).
- Output: dense alive[Hq, Npad] int8.

## v4.0

- Compact-list output version (no dense alive output).
- Pipeline:
  - `score_top_l` (v2.0)
  - `depth_only` (v3.2)
  - `alive_compact_tiled` (v4.0)
- Outputs:
  - `live_idx[Hq, Npad]` int32 (first `live_count[hq]` entries valid)
  - `live_count[Hq]` int32

## v6.0

v3.4 fast path with **compact-list output** instead of dense alive.
- Same per-key cost as v3.4: 1 ld.b64 packed + 4 smem bitmask lookups + sentinel.
- Same TILE_N=1024, BLOCK=256, PER_THREAD=4, strided per-thread layout.
- Output: `live_idx[Hq, Npad]` int32 + `live_count[Hq]` int32 (zeroed per call).
- Compaction overhead: per-warp batched ballot (PER_THREAD=4 ballots → 1 atomicAdd-to-smem per warp); 1 atomicAdd-to-GMEM per (hq, tile) for global offset; coalesced smem-flush.
- Output bandwidth: ~3× less than v3.4 dense at frac≈0.08 (no zeros written).
- Consumed directly by `sdpa_cuda_sparse_v2_0` (skips its mask compaction).

## Notes

- Specialized for **S=4** and **bf=4**.

## Future ideas to push v6.x compact-list faster

v6.0 loses ~1.6 µs vs v3.4 dense in the alive kernel + 6.5 µs for `live_count.zero_()`. v6.1 fixes the zero (fused into depth) and lowers register pressure. Further levers:

### Idea G1 — Fuse zero into depth kernel (DONE in v6.1)
`depth_with_reset` writes both `depth[hq]` and `live_count[hq]=0`. Eliminates separate `cudaMemsetAsync` launch.

### Idea G2 — Inline-write compaction (DONE in v6.1)
Drop batched `ballot_arr/rank_arr/popc_arr` register state. Per-iter ballot → atomicAdd-smem → smem write. Lower register pressure, fewer barriers.

### Idea G3 — Skip smem buffer; direct-to-GMEM warp writes (DONE in v6.2)
Each warp does `atomicAdd(&live_count[hq], popc)` once per iter to claim a contiguous global range; lane writes its alive key directly to `live_idx[hq, off + rank]`. Drops `s_live[1024]` (4 KB smem) and final flush. Cost: more GMEM atomics on same address (per-hq contention). May win if smem-flush dominates.

### Idea G4 — Persistent counter in workspace (no zero needed)
Make `live_count` a per-call SLOT: workspace allocates `live_count[Hq, MAX_INFLIGHT]`. Caller picks a slot via a host counter. Filter writes to `live_count[Hq, slot]`; downstream consumes. Eliminates reset entirely (next call uses next slot, wraps around). Trade: workspace size × MAX_INFLIGHT.

### Idea G5 — Per-tile output slots; no global atomic
Output `live_idx[Hq, N_TILES, TILE_N]` + `live_count_per_tile[Hq, N_TILES]`. Each tile writes to its OWN slot (no atomic). Reset per-tile counts via depth kernel (Hq × N_TILES counts to zero). Trade: downstream attention needs per-tile iteration; output buffer slightly larger (gaps). Eliminates GMEM atomic entirely.

### Idea G6 — Warp-scan in registers, single block-wide write
Instead of smem buffer + atomicAdd, do a block-wide prefix sum on per-warp popcs (e.g. via `cub::BlockScan`) to get per-warp offsets without atomics. Then each lane writes directly to GMEM with deterministic offset. No smem buffer, no GMEM atomic per warp.

### Idea G7 — Larger TILE_N + int4 vec packed (Idea A applied to compact) (DONE in v6.2)
TILE_N=2048, contiguous-per-thread layout, 4 int4 loads/thread = 8 keys/thread. Halves load issues. Pairs with batched ballot. May win if v6.1 register pressure was the bottleneck.

### Idea G8 — Coalesce ballot writes via warp shuffle
After per-warp ballot+popc, use `__shfl_xor`/`__shfl_up` to compute lane's write position. Skip smem `s_warp_off`. Saves smem bank traffic.

### Recommended v6.2 candidate
Idea G3 + G7: direct-to-GMEM per-warp atomics + int4 vec + TILE_N=2048. Strict superset of v6.1 if GMEM atomic contention is acceptable.

## v7.0

v6.2 alive replaced by **G6 BlockScan compaction**:
- Per-iter ballot bits stored in registers (8 uints/thread).
- Per-warp popc accumulated → smem array `s_warp_pop[8]`.
- Warp 0 lanes [0, 8) do exclusive shfl scan → `s_warp_off[8]`.
- Lane 7 does ONE `atomicAdd(&live_count[hq], block_total)` → broadcast block_off via smem.
- All threads scatter alive keys to deterministic GMEM offsets.
- Drops 8 GMEM atomics/block (one per warp in v6.2) → 1 GMEM atomic/block.
- Drops `s_live[2048]` smem buffer + final flush.
- Same int4-vec packed loads, TILE_N=2048, contiguous-per-thread layout as v6.2.

## v7.1

**Idea C compact (global bm precompute)** for compact-list pipeline:
- New `depth_and_bm_global` (Hq grid): writes `depth[hq]` + zeros `live_count[hq]` + writes `bm_global[Hq, 4*K_words]` in ONE kernel. Replaces v6.1 `depth_with_reset`.
- New `alive_compact_bmg` ((Hq, N_TILES) grid, TILE_N=2048): linear-copy `bm_global[hq]` GMEM→smem (no atomicOr build), then v6.2-style per-warp atomic compaction.
- Eliminates redundant smem-bitmask builds: 5 builds/hq → 1 at N=8000.

## v7.2

**GQA-reuse `score_top_l`** (v3.6 idea applied to compact-list pipeline):
- Grid: `(Hkv, 4)` was `(Hq, 4)`. Each block processes ALL queries sharing one (kvh, s) — centers slab loaded ONCE per group.
- Per-thread heap state replicated per query in group (`heap_keys[G_MAX][IPT_L]`).
- Per-query CUB BlockRadixSort + write at end.
- centers GMEM traffic drops by `gqa_factor` (Llama 3×, Qwen 7×).
- `q_groups[Hkv, G_MAX=8]` + `q_count[Hkv]` precomputed once and cached.
- Reuses depth_with_reset (v6.1) + alive_compact_v62 (v6.2).

## v7.3

v7.0 + v7.1 combo: BlockScan compaction + global bm precompute.
- Pipeline: `score_top_l` (v2) → `depth_and_bm_global` (v7) → `alive_compact_bmg_bs` (v7).
- Linear-copy bm_global→smem (no per-tile atomicOr build).
- BlockScan compaction (1 GMEM atomic/block instead of 8).

## v7.4

Grand combo (v7.0 + v7.1 + v7.2):
- `score_top_l_gqa` (v7.2) + `depth_and_bm_global` (v7.1) + `alive_compact_bmg_bs` (v7.3).

## v7.5

`score_top_l_b512` (BLOCK=512, IPT_L=1@L=512 / IPT_L=2@L=1024) + `depth_with_reset` (v6.1) + `alive_compact_v70` (BlockScan).
- Approximate top-K (heap-1 per thread can't capture multiple high-scorers per K-strip → ref_match dips ~0.996).
- No clear win in bench.

## v7.6

`score_top_l_b512` + `depth_and_bm_global` + `alive_compact_bmg_bs`. Combines v7.5 + bm_global + BlockScan. Regressed (bmg path overhead not amortized).

## v7.7

CUDA Graph wrapper around v7.5 pipeline.
- Captures (score, depth, alive) into single graph; replay = 1 graph launch.
- Lost in bench: `copy_(static_q)` + `copy_(static_th)` + `graph.replay()` overhead exceeds 3-launch naive.
- BUG fixed: must hold refs to ALL workspace tensors (top_scores/top_indices/depth/...) in cache dict — graph captures raw GMEM ptrs; freed tensors yield dangling pointers + corrupt output.

## v7.8

**Force L=256 path** + v6.1 depth + v7.0 BlockScan alive.
- Smaller heap, smaller CUB temp, ~2 µs score speedup.
- Trade: ref_match ~0.99 (vs 0.9995 for L=512 path).
- 8k Llama: 19.8 µs.

## v7.9

L=256 + Idea C bm_global path.
- score (L=256) + `depth_and_bm_global` + `alive_compact_bmg`.
- Slightly slower than v7.8 in bench (bmg pipeline overhead at L=256).

## v7.10

**Single-launch fused pipeline via cooperative groups.**
- Specialised: L=256, BLOCK=256, IPT_L=1, TILE_N=2048, S=4, bf=4.
- Grid (Hq, max(4, N_TILES)). Phases separated by `cg::grid_group::sync()`:
  - Phase 1: `blockIdx.y < 4` → score for `(hq, s)`.
  - Phase 2: `blockIdx.y == 0` → depth + reset live_count.
  - Phase 3: `blockIdx.y < N_TILES` → alive_compact (BlockScan).
- Smem unioned across phases (max of CUB temp / depth scratch / alive scratch).
- Launched via `cudaLaunchCooperativeKernel`.
- 8k Llama: **18.5 µs** (best for small grids).
- Regresses at large N (Qwen 20k: 42 µs) — cooperative wave grows with N_TILES.

## v7.11

v7.10 with **TILE_N=4096, PER_THREAD=16**.
- Cooperative wave: (Hq, max(4, N_TILES)). At N=8461 → N_TILES=3, max_blk=4 (aligned with score grid).
- Loses at small N (8k Llama 20.5 µs vs v7.10 18.5) due to PER_THREAD=16 register pressure.
- Wins at larger N (Qwen 12k: 24.6 µs vs v7.10 28.8) — fewer alive tiles.
- `tile_n` argument toggles 2048 (v7.10) or 4096 (v7.11) at launch time.

## Tuning summary (8k Llama, target ≤16 µs)

| Variant | µs/query | ref_match | Notes |
|---------|----------|-----------|-------|
| v3.4    | 20.9     | 0.9995    | dense alive (legacy best) |
| v6.1    | 22.9     | 0.9995    | compact-list, baseline |
| v6.2    | 22.9     | 0.9995    | G3+G7 |
| v7.0    | 21.2     | 0.9995    | BlockScan compaction |
| v7.1    | 22.5     | 0.9995    | bm_global precompute |
| v7.8    | 19.8     | 0.9915    | L=256 force |
| v7.9    | 20.8     | 0.9915    | L=256 + bm_global |
| v7.10   | **18.5** | 0.9915    | cooperative fused, TILE_N=2048 |
| v7.11   | 20.5     | 0.9915    | cooperative fused, TILE_N=4096 |

Score kernel = ~12.5 µs (dominant) — hard floor without algorithmic restructure
(GQA reuse via SMEM-cached centers, split-K with merge, or warp-per-query top-K).
