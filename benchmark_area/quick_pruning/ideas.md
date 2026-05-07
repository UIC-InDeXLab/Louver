# Ideas for Improving Halfspace Pruning

## Problem Analysis

The speedup ratio is `g/bf + scanned_fraction`, where:
- `g` = gate cost per cluster (in dot-product equivalents)
- `bf` = branching factor (points per cluster)
- `scanned_fraction` = fraction of points that pass the gate

**Constraint**: `g < bf` (gating must be cheaper than brute force on the cluster).

For speedup > 1x, we need `ratio < 1.0`, i.e., `g/bf + scanned < 1`.

### Current state (bf=4, comparison.txt):
- **Best pruning**: kcenter + outlier_aabb → 41% pruned, but ratio=1.21 (g=2.5 too expensive)
- **Best ratio**: pq_span + aabb → ratio=1.13 (scanned=0.76, g=1.5)
- **g=1.0 methods** (ball, span_ball): ~90% scanned → ratio ≈ 1.15

---

## RESULT: Speedup Achieved!

### Winning combination: `fast_balanced_nn + AABB, bf=2`

**27 out of 28 layers achieve asymptotic speedup** on Llama-3.2-3B-Instruct, with mean **1.10x speedup** (ratio ≈ 0.90).

| Layer | Scanned | Pruned | Ratio | Speedup |
|-------|---------|--------|-------|---------|
| Best (L7) | 0.115 | 88.5% | 0.865 | **1.16x** |
| Mean | 0.160 | 84.0% | 0.910 | **1.10x** |
| Worst passing (L27) | 0.236 | 76.4% | 0.986 | **1.01x** |
| Only failure (L1) | 0.293 | 70.7% | 1.043 | 0.96x |

### Why it works

**Two key insights combined**:

1. **bf=2 is the sweet spot**: With AABB (g=1.5), the gate overhead is `g/bf = 0.75`, leaving 25% headroom. With bf=4, the overhead is only slightly less (0.375) but the clusters are larger and harder to prune.

2. **Cluster balance is critical**: The standard kcenter/kmeans produce imbalanced clusters (at bf=2: std=1.9, max size=45, 36% singletons). A few huge clusters dominate the scanned fraction. The balanced NN pairing forces **every cluster to have exactly 2 points**, making all AABBs tight and uniform.

### Why other approaches failed

- **PCA projection box** (ultra-cheap g≈0.1): The per-cluster centroid dot product IS the signal. Avoiding it makes the residual term dominate (~18 in D=128). Even with centering, the residual is too large.
- **Query-adaptive gate**: Query variance is only 7-28% explained by top PCA directions; the residual overwhelms any benefit.  
- **Partial AABB** (g≈1.0): Only marginal improvement over span_ball; the top few dimensions don't capture enough of the AABB tightness.
- **Outlier removal** (g=2.5): Great pruning (41-81%) but the outlier check costs too much. At bf=2 with outlier_aabb: `2.5/2 + 0.17 = 1.42`, far from speedup.

### Clustering quality comparison at bf=2

| Method | Cluster Balance | AABB Scanned | Ratio |
|--------|----------------|-------------|-------|
| kcenter | std=1.9, max=45 | 0.30 | 1.05 |
| balanced_kcenter | std=0, exact bf | 0.30 | 1.05 |
| nn_greedy | std=0, exact bf | 0.20 | **0.95** |
| fast_balanced_nn | std=0, exact bf | 0.20 | **0.95** |

The NN-greedy pairing achieves **tighter AABBs** than balanced k-center even though both are perfectly balanced. This is because NN-greedy pairs the closest points, minimizing within-pair distance and therefore AABB span.

---

## Algorithms

### fast_balanced_nn (bf=2)
1. Compute all pairwise distances: `torch.cdist(keys, keys)` — O(N² D), fast on GPU
2. Iterative mutual-NN matching:
   - Find each point's nearest neighbor (vectorized)
   - Identify mutual nearest neighbors (A's NN is B and B's NN is A)
   - Match all mutual NNs at once
   - Repeat for remaining points (typically 3-5 rounds)
3. Any remaining singletons get paired greedily

Runtime: ~500ms for N=2000, 8 heads. Dominated by cdist.

### AABB gate (g=1.5)
For each cluster with keys lo, hi (per-dimension bounds):
```
UB = Σ_d max(q_d * lo_d, q_d * hi_d)
gate_pass = UB > threshold
```
Per-cluster cost: 3D FLOPs = 1.5 dot-product equivalents.

---

### bf=3 results (nn_greedy + AABB)
17 out of 28 layers achieve speedup at bf=3 (mean 1.05x for passing layers).
bf=2 is more consistent (27/28 layers).

### Summary of ratios across bf values
| bf | Clustering | Enclosing | g | Layers with speedup | Mean ratio |
|----|-----------|-----------|---|--------------------:|----------:|
| 2 | fast_balanced_nn | AABB | 1.5 | **27/28** | **0.91** |
| 3 | nn_greedy | AABB | 1.5 | 17/28 | 0.98 |
| 4 | kcenter | AABB | 1.5 | 0/28 | 1.07 |

---

## New Ideas to Explore

### Empirical observations to inform ideas

**Top-k key diversity** (measured on layer 15, head 0, 600 queries):
- 73% of all keys appear in at least one query's top-20 → top-k is diverse overall
- But 2 keys appear in >50% of all queries' top-20 → a few "always hot" keys
- ~68 keys appear in >5% of queries → a small "warm" set

**Threshold statistics** (top-20 threshold with unit-normalized queries):
- Mean threshold ≈ -0.7 (NEGATIVE — most dot products exceed threshold!)
- Std ≈ 0.7, [25th, 75th] percentile ≈ [-1.3, -0.15]
- Narrow concentration: thresholds are predictable

**Consecutive query overlap**:
- Mean top-20 overlap between consecutive queries: **44%**, up to 85%
- Queries are temporally correlated — consecutive queries share almost half their top-k

**Key norms**: mean=21, std=1.9, range [4, 27]. Relatively concentrated.

---

### A. Exploit threshold structure

The threshold is NOT a random scalar — it's the k-th largest dot product, concentrating in a narrow range.

**A1. Precomputed break-even threshold per cluster.**
For each cluster, compute `t_min_k = min threshold at which cluster passes` (i.e., the AABB upper bound when q maximally aligns with the cluster). If `t_min_k > t_max` (the max realistic threshold), the cluster NEVER passes → mark as "always prune" with zero runtime gate cost. Similarly, if the cluster always passes, mark as "always include." Only gate the borderline clusters.

**A2. Threshold-binned gate lookup.**
Since thresholds concentrate in [-2, 2], discretize into B bins. For each (cluster, bin) pair, precompute whether the cluster passes for a "representative" query at that threshold level. At runtime, look up the bin → O(1) per cluster. Handle the bin-boundary uncertainty with a small correction.

**A3. Threshold tightening via partial computation.**
During search, after evaluating a few clusters, we may discover keys above threshold that would raise the threshold (making remaining clusters easier to prune). Update the threshold progressively as we scan clusters, starting with the most likely-to-pass clusters first (ordered by centroid score). This is like a branch-and-bound: the bound tightens as we go.

---

### B. Exploit cross-query temporal coherence

Consecutive queries share 44% of their top-k keys. This means the pruning decisions are also similar.

**B1. Incremental pruning from previous query.**
After processing query q_t, store which clusters passed. For q_{t+1}, start by assuming the same clusters pass. Only re-evaluate clusters near the decision boundary (where UB_k ≈ threshold). This reduces the number of gate evaluations.

Cost: for clusters far from the boundary, reuse previous decision (g=0). For boundary clusters, evaluate gate (g=1.5). If 70% of clusters are far from boundary: effective g ≈ 0.3 * 1.5 = 0.45.

**B2. Query delta gating.**
Compute Δq = q_{t+1} - q_t. For each cluster, the change in upper bound is ΔUB_k = UB_k(q_{t+1}) - UB_k(q_t). Precompute a bound on |ΔUB_k| using ||Δq|| and cluster properties. If the previous UB was far enough from threshold that |ΔUB| can't change the decision, skip the gate.

**B3. Amortized gate over query windows.**
Process queries in windows of W. For a window, compute the "union gate": a cluster passes if it passes for ANY query in the window. This overapproximates but reduces the total gate evaluations by W×. The cost is that some clusters pass unnecessarily, increasing scanned fraction. If queries in a window are similar, the overapproximation is small.

---

### C. Auxiliary points and materialization

**C1. Hot key extraction.**
Identify the ~2-70 "hot" keys that appear in many queries' top-k. Always include these in the result (zero gate cost). Cluster only the remaining "cold" keys. This reduces N_cold = N - N_hot, making clusters tighter and pruning easier.

For N_hot=68, N=2064: the hot keys cost 68/2064 ≈ 3.3% of brute force. The cold keys (1996) are clustered with bf=2 → K_cold=998 clusters. If cold pruning improves because the hot keys (which forced clusters to be larger) are removed, overall speedup improves.

**C2. Materialize query templates.**
Observe queries over time and extract m "template" query directions via PCA or k-means. For each (cluster, template) pair, precompute whether the cluster passes. At query time:
1. Find the nearest template to q (m dot products, shared)
2. Look up precomputed gate result for that template (O(1) per cluster)
3. For clusters near the boundary, apply a correction using ||q - template||

If m=32 templates cover the query space well (query PCA shows 30% variance in top 8 directions), the corrections are small and most gate decisions are free.

**C3. Sentinel-based coarse filter.**
Place m sentinel points at strategic locations (e.g., cluster centroids of the key distribution, or points along high-variance directions). For each sentinel, precompute which key clusters it "covers." At query time:
1. Compute q·sentinel for each sentinel (m dot products)
2. If a sentinel's score < threshold, all clusters it covers are pruned

This is essentially a two-level hierarchy with m groups at the top level and K clusters at the bottom. The novelty is choosing sentinel positions to maximize coverage.

---

### D. Cheaper gating

**D1. Norm-based pre-filter (g=0).**
For unit query q: q·x ≤ ||x||. So if max_norm_in_cluster < threshold, the cluster is pruned with ZERO gate cost. Key norms range [4, 27] and thresholds range [-2, 2], so max_norm > threshold for all clusters → this doesn't help directly. BUT for non-unit queries: q·x ≤ ||q||·||x||, so if ||q||·max_norm < threshold_absolute... threshold_absolute = threshold * ||q|| ≈ -0.7 * 13.6 ≈ -9.5. Since all norms > 4, ||q||·||x|| > 54 >> -9.5. So norm pre-filter doesn't help here.

Where it COULD help: when threshold is large (rare queries with very high attention concentration). Or in a modified problem where we're looking for the minimum rather than maximum.

**D2. Single-dimension pre-filter.**
For each cluster, identify the dimension d* where the cluster is "most prunable" (largest gap between cluster's AABB bound and typical threshold). Store only lo_d* and hi_d*. Gate: `max(q_d* * lo_d*, q_d* * hi_d*) < threshold_gap`. Cost: 2 multiplies + 1 compare ≈ 2/2D = 1/D dp-equiv ≈ 0.008. Nearly free!

If this single-dim test prunes 30% of clusters, the remaining 70% go through full AABB. Effective g ≈ 0.008 * K + 0.70 * 1.5 = 1.05. With 30% pre-pruned, total scanned drops.

**D3. AABB with fused midpoint.**
For bf=2, keys {a, b}: the AABB gate is `max(q·a, q·b)`. Rewrite as:
```
max(q·a, q·b) = q·m + |q·δ|/2   where m=(a+b)/2, δ=a-b
```
The midpoint dot product `q·m` is already needed (shared with the centroid ball gate). The additional cost is `|q·δ|/2`: one absolute-value dot product = 1 dp. So total g = 2.0 for EXACT max(q·a, q·b).

At bf=2: ratio = 2.0/2 + 0 = 1.0. No speedup from exact computation alone. But if we use Cauchy-Schwarz on the |q·δ| term: |q·δ| ≤ ||δ||, giving the span_ball bound at g=1.0. The AABB is tighter than span_ball because it keeps the directional information.

The gap: 1.0 (span_ball, g=1.0, scanned≈0.70) vs 1.5 (AABB, g=1.5, scanned≈0.20). We need something in between.

**D4. Partial |q·δ| computation.**
Instead of computing |q·δ| exactly (1 dp) or bounding by ||δ|| (free but loose), compute the dot product on only the d largest dimensions of δ and bound the rest:
```
|q·δ| ≤ |Σ_{i∈S} q_i δ_i| + ||q_{-S}||·||δ_{-S}||
```
Cost: d/D dp + ~free. For d=32, D=128: 0.25 dp. Total g = 1.0 (midpoint) + 0.25 = 1.25. If this gets scanned ≈ 0.40: ratio = 1.25/2 + 0.40 = 1.025. Close!

---

### E. Better clustering for larger bf

**E1. AABB-volume-minimizing clustering.**
Standard k-means minimizes L2 distance. For AABB pruning, minimize the sum of per-cluster AABB half-span sums (the AABB "effective radius" ≈ ||half||₁ / √D):
```
minimize Σ_k (Σ_d span_d(cluster_k))
```
This is an L∞-flavor objective. Iterative: assign keys to clusters, update, repeat. At each step, assign each key to the cluster where adding it increases the AABB span the least.

**E2. Hierarchical bf=2 binary tree.**
Build a complete binary tree with bf=2 NN pairing at every level. Bottom: N/2 tight pairs. Next level: pair the pairs (N/4 groups of 4). Etc. Each level gates with AABB (g=1.5). Pruning compounds across levels.

If level i achieves speedup ratio r_i: total ratio ≈ Π r_i. With L=log₂(N) levels and each r_i ≈ 0.90: total ratio ≈ 0.90^11 ��� 0.31 → **3.2x speedup** (theoretical upper bound).

The key question: does the pruning per level degrade as cluster size grows? Probably yes (middle levels have larger AABBs), but the compounding effect may still win.

**E3. Cone + norm decomposition.**
Write each key as x = ||x|| · x̂. Cluster by direction x̂ (cone/spherical clustering). Within each cone, keys point in similar directions. Gate:
```
q·x = ||x|| · (q · x̂) ≤ max_norm * max_cos(q, cone)
```
The cone test costs ~1 dp (angle between q and cone axis). The norm test is a scalar comparison. Total g ≈ 1.0. This separates direction (angular) pruning from magnitude pruning. For keys with similar directions but varying norms, the cone is tight while the ball would be loose.

---

### F. Adaptive / online restructuring

**F1. Query-driven re-pairing.**
After observing W queries, identify which key pairs have "bad" pruning (always pass → wasteful). Re-pair those keys with different partners to improve pruning for the observed query distribution. This is an online version of "optimize clustering for the query distribution."

**F2. Per-head adaptive bf.**
Different attention heads have different pruning characteristics. Some heads have concentrated attention (easy to prune), others are diffuse. Detect this online and use bf=2 for easy heads, bf=4+ for hard heads (or skip pruning entirely for heads where it doesn't help).

**F3. Split-on-fail.**
Start with larger bf (e.g., bf=4). When a cluster consistently passes the gate (fails to prune), split it into two sub-clusters (bf=2 each). Over time, clusters that are hard to prune get refined, while easy-to-prune clusters stay coarse (saving gate cost).

---

### G. Practical engineering

**G1. Faster NN pairing.** For large N, O(N²) cdist is expensive. Approximate NN via random projections, locality-sensitive hashing, or KD-trees could reduce to O(N log N).

**G2. Online update.** When new keys arrive, insert into existing pairs by: (a) joining the nearest existing pair to form a triple then re-splitting, or (b) maintaining a buffer and re-pairing periodically.

**G3. Adaptive bf per level.** Use bf=2 where pruning is effective (middle layers of the model), larger bf where it's not (early/late layers).

---

## H. Measured AABB Gate Cost (Empirical)

### Benchmark setup

Measured wall-clock time of AABB gating vs dot product on GPU, D=128, H=8, various K values. Implementations tested:
- **PyTorch elementwise**: `torch.maximum(q*lo, q*hi).sum(-1)` vs `(q*keys).sum(-1)`
- **Triton kernels**: custom fused AABB kernel vs custom fused dot kernel
- **cuBLAS**: midpoint AABB via 2× `bmm` vs 1× `bmm`

Code in `aabb_kernel_bench/triton_vs_cublas.py`.

### Key finding: g depends on what you compare against

| Comparison family | g (AABB) | Notes |
|---|---|---|
| Triton AABB vs Triton dot | **~1.42** | Fair: same framework, same launch overhead |
| PyTorch elementwise vs elementwise | **~1.6** | Fair: same dispatch path |
| cuBLAS (2× bmm) vs 1× bmm | **~2.0** | cuBLAS GEMM uniquely fast for dot products |
| Best AABB (torch.max) vs Triton dot | ~1.08 | Unfair: different frameworks |

**g ≈ 1.4 is the fair measurement** — Triton-vs-Triton within the same execution model. The cuBLAS bmm baseline (g≈2.0) is misleading because in practice brute-force attention wouldn't use per-cluster `bmm`.

### Why the theoretical FLOP count overestimates g

- `max(a, b)` maps to a single GPU instruction (`fmax`) that co-issues with FMA — it does NOT cost a separate FLOP cycle
- Memory bandwidth dominates at these sizes, not compute
- AABB loads 2D floats (lo + hi) vs dot product's 1D floats, but the compute pipeline hides much of this because the loads interleave with FMA

### Detailed timing (median, microseconds)

| K | cuBLAS bmm | Triton dot | torch manual | Triton AABB | torch.max AABB | mid bmm |
|---|---|---|---|---|---|---|
| 256 | 12.4 | 17.4 | 12.2 | 25.1 | 19.3 | 24.2 |
| 512 | 9.4 | 17.6 | 13.1 | 25.0 | 21.3 | 19.0 |
| 1024 | 9.8 | 18.2 | 13.8 | 25.9 | 22.0 | 19.0 |

### Speedup ratios with measured g=1.42

| bf | scanned=0.15 | scanned=0.20 | scanned=0.30 |
|---|---|---|---|
| **2** | **1.16x** | **1.10x** | ~1.0x (break-even) |
| **3** | **1.61x** | **1.49x** | **1.29x** |
| **4** | **1.98x** | **1.80x** | **1.53x** |

### Implications

1. **bf=2 + AABB still works** (ratio ≈ 0.91 at scanned=0.20), but the margin is thin — only ~25% headroom before break-even.
2. **bf=3 is very attractive** if clustering quality can keep scanned ≤ 0.30. At g=1.42: ratio = 0.47 + 0.30 = 0.77 → **1.30x speedup**.
3. **bf=4+ has massive headroom** (g/bf = 0.35), so even scanned=0.50 gives speedup. The challenge is achieving low scanned fraction with larger clusters.
4. The original g=1.5 estimate was reasonable — slightly optimistic vs Triton (1.42) but in the right ballpark.
