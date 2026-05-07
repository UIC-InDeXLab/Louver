# New Ideas for Improving Pruning (Round 2)

## Critical Constraint: g < bf Required

**An enclosing with gate cost g ≥ bf can NEVER achieve speedup** since ratio = g/bf + scanned ≥ 1.0 + scanned > 1.0. Always filter out invalid (enclosing, bf) combinations before benchmarking.

### Asymptotic gate costs

At large N (asymptotic regime), gate cost is dominated by memory bandwidth. AABB loads 2 vectors (lo, hi) of size D per cluster, same as 2 dot products → **g=2.0 for AABB**.

| Enclosing | g (asymptotic) | Valid bf | Reason |
|-----------|----------------|----------|--------|
| ball (centroid+radius) | 1.0 | bf≥2 | 1 dot product + scalar add |
| AABB (lo/hi) | 2.0 | **bf≥3** | 2× bandwidth of dot product |
| fp16 AABB | ~1.0 | bf≥2 | half bandwidth → ~1 dp equiv |
| partial_aabb_d32 | ~1.25 | bf≥2 | 32/128 dims × 2 vectors |
| ellipsoid | ~2.5 | bf≥3 | covariance + scaled norm |

**AABB + bf=2 is INVALID** (g=2.0 ≥ bf=2). Must use bf≥3 for AABB or find an enclosing with g<2 for bf=2.

---

## Fresh Benchmark Results (N=2000, Llama-3.2-3B-Instruct)

### Clustering method comparison (bf=2, AABB, layer 15)

| Method | Scanned | Notes |
|--------|---------|-------|
| l1_nn | **0.186** | L1 distance → directly minimizes AABB span |
| weighted_l1_nn | 0.190 | 1/std weighting, no real improvement |
| fast_balanced_nn (L2) | 0.193 | Previous best |
| linf_nn | 0.251 | Worst! L_inf minimizes max span, not sum |

**Insight**: L1 distance is the correct metric for AABB-optimized pairing because the AABB upper bound looseness = Σ_d |q_d| * half_span_d, and half_span = |a_d - b_d|/2 for bf=2 pairs. Minimizing L1 distance = minimizing Σ|a_d - b_d| = minimizing sum of half-spans.

### bf=3 + AABB results (layer 15, g=2.0)

Headroom = 1 - 2.0/3 = 0.333, need scanned < 0.333.

| Clustering | Scanned | Ratio (g=2.0) | Speedup |
|-----------|---------|---------------|---------|
| l1_batch_nn | 0.557 | 1.224 | 0.82x |
| batch_nn (L2) | 0.559 | 1.226 | 0.82x |
| kcenter | 0.576 | 1.243 | 0.80x |

**bf=3 + AABB does NOT achieve speedup.** Scanned is ~0.56, need < 0.333. Large gap (22 percentage points).

### bf=2 + ball results (layer 15, g=1.0)

Headroom = 1 - 1.0/2 = 0.500, need scanned < 0.500.

| Enclosing | g | Scanned | Ratio | Speedup |
|-----------|---|---------|-------|---------|
| ball (centroid) | 1.0 | ~0.75 | 1.25 | 0.80x |
| span_ball | 1.0 | 0.756 | 1.256 | 0.80x |

Ball enclosings prune very poorly in D=128 — scanned ~0.75 (need < 0.50).

### bf=2 + fp16_aabb results (layer 15, g≈1.0)

Headroom = 1 - 1.0/2 = 0.500, need scanned < 0.500.

| Enclosing | g | Scanned | Ratio | Speedup |
|-----------|---|---------|-------|---------|
| fp16_aabb | ~1.0 | 0.188 | **0.688** | **1.45x** |

**fp16_aabb is the winner for bf=2!** Same pruning quality as fp32 AABB (scanned 0.188 vs 0.186) but half the memory bandwidth → g≈1.0.

### All-layer sweep: L1-NN + AABB, bf=2 (with g=2.0)

With g=2.0, bf=2 + fp32 AABB is invalid (ratio always > 1.0). But **fp16_aabb with g≈1.0**:

| Layer | Scanned | Ratio (g=1.0, fp16) | Speedup |
|-------|---------|---------------------|---------|
| Best (L0) | 0.094 | 0.594 | **1.68x** |
| L7 | 0.172 | 0.672 | **1.49x** |
| L11 | 0.135 | 0.635 | **1.57x** |
| L14 | 0.189 | 0.689 | **1.45x** |
| Mean (easy layers) | 0.202 | 0.702 | **1.42x** |
| L27 | 0.292 | 0.792 | **1.26x** |
| L2 | 0.305 | 0.805 | **1.24x** |
| L1 (worst) | 0.337 | 0.837 | **1.19x** |

**With fp16_aabb (g≈1.0), ALL 28 layers achieve speedup at bf=2.** Mean ≈ 1.40x.

---

## What Didn't Work (Newly Tested)

### Hierarchical bf=2 binary tree
Built a complete binary tree with NN pairing at each level. Result: **upper levels can't prune** (pass_frac=1.0 from level 3 upward). Only the bottom level (bf=2 pairs) does any pruning, making the tree worse than flat due to gate cost overhead.

- Layer 15: Tree ratio=1.51 vs Flat ratio=0.69
- The AABB of 4+ keys is too loose to prune in D=128

### Threshold-aware static pruning
Precompute max possible UB per cluster: max_UB = Σ_d max(|lo_d|, |hi_d|) ≈ 216. Since thresholds are in [-3, 3], every cluster can pass for some query → **zero clusters can be statically pruned**.

### Hot key removal
Removing 3-10% highest-norm keys barely changes scanned fraction (<1% improvement). Hot keys are already scanned, and their removal doesn't make remaining clusters tighter enough.

### L_inf NN pairing
L_inf minimizes max per-dimension span, but AABB tightness depends on the SUM of |q_d * half_d|. L_inf produces worse AABBs than L1 or L2.

### Weighted L1 pairing
Weighting dimensions by 1/std doesn't help because the optimal weights depend on query direction (unknown at build time).

### bf=2 + fp32 AABB
AABB has g=2.0 asymptotically. Since g ≥ bf=2, this combination **can never achieve speedup**. Must use fp16_aabb (g≈1.0) or ball (g=1.0) for bf=2.

### bf=3 + AABB
Even with the best clustering (l1_batch_nn), scanned=0.557 is far above the 0.333 threshold needed for speedup with g=2.0.

---

## Why Hard Layers Are Hard

**Layers 1, 2, 27** have very negative top-20 thresholds (mean ≈ -2.4 vs -0.3 for easy layers). This means attention is diffuse: many keys are relevant, making pruning fundamentally harder.

| | Easy layers (11, 14) | Hard layers (1, 2) |
|---|---|---|
| Threshold mean | -0.24, -0.33 | -2.42, -2.38 |
| L2-NN dist mean | 9.2, 10.5 | 10.4, 12.9 |
| PCA top-1 var | 8-10% | 3-5% |
| Scanned (bf=2) | 0.13, 0.19 | 0.34, 0.30 |

Even hard layers achieve speedup with fp16_aabb (g≈1.0): worst case ratio = 0.50 + 0.337 = 0.837 → 1.19x.

---

## Remaining Ideas to Try

### A. Verify fp16_aabb g≈1.0 asymptotically
fp16 loads half the bytes of fp32 → same bandwidth as one dot product → g≈1.0. Need to confirm this holds at large N in a real kernel. The slight precision loss (we add ε=1e-3 to bounds) has negligible effect on scanned fraction.

### B. Make bf=3 work with tighter enclosings
bf=3 + AABB fails because scanned=0.56 >> 0.33. But bf=3 + fp16_aabb (g≈1.0) needs scanned < 0.667 → **bf=3 + fp16_aabb works** (0.557 < 0.667, ratio=0.890, speedup=1.12x). However, bf=2 + fp16_aabb is still better (ratio≈0.69 vs 0.89).

### C. Reduce scanned fraction further
Current best: l1_nn + fp16_aabb + bf=2 → scanned ≈ 0.186.
- Better clustering that accounts for query distribution
- Tighter AABB bounds (currently conservative with ε=1e-3)
- Signed AABB: separate lo/hi for positive and negative query dimensions

### D. Larger N improves pruning
With more keys, the key space becomes denser, NN pairs get closer, and AABBs get tighter. N=5000 should show even better scanned fractions.

### E. Query-aware re-clustering (online)
After observing W queries, identify which pairs have bad pruning. Re-pair those keys to optimize for the observed query distribution. Requires online clustering, but could significantly improve hard layers.
