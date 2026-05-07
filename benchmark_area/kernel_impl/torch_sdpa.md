# Torch SDPA fp16 (Flash-Decoding) — Deep Dive

Reference point: the `sdpa fp16` row in `benchmark_area/kernel_impl/kernels/kernel_bench/bench_attention.py`. Goal: document every implementation idea in the SDPA fp16 path so we can pull the useful ones into our own fused anchor-block attention kernels.

Local environment verified: `torch 2.9.0+cu128`, CUDA 12.8. All SDPA backend flags (`flash`, `mem_efficient`, `math`, `cudnn`) return `True` from Python, but at runtime `mem_efficient` and `cudnn` are disabled in this build — so for our bench shape, SDPA only has `flash` or `math` to pick from.

---

## 1. What actually runs for our bench shape

Bench call (after reshape):
```python
q4 = qn.view(1, H_q, 1, D)          # B=1, H_q=24, M=1, D=128, fp16
k4 = keys_full_f16.view(1, H_kv, N, D)   # H_kv=8, N≈1000
v4 = values_full_f16.view(1, H_kv, N, D_v)
F.scaled_dot_product_attention(q4, k4, v4, is_causal=False, scale=scale, enable_gqa=True)
```

Profiler (CUDA, 5 calls) shows exactly two CUDA kernels per SDPA call:

| Kernel | Per-call time | Role |
| --- | --- | --- |
| `pytorch_flash::flash_fwd_splitkv_kernel<...>` | ~5.72 µs | fused Q·Kᵀ, online softmax, P·V, per-split partial O + LSE |
| `pytorch_flash::flash_fwd_splitkv_combine_kernel<...>` | ~3.32 µs | reduction across splits using LSE weights |

So "sdpa fp16" in our bench = **Flash-Decoding** (the split-KV flavor of FlashAttention-2), vendored into PyTorch under `aten/src/ATen/native/transformers/cuda/flash_attn/`. Comparison:

| Backend forced via `sdpa_kernel()` | Per-call ms |
| --- | --- |
| FLASH (splitkv) | 0.022 |
| MATH (dense-equivalent, 3 launches) | 0.095 |
| default dispatch | 0.011 |

The `default` path is ~2× faster than forcing `FLASH_ATTENTION` explicitly because the context-manager enter/exit adds a measurable Python + C++ overhead that the default fast path sidesteps.

Dispatcher constraint that matters for GQA: `enable_gqa=True` is only honored by the FLASH backend (and MATH). cuDNN/mem_efficient will fall over. For GQA to hit FLASH, H_kv must divide H_q (24/8 = 3 ✓). If H_q were 28 (not divisible by 8), flash would also fall back to MATH — this is easy to hit by accident and makes sdpa look slow for the wrong reason.

---

## 2. Split-KV kernel (`flash_fwd_splitkv_kernel`) — the fused body

### 2.1 Grid layout

```cpp
dim3 grid(num_m_block,
          params.num_splits > 1 ? params.num_splits : params.b,
          params.num_splits > 1 ? params.b * params.h : params.h);
```

For our bench: `num_m_block = ceil(M/kBlockM) = 1`, so `grid = (1, num_splits, B*H_q) = (1, S, 24)`. Each threadblock owns one output row-block for one (batch, head) for one K-range split.

### 2.2 Split range assignment

Every threadblock gets a contiguous stripe of K/V blocks:

```cpp
const int n_blocks_per_split = (ceil_div(seqlen_k, kBlockN) + num_n_splits - 1) / num_n_splits;
const int n_block_min = n_split_idx * n_blocks_per_split;
const int n_block_max = min(ceil_div(seqlen_k, kBlockN), (n_split_idx+1) * n_blocks_per_split);
```

Ranges are pure integer arithmetic — no atomic queues, no dynamic scheduling. Every block knows its K range up front.

### 2.3 `num_splits_heuristic` — fill the GPU on small grids

This is the key trick for M=1 decoding. Without splitting, grid would be `(1, B, H) = (1, 1, 24)` → only 24 SMs busy on a 108-SM A100 / 132-SM H100.

```cpp
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs,
                                int num_n_blocks, int max_splits) {
    if (batch_nheads_mblocks >= 0.8f * num_SMs) return 1;     // already full
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    auto is_split_eligible = [&](int s) {
        return s == 1 || ceildiv(num_n_blocks, s) != ceildiv(num_n_blocks, s-1);
    };
    for (int s = 1; s <= max_splits; ++s) {
        if (!is_split_eligible(s)) { efficiency.push_back(0.f); continue; }
        float n_waves = float(batch_nheads_mblocks * s) / num_SMs;
        float eff = n_waves / ceil(n_waves);     // how close to a full wave
        max_efficiency = max(max_efficiency, eff);
        efficiency.push_back(eff);
    }
    for (int s = 1; s <= max_splits; ++s) {
        if (!is_split_eligible(s)) continue;
        if (efficiency[s-1] >= 0.85f * max_efficiency) return s;   // smallest good one
    }
    return 1;
}
```

Ideas worth stealing:
- **Wave efficiency metric** `n_waves / ceil(n_waves)` = fraction of SMs utilized in the final wave. Trivially portable to any kernel whose grid is too small to fill the GPU.
- **"≥ 85% of best" → pick the smallest such split.** Avoids paying reduction cost unless it actually buys you occupancy. Biases toward fewer splits when ties exist.
- **`is_split_eligible` filter.** Only considers splits where `ceil(N/s)` actually changes — skips redundant candidates that produce the same block count as a smaller split.

Force-enable path: `mha_fwd_kvcache` sets `force_split_kernel=true` when paged KV / cache_batch_idx is used, so the splitkv dispatch is also used for cache-backed decoding regardless of occupancy.

### 2.4 Shared memory layout (swizzled)

```cpp
Tensor sQ = make_tensor(make_smem_ptr(smem_),               SmemLayoutQ{});
Tensor sK = make_tensor(sQ.data() + size(sQ),               SmemLayoutKV{});
Tensor sV = make_tensor(sK.data() + size(sK),               SmemLayoutKV{});
Tensor sVt       = make_tensor(sV.data(),    SmemLayoutVtransposed{});
Tensor sVtNoSwz  = make_tensor(sV.data().get(), SmemLayoutVtransposedNoSwizzle{});
```

Two separate "views" over V in SMEM: the swizzled view for the GEMM, and a non-swizzled view for the `ldmatrix` during P@V (when the access pattern would fight the swizzle). `SmemLayout*` come from CUTLASS CuTe — they encode a 3D swizzle `(Bits, Base, Shift)` that rotates column indices so that each 16-byte chunk of a 128-byte MMA tile lands on a distinct bank. Net: zero bank conflicts on `ldmatrix.x4` for Q, K, and V, both forward and transposed.

### 2.5 Inner loop per K/V block

```cpp
for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);
    FLASH_NAMESPACE::cp_async_wait<0>();                 // finish prior K load
    __syncthreads();
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV,
        binfo.actual_seqlen_k - n_block * kBlockN);      // async issue next V load
    cute::cp_async_fence();
    FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, // Q @ K^T via Tensor Cores
                          tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                          smem_thr_copy_Q, smem_thr_copy_K);
    mask.apply_mask<Is_causal, Is_even_MN>(acc_s, ...);
    softmax.softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/...>(
        acc_s, acc_o, params.scale_softmax_log2);
    // P @ V into acc_o (another GEMM, reading from sVt / sVtNoSwizzle)
}
```

Key pipelining points:
- `cp.async` (vectorized 16-byte `cp.async.cg`) issued for the *next* K/V block before the current MMA runs. Memory latency is hidden behind math.
- `cp_async_wait<0>` + `__syncthreads()` placed exactly once per loop iteration — the minimum fence pattern.
- `acc_s` (S matrix) and `acc_o` (O accumulator) live in **registers only**. Never touch SMEM, never touch HBM. For kBlockM=64, kBlockN=64, D=128 at fp32 accum: `acc_s = 64*64 = 4 KB` and `acc_o = 64*128 = 8 KB` in fp32, distributed across the warp.
- `Is_first=true` specialization on the first iteration: skips the rescale of `acc_o` (which is zero) and just writes the first max/sum. Saves one `exp2f` per element on step 0.
- `Clear_OOB_MN=true` keeps out-of-bounds elements zeroed instead of requiring a runtime mask, so the inner GEMM has a tight static shape.

### 2.6 Online softmax (`softmax_rescale_o`)

```cpp
reduce_max</*zero_init=*/false>(scores, row_max);            // warp shfl (__shfl_xor_sync)
float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
row_sum(mi) *= scores_scale;
for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
    acc_o_rowcol(mi, ni) *= scores_scale;                    // rescale running O
reduce_sum</*zero_init=*/false>(scores, row_sum);            // l_i update
```

Tricks:
- **`exp2f` not `expf`.** Single hardware instruction (`MUFU.EX2`) on Ampere+. The softmax scale is pre-multiplied by `log2(e)` on the host (`params.scale_softmax_log2 = softmax_scale * M_LOG2E`) so `exp2(x * scale_log2) == exp(x * scale)` exactly. Saves ~15 cycles per element vs `__expf`.
- **Fused rescale into both `row_sum` and `acc_o`** in one pass — `scores_scale` is computed once per row and applied to all of `acc_o` and the running `l_i`.
- **Warp-level reduction only.** Row-max / row-sum are reduced across lanes of a warp via `__shfl_xor_sync` — no SMEM traffic, no `__syncthreads()`. Works because each row of S is assigned to one warp (or a fixed subset of threads) in the MMA layout.
- **`Check_inf` templated on whether the row can go fully `-inf`** (causal / local / ragged cases). When `false`, a branch is removed from the hot path.
- **LSE correctness preserved through tiling.** Final row LSE = `row_max + log(row_sum)`. Everything downstream (combine kernel) uses this.

### 2.7 Epilogue — write partial O + LSE

```cpp
if (get<1>(taccOcO_row(0)) == 0) {
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccOcO_row(mi));
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
            gLSEaccum(row) = lse(mi);
        }
    }
}
// acc_o flushed to gOaccum[split, b, h, m, d]
```

- Only lane 0 of each row writes LSE — same MMA-row-layout trick as the softmax reduction.
- `acc_o` goes to `oaccum[num_splits, B, H, M, D]` in fp32 (no down-cast yet). Down-cast happens in the combine kernel, so split boundaries don't accumulate fp16 rounding error.
- Per-split overhead is exactly `num_splits * B * H * M * (D + 1)` extra global-memory words. For our bench (S≈5, B=1, H=24, M=1, D=128): ~15 KB — trivial next to the K/V reads.

---

## 3. Combine kernel (`flash_fwd_splitkv_combine_kernel`)

```cpp
constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 :
                              (Kernel_traits::kHeadDim % 64  == 0 ? 8 : 16);
dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
```

Grid is *tiny* (here: `ceil(1*24*1 / 4) = 6` blocks) and the kernel is mostly memory-bound on LSE/oaccum reads.

Body:

```cpp
// 1) Load LSE for all splits of this row block
ElementAccum lse = (row < params.num_splits && col < lse_size - bidx*kBlockM)
                 ? gLSEaccum(row, col) : -INFINITY;

// 2) Max across splits (warp allreduce)
ElementAccum lse_max = lse_accum(0);
#pragma unroll
for (int l = 1; l < kNLsePerThread; ++l) lse_max = max(lse_max, lse_accum(l));
lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, MaxOp<float>{});

// 3) Denominator (sum of exp(lse_i - lse_max))
float lse_sum = expf(lse_accum(0) - lse_max);
#pragma unroll
for (int l = 1; l < kNLsePerThread; ++l) lse_sum += expf(lse_accum(l) - lse_max);
// ... allreduce lse_sum, then store normalized per-split weights to sLSE

// 4) Weighted sum of partial outputs
ElementAccum lse_scale = sLSE[split][row];
#pragma unroll
for (int k = 0; k < size<2>(tOrOaccum); ++k)
#pragma unroll
for (int i = 0; i < size<0>(tOrOaccum); ++i)
    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);

// 5) Final write (down-cast to fp16 here)
auto o_ptr = reinterpret_cast<Element *>(params.o_ptr)
           + batch_idx * params.o_batch_stride
           + head_idx  * params.o_head_stride
           + row       * params.o_row_stride;
```

Ideas worth stealing:
- **Same LSE merge rule used across blocks is reused across splits.** No new math — splits are just "macro-blocks" that happened to run concurrently. One mental model for both the intra-kernel reduce and the cross-kernel reduce.
- **Per-split weights computed in SMEM, shared across all `d` lanes.** `sLSE[split][row]` is read once by every thread that owns a `d` slice of that row.
- **Down-cast from fp32 accum to fp16 happens *after* combining**, not before. Protects against cancellation.
- **kBlockM tuned to head dim.** For D=128 → kBlockM=4 (so each block handles 4 rows × 128 dims = fits neatly in the warp). For D=64 → 8 rows. Keeps threadblock size independent of D.

---

## 4. GQA — how enable_gqa keeps K/V at H_kv

FlashAttention's splitkv grid third axis is `B*H_q` (here 24). But K/V in HBM stay at H_kv=8. The mapping from threadblock → (q_head, kv_head) is computed inside the kernel:

```
h_q  = blockIdx.z % H_q
h_kv = h_q / (H_q / H_kv)   // groups = H_q / H_kv
```

Every Q head in a group loads the *same* K/V tile from HBM. Because the loads go through SMEM, only the first block in the group that touches a given K/V stripe causes the actual HBM read — subsequent groupmates refill the same stripe but the L2 cache serves it. Net effect: K/V HBM bandwidth ≈ `H_kv * N * D * 2` bytes, not `H_q * N * D * 2`.

This is exactly why `enable_gqa=True` → FLASH is faster than `repeat_interleave(K) → cuDNN` even though cuDNN is normally faster on Hopper: the expand materializes 3× K/V in HBM.

---

## 5. Why this crushes the dense fp16 baseline

Dense baseline (`torch.einsum + softmax + einsum`) for our shape costs ~95 µs; Flash-Decoding costs ~9 µs. Four compounding reasons:

1. **Kernel launch count: 3 → 2.** And both Flash kernels are tiny.
2. **S/P never materialized in HBM.** Dense writes `S ∈ R^{H_q × N}` (fp16 ≈ 48 KB for our shape) and reads it back for softmax, then writes `P` and reads it back for P@V. Flash keeps S and P in registers.
3. **K and V each read once, not twice.** Dense reads K in the first einsum and V in the second — two separate streaming passes over the KV cache. Flash does both GEMMs against the SMEM-resident tile.
4. **SM saturation via split-KV.** Without splits, `batch*heads = 24` active blocks on a 108-SM A100 = 22% utilization. `num_splits_heuristic` picks ~5 splits → ~120 blocks → full occupancy with one small tail. This is the single biggest win at M=1.

Memory traffic per call (rough):
```
Dense fp16:  2 * (H_kv * N * D * 2)  +  2 * (H_q * N * 2)       // K/V twice + S/P round-trip
Flash-fp16:  1 * (H_kv * N * D * 2)  +  O(num_splits * H_q * (D+1) * 4)
```
For N=1000, H_kv=8, D=128: dense ≈ 2 MB, Flash ≈ 1 MB + 30 KB. Roughly **2× less HBM traffic**, and we're already bandwidth-bound at M=1 (compute is ~20 GFLOPs, memory is the gate).

---

## 6. Ideas to port into our anchor-block attention kernels

Ranked by expected impact for our workload (pruning/search + fused softmax + @V, M=1 decoding):

1. **Two-kernel split-KV pattern + `num_splits_heuristic`.** Our grid today is `(1, H_q)` or similar — if `H_q * mblocks < 0.8 * SMs`, we leave compute on the table. Drop-in: add a split dim, write `oaccum[split, h, d]` and `lseaccum[split, h]`, then a combine kernel with weighted LSE merge. Use the exact wave-efficiency heuristic. Expected: largest win on smaller models / higher occupancy GPUs.
2. **`exp2f` + pre-multiplied `scale_softmax_log2`.** Strict replacement for `__expf` inside softmax. One-line change, ~10–15 cycle save per call per softmax element.
3. **fp32 `acc_s` / `acc_o` in registers, fp16 everywhere else.** No SMEM for S/P. Only fp16 in HBM. Already close to what we do, but worth auditing our kernels for any accidental SMEM store of the score matrix.
4. **CuTe-style swizzled SMEM layouts for K and V.** Eliminates `ldmatrix` bank conflicts for the MMA. Larger lift (needs CuTe or manual swizzle), but the swizzle is formulaic — `(3,3,3)` for 128B tiles at fp16.
5. **`cp.async` double-buffering of the next K/V block while current MMA runs.** Biggest perf win if we're currently loading K/V with synchronous loads. Use `cp.async.cg` (128B vectorized) + `cp_async_fence` + `cp_async_wait<0>` pattern. Pair with a 2-buffer SMEM layout so load-block-j+1 doesn't clobber block-j.
6. **Is_first specialization.** Template on first-iteration flag to skip `acc_o` rescale. Tiny, free.
7. **Warp-level `__shfl_xor_sync` reductions for row_max / row_sum.** Assuming one-warp-per-row MMA layout, no SMEM reduction needed.
8. **LSE-based cross-block merge (generalize beyond our subspace pruning).** If we ever split anchor processing across threadblocks (e.g. large BF), the same LSE merge rule from the combine kernel works for us — emit partial `(o, lse)` per block and combine.
9. **Avoid H_q-expansion for GQA.** If we ever repeat_interleave K/V for convenience, stop. Keep K/V at H_kv and map `h_q → h_kv` inside the kernel; rely on L2 + SMEM reuse across groupmates.
10. **`Clear_OOB_MN=true`-style static shape + cooperative clear.** Instead of runtime masks inside the GEMM, zero-fill out-of-bounds SMEM entries once on load. Lets the inner GEMM run on a constant shape.

### Hazards to avoid (things Flash handles and we'd have to match)

- Splitkv + non-contiguous KV (paged / cache_batch_idx) needs extra per-block gather logic. Flash only force-enables splitkv for kvcache when the caller opts in.
- `enable_gqa` compatibility: if our kernel's grid or masking logic assumes H_q==H_kv, split-KV will double-book K loads. Index map must be explicit.
- LSE must stay fp32 in the intermediate buffer. fp16 LSE silently collapses for large context lengths (overflow in `exp(lse)` before normalization).
- Combine kernel block size should track head dim (`kBlockM = 4 for D%128==0, 8 for D%64==0, else 16`) or we waste threads.

---

## 7. Quick decision matrix for which tricks to try first

| Trick | Implementation cost | Expected speedup on our M=1 decode | Notes |
| --- | --- | --- | --- |
| `exp2f` + `scale_softmax_log2` | hours | 2–5% | one-liner, no risk |
| `cp.async` next-block prefetch | 1–2 days | 10–30% | needs 2-buffer SMEM |
| Split-KV + combine kernel | 2–4 days | 20–50% at low occupancy | biggest structural win |
| Swizzled SMEM (CuTe layouts) | 2–3 days | 5–15% | eliminates bank conflicts |
| Is_first + branch templating | 0.5 day | 1–3% | free once layout is right |
| GQA-native K/V sharing | already have it | — | sanity-check our current code |

Starting point I'd recommend: **(1) `exp2f` swap for a free win, (2) split-KV + combine kernel for the big structural win, (3) cp.async double-buffer once split-KV is in.** Swizzling is last because CuTe integration is the heaviest lift and the payoff is smaller than the occupancy win.
