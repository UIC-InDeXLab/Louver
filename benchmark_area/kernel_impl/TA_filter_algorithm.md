# TA-Filter Algorithm

## Inputs

- `q ∈ ℝ^D` — a single query vector (one head).
- `T ∈ ℝ` — a scalar threshold on the full dot product `q·k`.
- A built index over `N` keys, comprising:
  - `S` subspaces with disjoint dimension partitions; subspace `s` covers
    dimensions of width `w_s` (so `Σ_s w_s = D`).
  - In each subspace `s`, the keys are partitioned into `K` clusters
    `C_{s,1}, …, C_{s,K}` of branching factor `bf` (each cluster contains up
    to `bf` key indices).
  - For each cluster `C_{s,c}` we have:
    - a centroid `μ_{s,c} ∈ ℝ^{w_s}`,
    - the list of key indices it owns.

## Output

The set of keys `S_T = { k : q · k ≥ T }`, i.e. all keys whose true full-vector
dot product with `q` reaches or exceeds the threshold `T`.

## Per-subspace centroid score

For each `(s, c)`, define
```
M_{s,c} = q_s · μ_{s,c}
```
where `q_s` is the slice of `q` on the dimensions of subspace `s`.

## Sorted lists

For each subspace `s`, sort the clusters in **descending** order of `M_{s,·}`.
Let `π_s` denote the resulting permutation, so that
```
M_{s, π_s(1)} ≥ M_{s, π_s(2)} ≥ … ≥ M_{s, π_s(K)}.
```
Conceptually, the sorted lists form an `S × K` table whose rows are indexed
by `L = 1, …, K` and whose columns are the `S` subspaces.

## Row-by-row sweep

Maintain a global set `Visited ⊆ {1, …, N}` of keys that have already been
checked, and a working set `Survivors ⊆ {1, …, N}` of keys that have
been confirmed to satisfy `q · k ≥ T`. Both start empty.

For `L = 1, 2, …, K`:

1. **Pop one cluster from each subspace at row `L`.**
   For every subspace `s`, take the cluster `c_s = π_s(L)`. Collect the union
   of their key index lists into a row-`L` candidate set:
   ```
   X_L = ⋃_{s=1}^{S} C_{s, c_s}.
   ```
   Up to `S · bf` candidates per row.

2. **Skip already-visited keys.**
   Remove from `X_L` any keys already in `Visited`. Add the remaining keys
   to `Visited`.

3. **Score the new candidates.**
   For each key `k ∈ X_L \ Visited_prev`, compute the full dot product
   `score(k) = q · k`.

4. **Filter by threshold.**
   For each such `k`, if `score(k) ≥ T`, add `k` to `Survivors`.

5. **Stopping rule.**
   Stop at row `L` if `L` is the first row where the sum of the centroid
   scores is less than `T`:
   ```
   Σ_s M_{s, π_s(L)} < T.
   ```

The sweep also stops naturally if `L` exceeds `K`.

## Result

Return `Survivors` (and, for attention, the softmax over `{ score(k) : k ∈ Survivors }`
weighted by the corresponding values).

## Notes

- Steps 1–4 do per-row work bounded by `S · bf` keys, so each row costs
  `O(S · bf · D)` for the full-vector dot products plus `O(S · bf)` for the
  set membership and threshold tests.
- The `Visited` set ensures every key is scored at most once across the whole
  sweep.
- This centroid-score variant is a heuristic ordering/stopping rule. Unlike an
  upper-bound variant, `M_{s,c}` does not prove that every key in a later
  cluster has score below `T`.
