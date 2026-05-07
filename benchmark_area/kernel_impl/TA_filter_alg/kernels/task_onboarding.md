## Algorithm Steps
These are the main TA algorithm steps:

### Index Build
Currently implemented in `benchmark_area/kernel_impl/TA_filter_alg/kernels/v13/TA_build_v_13_0.py`.

- The index is built on the key vectors, also called children. 
- The space is divided into `S=4` contiguous subspaces. 
- For each subspace a clustering algorithm is used to find `n/bf` centroids (parents) and each parent is assigned exactly `bf=4` children. In other words, each cluster has 4 points.

*The index contains following:*
- Mapping of key->parent per subspace.
- Mapping of parent->key per subspace.

### Fixed params
- `bf=4`
- `S=4`

More details of the algorithm is in `benchmark_area/kernel_impl/TA_filter_algorithm.md`.

### Inference time
These are major algorithm steps, given query `q` and threshold `T`:

1. [Scoring] Compute all parents scores
2. [ParentFiltering] Find candidate parents `P*`: consider a logical table with `n/bf` rows and `S=4` columns, one column per subspace. Sort parents of each subspace based on their score in descending order. Find the first row `L*` that sum of scores at that row is less than `T`. (Like Fagin and threshold algorithm for top-k retrieval).
3. [Mapping] Find candidate children `C*`: children of candidate parents.
4. [SparseAttention] Calculate sparse attention only on the key set `C` and their corresponding values.

## Kernels
- Step 4 is implemented in a different kernel inside `benchmark_area/kernel_impl/TA_filter_alg/kernels/sparse_attn/sdpa_cuda_sparse_v1_6` and `v1_20`. Called **sparse attention** kernels

- Steps 1 and 2 should be in a separate kernel: **filtering kernels**. They should be implemented in `kernels/filtering/` directory.

- Kernels are benchmarked with simple scripts in `kernel_impl/TA_filter_alg/kernel_bench/*`. For benchmarking we use capture files in `benchmark_area/quick_pruning/capture*.pt`. We also use `~/venv/bin/python`.