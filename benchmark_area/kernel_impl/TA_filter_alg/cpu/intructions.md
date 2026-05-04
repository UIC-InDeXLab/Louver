I want to write the CPU pipeline of our `TA_filter` algorithm in this directory. The `CUDA` version is in kernels/ directory. Like CUDA version, for CPU, I want fast `.cpp` kernels to achieve significant speedup vs. baselines (on CPU).

First, I need `sparse_attn` for CPU. like in `kernels/sparse_attn`. Bring one baseline of full attention, and either use existing or implement our own sparse attention to be extremely faster than full attention in `0.2` fraction. The attention should support buffer like the CUDA version. Add a simple benchmark script for that in `cpu/benchmarking`.

Second, I need filtering kernels to be very fast. Like filtering kernels for CUDA. But in CPU, the bottlenecks are different so the implementation should be specific fast on CPU. Also simple benchmark for filtering kernels. We also need a build kernel for building the index, like `TA_build.py` which was on CUDA.

Third, I need updating kernels. The process is similar to CUDA ones. The update is incremental on buffer of size 256.

Finally, I want to combine these to end-to-end implementation, and have `cpu_index.py` and `cpu_bench.py` to pack all these into a single decoding attention. Report the times like in `bench.py` for CUDA and I want to see speedup (significant) vs. the dense and sdpa baselines.

All implementations should go to `TA_filter_alg/cpu` directory. Use this venv: `~/venv/bin`.

### References
- Task onboarding: `benchmark_area/kernel_impl/kernels/task_onboarding.md`
- Algorithm: `benchmark_area/kernel_impl/TA_filter_algorithm.md`
- CUDA kernel implementations: `benchmark_area/kernel_impl/kernels/{filtering,sparse_attn,update}`
- CUDA end-to-end: `TA_filter_alg/{index.py,bench.py}`