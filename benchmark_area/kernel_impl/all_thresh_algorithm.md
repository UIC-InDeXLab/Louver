## Task
The task is to simulate the attention score calculation with my new index defined below. 

### Prefilling
During the prefilling, there are a set of keys created. Right after the prefilling, we are going to build an index on those keys. 

### Decoding
Then, during the decoding, when a new query arrives, we need to look for the keys (top-k ones) and run an index search to extract the important ones. During the decoding, also new keys are generated. They are kept in a `key_buffer`. After `updating_interval` of decoding steps, we update the index with the buffer and empty the buffer. 
As a result, for each query, the calculation contains two parts:
    1- search the index and return the dot products of query to top-k keys inside index
    2- also calculate the dot product of query to `key_buffer`

These two steps overall, show us the `search speed` of our method.

### Baseline
The baseline is to calculate the dot product of query to all the keys. This will be the search speed of baseline.

## Index Implementation
We need to have a clean `Index` class that implements the `subspace_kcenter` + `ball_centroids`. The simple implementation of this method of indexing can be seen in `benchmarking/quick_pruning/comparison_subspace_kcenter.py`. See that implementation to get hints, but here I want a more optimized and cleaner implementation of this index.

This index class should support three main functionalities:

### build()
Given a set of initial key vectors, build the index. This build contains two steps: first, divide the D-dimensional key vectors into `n_subspaces` *contiguous* subspaces. Second, within each subspace independently create an index by running `kcenter` clustering and `ball_centroid` enclosing.

Within each subspace, I call the base layer (that contains all the key points) as `children`. And I call the layer on top of it with `n/bf` centroids as the `parent` layer.

#### Optimization
I need a very fast implementation of `build()`. It is going to be on GPU. So, you need to write write kernels to have to get a fast building for our index. Write your kernels in a dir called `kernel_impl/kernels/**`. Version them, for example, the first kernel is `kernel_impl/kernels/build_v1.0` and etc.

I will use this versioning to compare different variations of kernel implementation in future and compare them. Also add a new simple benchmarking experiment, to compare different kernel implementations with other simple `torch` vectorized implementations. This simple benchmark should simply find all `build_vx` kernels, benchmark them, and compare with torch versions. Finally, in the console output, it should print a simple timing comparison. The goal of this is to simply track and optimize the kernels in future.

### search()
Given a pair of query vector and threshold, you need to search in the index. The input is a query vector similar to attention (the same shape). The threshold is a vector of size `n_subspaces`. In other words, for each subspace you are given a new threshold. The `search()` should search on the indexes within each subspace efficiently and return those keys with an AND filtering logic: "those keys that pass the thresholds in all the subspaces."

#### Optimization
The `search()` is the *number one priority* for being fast and optimized. Currently, in `benchmarking/quick_pruning/comparison_subspace_kcenter.py` I see an asymptotic speedup. Meaning that the pruning is good enough to get speedup vs. the brute-force baseline.

Similar to build, for `search()` you need to have kernel implementations that are fast. The implementations should be in `kernel_impl/kernels/search_vx.y`. In addition, we need a similar simple benchmarking script, with auto detection of kernels, to benchmark only search kernels in isolation. This benchmark should use real key/queries because the search speedup (and pruning power) depends on input distribution.

### Update
There are two methods to update the index and I want to have both of them implemented and try them, to see which one works best. So, you need to give me option to choose between these two:
1. `full`: Full update. After `updating_interval` steps of decoding, rebuild the whole index on all the keys.
2. `inc`: Incremental update. After `updating_interval` steps of decoding, build a new index `I_2`. Then, append the children of `I_2` to the children of existing index and append the parents of `I_2` to the parents of existing index.

#### Optimization
For update I also need to be fast because it is part of the `inference` process. However, it is amortized on `updating_interval` decoding steps. As a result, similar to build and search, implement fast fused kernels in `kernel_impl/kernels/update_vx.y` and a simple benchmarking script to only compare updating kernels.


## Benchmarking
I need a script with input args to run an end-to-end benchmark (alongside micro-benchmarks for optimizations). Call this end-to-end as `bench.py`. This benchmark should use real query/keys captured from an LLM model. See and use `benchmarking/quick_pruning/pruning_bench_utils.py`. I may also load existing captured `.pt` files.

The benchmarking works as follows: it simulates the inference (decoding) of an LLM, where new queries arrive at each step of decoding and new keys are generated. The simulation happens on the captured query/keys.

### Thresholds
To find out the thresholds, we need a simple simulation: First, given the query and all the keys that we have so far at step `i` of decoding, find out the top-k dot product (high scored) keys for this query (each head independently). For the top-k set `T`, find the smallest values of dot product in each subspace: `t_1`, `t_2`, ..., `t_{n_subspaces}`. I mean, find out which point in `T` has smallest dot and use that as threshold. Finally pass these `n_subpsaces` thresholds to the search function. 

Remember, *exclude* this step from your timing reports. It is not part of search. This threshold finding is just a simulation. 

### Reports
I need to have an output `.csv` file with these reports (stored in `kernel_impl/reports`): 
1. search time per token position (our method and baseline)
2. memory usage per token position
3. update time per token position (0 if no update >0 if it was `update_interval` position)
4. amortized search time: (search time + update time / `update_interval`)

#### Incremental reporting
I need to get the report plots (on top of .csv files) incrementally. After a couple of steps, incrementally update the csv file. Have some *separate* simple plotting scripts in `kernel_impl/plots` to plot the csv file and output it in `kernel_impl/reports`.

The plots should show: step `i` of decoding in the x-axis and amortized time in the y-axis. For now, this is the only plot I need.

## Final notes
- All your implementations should go to `kernel_impl`
- In this project directory, you may find some similar **old** implementations. **Ignore them** and only focus on those parts I explicitly referenced in this document.
- At some point, for some of functions above, you may not implement kernels, if torch vectorized is fast enough.
- Use `~/venv/bin/python` if necessary.
- Keep benchmarking scripts simple.
- If you have questions and there are unclear parts or there are important design/implementation choices, ask me to choose and clarify.
- For now, everything thing is on CUDA (GPU)