# %%
import sys

sys.path.append("../")
sys.path.append("../../")

from thresholding.helpers import ObserveAttentionHelper
from hira.indexer import CUDAIndexer
from hira.searcher import CUDASearcher

import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import random
from tqdm import tqdm
import pandas as pd
import triton.testing as tt

# %% [markdown]
# ### Benchmarking

# %%
# config
config = {
    "branching_factor": [4, 8, 16],  # , 32, 64],  # , 128],,
    "update_every": [2**9],
    "num_levels": [CUDAIndexer.DEPTH.TWO_LEVELS, CUDAIndexer.DEPTH.THREE_LEVELS],
    "output_csv": "cuda_results_values.csv",
    "methods": {"brute_force", "search"},
}

# %%
# observer.snapshot()


# %%
def repeat_kv(x, n_rep):
    # x: [batch, n_kv_heads, seq_len, head_dim]
    b, n_kv, s, d = x.shape

    x = x[:, :, None, :, :]  # [b, n_kv, 1, s, d]
    x = x.expand(b, n_kv, n_rep, s, d)  # [b, n_kv, n_rep, s, d]
    return x.reshape(b, n_kv * n_rep, s, d)  # [b, n_kv*n_rep, s, d]


def run_func(func, warmups=10, runs=50):
    # Warmup
    for _ in range(warmups):
        func()
    torch.cuda.synchronize()

    ms = tt.do_bench(func, warmup=warmups, rep=runs)
    return ms


def run_benchmark(
    layer_idx,
    depth,
    branching_factor,
    update_every,
):
    # heads = 24
    offset = 81

    print("Loading key/queries...")
    keys = (
        torch.load("../cpu_search/keys.pt")[layer_idx, :]
        .unsqueeze(0)
        .to("cuda")
        .contiguous()
    )
    # keys = repeat_kv(keys, 3)
    # keys = keys[:, 0:heads, :, :]
    queries = (
        torch.load("../cpu_search/queries.pt")[layer_idx, :]
        .unsqueeze(0)
        .to("cuda")
        .contiguous()
    )
    # queries = queries[:, 0:heads, :, :]
    print(f"query shape: {queries.shape}, key shape: {keys.shape}")

    print("\tBuilding index...")
    indexer = CUDAIndexer(
        num_levels=depth,
        branching_factor=branching_factor,
        max_iterations=1,
    ).build(
        keys[:, :, :offset, :], values=keys[:, :, :offset, :]
    )  # only first prefilled keys

    result = {
        "position": [],
        "method": [],
        "time": [],
        "update_time": [],
        "num_levels": [],
        "branching_factor": [],
        "update_every": [],
    }

    for i in tqdm(range(offset, queries.size(2), update_every), desc="Querying"):
        query = queries[:, :, i : i + 1, :]

        update_time = 0
        if (i - offset) % update_every == 0:
            # tqdm.write("updating...")
            torch.cuda.synchronize()

            # Create CUDA events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            # Record start
            start.record()

            # Operation to measure
            indexer.update(
                keys[:, :, i - update_every : i, :],
                new_values=keys[:, :, i - update_every : i, :],
            )

            # Record end
            end.record()

            # Wait for GPU to finish
            torch.cuda.synchronize()
            update_time = start.elapsed_time(end) / update_every  # average time per key

            # tqdm.write("updating done.")

        # threshold
        rep_keys = repeat_kv(keys, 3)
        query = query / query.norm(dim=-1, keepdim=True)
        scores = (query * rep_keys).sum(dim=-1)
        # n-th largest = kth smallest of negative (shape = H)
        threshold = (-scores).kthvalue(k=20, dim=-1).values
        threshold = -threshold.squeeze(0)

        for method in config["methods"]:
            prunes = []
            prune_tmps = []

            searcher = CUDASearcher(block_c=branching_factor)

            if method == "brute_force":
                took = run_func(
                    lambda: torch.matmul(query, rep_keys[:, :, :i, :].transpose(-2, -1))
                )
                took -= update_time  # exclude update time for brute-force
            else:  # search
                took = run_func(
                    lambda: searcher.search(
                        query,
                        threshold,
                        indexer,
                    )
                )

            result["method"].append(method)
            result["time"].append(took + update_time)  # + amortized update time
            result["position"].append(i)
            result["update_time"].append(update_time)
            result["num_levels"].append(depth)
            result["branching_factor"].append(branching_factor)
            result["update_every"].append(update_every)

            # build from scratch
            # indexer_tmp = CUDAIndexer(
            #     depth=depth,
            #     branching_factor=branching_factor,
            #     max_iterations=1,
            # ).build(
            #     keys[:, :, :i, :]
            # )  # only first prefilled keys
            # output = searcher.synthetic_scanned_fraction(query, threshold, indexer_tmp)
            # prune_tmps.append(output["scanned_fraction_mean"])
            # output = searcher.synthetic_scanned_fraction(query, threshold, indexer)
            # prunes.append(output["scanned_fraction_mean"])
        # tqdm.write(f"pruning: {sum(prunes) / len(prunes):.6f}")

        # tqdm.write(
        # f"position={i} | pruning_ratio_v2={sum(prunes) / len(prunes):.6f} | pruning_ratio_v3={sum(prune_tmps) / len(prune_tmps):.6f}"
        # )

        # tqdm.write(
        #     f"parents.shape: {indexer.parents.shape}, grand_parents.shape: {indexer.grand_parents.shape if indexer.grand_parents is not None else -1}"
        # )

        # if i % (2**6) == 0:
        pd.DataFrame(result).to_csv(
            config["output_csv"], mode="a", header=False, index=False
        )

    return pd.DataFrame(result)


# %%
layer_idx = 20

df = None

# empty output
pd.DataFrame(
    {
        "position": [],
        "method": [],
        "time": [],
        "update_time": [],
        "num_levels": [],
        "branching_factor": [],
        "update_every": [],
    }
).to_csv(config["output_csv"], index=False)

for depth in config["num_levels"]:
    for branching_factor in config["branching_factor"]:
        for update_every in config["update_every"]:
            print(f"depth={depth} | bf={branching_factor} | update={update_every}")
            result = run_benchmark(
                layer_idx,
                depth,
                branching_factor,
                update_every,
            )
            if df is not None:
                df = pd.concat([df, result])
            else:
                df = result

            df.to_csv(config["output_csv"], mode="a", header=False, index=False)
