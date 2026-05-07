# %%
import sys

sys.path.append("../")
sys.path.append("../../")

from thresholding.helpers import ObserveAttentionHelper
from hira.indexer import CPUIndexer
from hira.searcher import CPUSearcher

import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import random
from tqdm import tqdm
import pandas as pd

# %% [markdown]
# ### Benchmarking

# %%
# config
config = {
    # indexer
    "num_levels": [5],
    "branching_factor": [8],
    "balance_every": [2**9],
    "centroid_refine_iters": [0],
    # general
    "update_every": [2**9],
    "output_csv": "result_v1.0_values.csv",
    # searcher
    "chunk_size": [8 * 1024],
    "methods": [
        # "search_numba_vectorized",
        # "search_torch_vectorized",
        # "search_cpp_vectorized",
        # "search_numba_loop",
        # "search_torch_loop",
        # "search_cpp_loop",
        # "search_fused",
        "brute_force",
        # "search_fused_torch_ext",
        "fused_v1",
        "fused_v2",
        "fused_v3",
        "fused_v4",
        # "search_exact_torch_ext",
    ],
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


def run_benchmark(
    layer_idx,
    observer,
    num_levels,
    branching_factor,
    balance_every,
    update_every,
    centroid_refine_iters,
):
    # heads = 24
    offset = 81

    print("Loading key/queries...")
    keys = torch.load("keys.pt")[layer_idx, :].unsqueeze(0).to("cpu").contiguous()
    # keys = repeat_kv(keys, 3)
    # keys = keys[:, 0:heads, :, :]
    queries = torch.load("queries.pt")[layer_idx, :].unsqueeze(0).to("cpu").contiguous()
    # queries = queries[:, 0:heads, :, :]
    print(f"query shape: {queries.shape}, key shape: {keys.shape}")

    print("\tBuilding index...")
    indexer = CPUIndexer(
        num_levels=num_levels,
        branching_factor=branching_factor,
        max_iterations=1,
        centroid_refine_iters=centroid_refine_iters,
    ).build(
        keys[:, :, :offset, :], values=keys[:, :, :offset, :]
    )  # only first prefilled keys
    # indexer_1.update_v2_use_faiss_kernel = True
    # # different updates
    # indexer_2 = CPUIndexer(
    #     num_levels=num_levels,
    #     branching_factor=branching_factor,
    #     max_iterations=1,
    #     balance_every=balance_every,
    #     centroid_refine_iters=centroid_refine_iters,
    # ).build(keys[:, :, :offset, :])

    result = {
        "position": [],
        "method": [],
        "time": [],
        "chunk_size": [],
        "update_time": [],
        "num_levels": [],
        "branching_factor": [],
        "balance_every": [],
        "update_every": [],
        "centroid_refine_iters": [],
    }

    for i in tqdm(range(offset, queries.size(2), update_every), desc="Querying"):
        query = queries[:, :, i : i + 1, :]

        update_time = 0
        if (i - offset) % update_every == 0:
            # tqdm.write("updating...")
            # update_start = time.time()
            # indexer_1.update_v2(keys[:, :, i - update_every : i, :])
            # update_end = time.time()
            # update_time_v2 = (update_end - update_start) / update_every

            update_start = time.time()
            indexer.update(
                keys[:, :, i - update_every : i, :],
                new_values=keys[:, :, i - update_every : i, :],
            )
            update_end = time.time()
            update_time = (update_end - update_start) / update_every

            # tqdm.write("updating done.")
            # indexer = CPUIndexer(
            #     num_levels=num_levels,
            #     branching_factor=branching_factor,
            #     max_iterations=1,
            #     balance_every=balance_every,
            # ).build(
            #     keys[:, :, :i , :]
            # )  # only first prefilled keys
            # update_time = 0

        # threshold
        rep_keys = repeat_kv(keys, 3)
        query = query / query.norm(dim=-1, keepdim=True)
        scores = (query * rep_keys).sum(dim=-1)
        # n-th largest = kth smallest of negative (shape = H)
        threshold = (-scores).kthvalue(k=20, dim=-1).values
        threshold = -threshold.squeeze(0)

        for chunk_size in config["chunk_size"]:
            prunes = []
            prunes_tmp = []
            prunes_time = []
            prunes_time_tmp = []

            for method in config["methods"]:
                # vectorize = "vectorized" in method
                # kernel = (
                #     "numba"
                #     if "numba" in method
                #     else "cpp" if "cpp" in method else "torch"
                # )

                # if method == "search_fused":
                #     for _ in range(10):  # warmup
                #         searcher.search_fused(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_fused(query, threshold, indexer_2)
                #     end = time.time()
                # elif method == "search_fused_torch_ext":
                #     for _ in range(10):  # warmup
                #         searcher.search_fused_torch_ext(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_fused_torch_ext(query, threshold, indexer_2)
                #     end = time.time()
                # elif method == "search_fused_torch_ext_v2":
                #     for _ in range(10):  # warmup
                #         searcher.search_fused_torch_ext_v2(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_fused_torch_ext_v2(query, threshold, indexer_2)
                #     end = time.time()
                # elif method == "search_fused_torch_ext_v3":
                #     for _ in range(10):  # warmup
                #         searcher.search_fused_torch_ext_v3(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_fused_torch_ext_v3(query, threshold, indexer_2)
                #     end = time.time()
                # elif method == "search_fused_torch_ext_v4":
                #     for _ in range(10):  # warmup
                #         searcher.search_fused_torch_ext_v4(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_fused_torch_ext_v4(query, threshold, indexer_2)
                #     end = time.time()
                # elif method == "search_exact_torch_ext":
                #     for _ in range(10):  # warmup
                #         searcher.search_exact_torch_ext(query, threshold, indexer_2)

                #     start = time.time()
                #     searcher.search_exact_torch_ext(query, threshold, indexer_2)
                #     end = time.time()
                if method == "brute_force":
                    for _ in range(10):  # warmup
                        query = query / query.norm(dim=-1, keepdim=True)
                        scores = torch.matmul(
                            query, rep_keys[:, :, :i, :].transpose(-2, -1)
                        )

                    start = time.time()
                    query = query / query.norm(dim=-1, keepdim=True)
                    scores = torch.matmul(
                        query, rep_keys[:, :, :i, :].transpose(-2, -1)
                    )
                    end = time.time()
                    end -= update_time  # exclude update time for brute-force
                else:  # search
                    searcher = CPUSearcher(
                        chunk_size=chunk_size, search_strategy=method
                    )

                    for _ in range(10):  # warmup
                        searcher.search(query, threshold, indexer)

                    start = time.time()
                    searcher.search(query, threshold, indexer)
                    end = time.time()

                # report pruning
                # searcher = CPUSearcher(
                #     chunk_size=chunk_size, kernel="cpp", profiling=True
                # )
                # for _ in range(10):  # warmup
                #     searcher.search_fused_torch_ext_v3(query, threshold, indexer_1)
                # t = time.time()
                # searcher.search_fused_torch_ext_v3(query, threshold, indexer_1)
                # t = time.time() - t
                # searcher.search(query, threshold, indexer_1)
                # exact_checks = float(searcher.stats["exact_checks"])
                # h = int(query.size(1))
                # n = int(indexer_1.num_keys)
                # pruning_ratio = exact_checks / (h * i)
                # # print("pruning ratio: {:.6f} | time: {:.6f}".format(pruning_ratio, t))
                # prunes.append(pruning_ratio)
                # prunes_time.append(t)

                # # indexer_tmp = CPUIndexer(
                # #     num_levels=num_levels,
                # #     branching_factor=branching_factor,
                # #     max_iterations=1,
                # #     balance_every=balance_every,
                # #     centroid_refine_iters=centroid_refine_iters,
                # # ).build(
                # #     keys[:, :, :i, :].contiguous()
                # # )  # only first prefilled
                # for _ in range(10):  # warmup
                #     searcher.search_fused_torch_ext_v3(query, threshold, indexer_2)
                # t = time.time()
                # searcher.search_fused_torch_ext_v3(query, threshold, indexer_2)
                # t = time.time() - t
                # searcher.search(query, threshold, indexer_2)
                # exact_checks_tmp = float(searcher.stats["exact_checks"])
                # pruning_ratio_tmp = (exact_checks_tmp / (h * i)) if i > 0 else 0.0
                # # print("pruning ratio (tmp): {:.6f} |\t time: {:.6f}".format(pruning_ratio_tmp, t))
                # prunes_tmp.append(pruning_ratio_tmp)
                # prunes_time_tmp.append(t)

                #

                result["method"].append(method)
                result["chunk_size"].append(chunk_size)
                result["time"].append(
                    end - start + update_time
                )  # + amortized update time
                result["position"].append(i)
                # result["update_time_v2"].append(update_time_v2)
                result["update_time"].append(update_time)
                result["num_levels"].append(num_levels)
                result["branching_factor"].append(branching_factor)
                result["balance_every"].append(balance_every)
                result["update_every"].append(update_every)
                result["centroid_refine_iters"].append(centroid_refine_iters)

            # print(
            #     f"position={i} | pruning_ratio_v2={sum(prunes) / len(prunes):.6f} | pruning_ratio_v3={sum(prunes_tmp) / len(prunes_tmp):.6f} | time_v2={sum(prunes_time) / len(prunes_time):.6f}s | time_v3={sum(prunes_time_tmp) / len(prunes_time_tmp):.6f}s"
            # )

        # if i % (2**6) == 0:
        pd.DataFrame(result).to_csv(config["output_csv"], index=False)

    return pd.DataFrame(result)


# %%
layer_idx = 20

df = None

for num_levels in config["num_levels"]:
    for branching_factor in config["branching_factor"]:
        for update_every in config["update_every"]:
            for balance_every in config["balance_every"]:
                for centroid_refine_iters in config["centroid_refine_iters"]:
                    print(
                        f"levels={num_levels} | bf={branching_factor} | update={update_every} | "
                        f"balance={balance_every} | refine={centroid_refine_iters}"
                    )
                    result = run_benchmark(
                        layer_idx,
                        None,
                        num_levels,
                        branching_factor,
                        balance_every,
                        update_every,
                        centroid_refine_iters,
                    )
                    if df is not None:
                        df = pd.concat([df, result])
                    else:
                        df = result

                    df.to_csv(config["output_csv"], index=False)
