import argparse
import time

import faiss
import numpy as np


def run_once(x: np.ndarray, k: int, niter: int, seed: int) -> tuple[float, np.ndarray]:
    t0 = time.perf_counter()
    km = faiss.Kmeans(
        d=x.shape[1],
        k=k,
        niter=niter,
        verbose=False,
        gpu=True,
        seed=seed,
        min_points_per_centroid=1,
    )
    km.train(x)
    _, idx = km.index.search(x, 1)
    dt = time.perf_counter() - t0
    return dt, idx.reshape(-1)


def main():
    parser = argparse.ArgumentParser(description="FAISS GPU k-means smoke benchmark")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--k", type=int, default=1_250)
    parser.add_argument("--niter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    print(f"faiss_version={getattr(faiss, '__version__', 'unknown')}")
    ngpu = faiss.get_num_gpus()
    print(f"faiss_num_gpus={ngpu}")
    if ngpu < 1:
        raise RuntimeError("No FAISS GPUs visible. Cannot run GPU k-means smoke test.")

    if args.k > args.n:
        raise ValueError(f"k must be <= n, got k={args.k}, n={args.n}")

    rng = np.random.default_rng(args.seed)
    x_all = rng.standard_normal((args.heads, args.n, args.d), dtype=np.float32)

    times = []
    for h in range(args.heads):
        dt, idx = run_once(x_all[h], k=args.k, niter=args.niter, seed=args.seed + h)
        times.append(dt)
        print(
            f"head={h} time_s={dt:.4f} assign_min={int(idx.min())} assign_max={int(idx.max())}"
        )

    total_points = args.heads * args.n
    total_time = sum(times)
    print(
        f"total_heads={args.heads} total_points={total_points} total_time_s={total_time:.4f}"
    )
    print(f"avg_time_per_head_s={np.mean(times):.4f}")
    print(f"throughput_points_per_s={total_points / total_time:.2f}")


if __name__ == "__main__":
    main()
