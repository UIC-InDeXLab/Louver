"""
Recall benchmark: Louver halfspace range search vs ANN and sparse-attention methods.

Phase 1 — Index recall (budget = k retrieved keys):
    louver          Halfspace range search — zero false-negative guarantee
    hnsw            HNSW (RetrievalAttention)
    ivf             IVF clustering (InfLLM)
    pq              Product quantization (PQCache)
    lsh             Random-projection LSH (MagicPIG)

Phase 2 — Sparse-attention retrieval recall at budget = k:
    louver          Oracle threshold → exactly the true top-k
    quest           Page-level max-score proxy
    streamingllm    Attention sinks (first 4) + recent window
    twilight        Softmax top-p=0.85 cumulative mass

Metric: recall@k = |retrieved ∩ true_top_k| / k
k ∈ {10, 20, 50, 100}

Speed: ANN indices built once per KV head (not per query). Full key set used.

Usage:
    python recall_bench.py --input-qkv ../latency/captures/llama*.pt ../latency/captures/qwen*.pt
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("FAISS_NOISY", "1")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import types as _types
_hira = _types.ModuleType("hira")
_hira.__path__ = [str(REPO_ROOT)]
_hira.__package__ = "hira"
sys.modules["hira"] = _hira

from benchmark_area.quick_pruning.pruning_bench_utils import CaptureState, _q_to_kv_map

import faiss

K_VALUES   = [10, 20, 50, 100]
QUEST_PAGE = 16
SINKS      = 4
TOP_P      = 0.85
BF         = 4   # Louver branching factor (clusters per BF keys)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── GPU helpers ────────────────────────────────────────────────────────────────

def to_gpu(t: torch.Tensor) -> torch.Tensor:
    return t.to(DEVICE, dtype=torch.float32)


def exact_scores_and_topk(q_all: torch.Tensor, keys: torch.Tensor,
                           q_to_kv: list[int], k_max: int):
    """
    q_all: (H_q, D)  on GPU
    keys:  (H_kv, N, D)  on GPU
    Returns scores (H_q, N), top_idx (H_q, k_max), top_vals (H_q, k_max) — all on GPU.
    """
    H_q, D   = q_all.shape
    H_kv, N, _ = keys.shape
    scores = torch.empty(H_q, N, device=DEVICE, dtype=torch.float32)
    for h_kv in range(H_kv):
        hq_slice = [h for h, kv in enumerate(q_to_kv) if kv == h_kv]
        if not hq_slice:
            continue
        scores[hq_slice] = q_all[hq_slice] @ keys[h_kv].T   # (G, N)
    k_eff = min(k_max, N)
    top_vals, top_idx = scores.topk(k_eff, dim=1)
    return scores, top_idx, top_vals


# ── ANN index build (once per KV head) ────────────────────────────────────────

def _norm_np(k_np: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(k_np, axis=1, keepdims=True) + 1e-12
    return k_np / norms


def build_hnsw(k_np: np.ndarray, M: int = 32) -> faiss.IndexHNSWFlat:
    n, d = k_np.shape
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efSearch = 64
    idx.add(_norm_np(k_np))
    return idx


def build_ivf(k_np: np.ndarray) -> faiss.IndexIVFFlat:
    n, d = k_np.shape
    nlist = max(1, min(int(math.sqrt(n)), n // 4))
    q     = faiss.IndexFlatIP(d)
    idx   = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(k_np)
    idx.add(k_np)
    idx.nprobe = max(1, nlist // 4)
    return idx


def build_pq(k_np: np.ndarray) -> faiss.IndexPQ | None:
    n, d = k_np.shape
    M = 8
    while d % M != 0:
        M //= 2
    nbits = 8
    if n < (1 << nbits):
        return None
    idx = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
    idx.train(k_np)
    idx.add(k_np)
    return idx


def build_lsh(k_np: np.ndarray, n_planes: int = 64, seed: int = 42) -> dict:
    rng    = np.random.RandomState(seed)
    d      = k_np.shape[1]
    planes = rng.randn(n_planes, d).astype(np.float32)
    planes /= np.linalg.norm(planes, axis=1, keepdims=True) + 1e-12
    key_bits = (k_np @ planes.T) >= 0    # (N, n_planes)
    return {"planes": planes, "key_bits": key_bits}


def build_louver(k_np: np.ndarray) -> dict:
    """Kmeans clustering + radii for halfspace filter."""
    n, d       = k_np.shape
    n_centers  = max(1, n // BF)
    km = faiss.Kmeans(d, n_centers, niter=20, verbose=False)
    km.train(k_np)
    centers = km.centroids                               # (K, D)

    flat = faiss.IndexFlatL2(d)
    flat.add(centers)
    _, assigns = flat.search(k_np, 1)
    assigns    = assigns[:, 0]                           # (N,)

    radii = np.zeros(len(centers), dtype=np.float32)
    for c in range(len(centers)):
        mask = assigns == c
        if mask.sum() > 0:
            diffs    = k_np[mask] - centers[c]
            radii[c] = float(np.sqrt((diffs ** 2).sum(1).max()))

    return {"centers": centers, "radii": radii, "assigns": assigns}


# ── Per-query recall functions ─────────────────────────────────────────────────

def _topk_set(top_idx: torch.Tensor, k: int) -> set:
    return set(top_idx[:k].cpu().tolist())


def query_louver(idx: dict, q_np: np.ndarray,
                 scores_np: np.ndarray, top_idx: torch.Tensor, k: int) -> float:
    n = len(scores_np)
    if n < k:
        return 1.0
    top_k     = _topk_set(top_idx, k)
    threshold = float(np.partition(scores_np, -k)[-k])
    centers, radii, assigns = idx["centers"], idx["radii"], idx["assigns"]
    q_norm    = float(np.linalg.norm(q_np))
    c_scores  = centers @ q_np                       # (K,)
    keep      = (c_scores + q_norm * radii) >= threshold
    retrieved = set(np.where(keep[assigns])[0].tolist())
    return len(retrieved & top_k) / k


def query_hnsw(idx: faiss.IndexHNSWFlat, q_np: np.ndarray,
               top_idx: torch.Tensor, k: int) -> float:
    top_k  = _topk_set(top_idx, k)
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-12)
    idx.hnsw.efSearch = max(64, k * 2)
    _, I = idx.search(q_norm.reshape(1, -1), k)
    return len(set(I[0].tolist()) & top_k) / k


def query_ivf(idx: faiss.IndexIVFFlat, q_np: np.ndarray,
              top_idx: torch.Tensor, k: int) -> float:
    top_k = _topk_set(top_idx, k)
    _, I  = idx.search(q_np.reshape(1, -1), k)
    return len(set(I[0].tolist()) & top_k) / k


def query_pq(idx, q_np: np.ndarray,
             top_idx: torch.Tensor, k: int) -> float:
    if idx is None:
        return 1.0
    top_k = _topk_set(top_idx, k)
    _, I  = idx.search(q_np.reshape(1, -1), k)
    return len(set(I[0].tolist()) & top_k) / k


def query_lsh(idx: dict, q_np: np.ndarray,
              top_idx: torch.Tensor, k: int) -> float:
    top_k    = _topk_set(top_idx, k)
    q_bits   = (q_np @ idx["planes"].T) >= 0
    hamming  = (idx["key_bits"] != q_bits).sum(axis=1)
    n_ret    = min(2 * k, len(hamming))
    retrieved = set(np.argsort(hamming)[:n_ret].tolist())
    return len(retrieved & top_k) / k


def quest_recall(scores_np: np.ndarray, top_idx: torch.Tensor, k: int) -> float:
    """Page-level: score page by max score, select top pages."""
    n = len(scores_np)
    if n < k:
        return 1.0
    top_k   = _topk_set(top_idx, k)
    n_pages = math.ceil(n / QUEST_PAGE)
    page_sc = np.array([scores_np[p*QUEST_PAGE : min((p+1)*QUEST_PAGE, n)].max()
                        for p in range(n_pages)], dtype=np.float32)
    n_pages_needed = max(1, math.ceil(k / QUEST_PAGE))
    top_pages      = np.argsort(page_sc)[-n_pages_needed:]
    retrieved = set()
    for p in top_pages:
        s, e = p * QUEST_PAGE, min((p + 1) * QUEST_PAGE, n)
        retrieved.update(range(s, e))
    return len(retrieved & top_k) / k


def streamingllm_recall(top_idx: torch.Tensor, k: int, n: int) -> float:
    if n < k:
        return 1.0
    top_k    = _topk_set(top_idx, k)
    n_recent = max(0, k - SINKS)
    retrieved = set(range(min(SINKS, n)))
    if n_recent > 0:
        retrieved.update(range(max(SINKS, n - n_recent), n))
    return len(retrieved & top_k) / k


def twilight_recall(scores_np: np.ndarray, top_idx: torch.Tensor, k: int) -> float:
    n = len(scores_np)
    if n < k:
        return 1.0
    top_k = _topk_set(top_idx, k)
    probs = torch.softmax(torch.from_numpy(scores_np).float(), dim=0).numpy()
    order = np.argsort(probs)[::-1]
    cum   = 0.0
    retrieved = set()
    for i in order:
        retrieved.add(int(i))
        cum += float(probs[i])
        if cum >= TOP_P:
            break
    return len(retrieved & top_k) / k


# ── Per-capture benchmark ──────────────────────────────────────────────────────

def run_capture(cap: "CaptureState", layer: int,
                n_samples: int, rng: np.random.RandomState) -> dict:
    from tqdm import tqdm

    queries_cpu, keys_cpu, _ = cap.to_layer_tensors(layer)
    queries = queries_cpu.float()   # (H_q, N, D)
    keys    = keys_cpu.float()      # (H_kv, N, D)

    H_q, N, D = queries.shape
    H_kv      = keys.shape[0]
    q_to_kv   = [h * H_kv // H_q for h in range(H_q)]

    n_prefill = int(cap.prompt_length) if cap.prompt_length else max(1, N // 20)
    k_max     = max(K_VALUES)
    min_n     = k_max + 10

    valid = [t for t in range(n_prefill, N) if t >= min_n]
    if not valid:
        print("  [warn] no valid positions")
        return {}

    n_samples = min(n_samples, len(valid))
    positions = sorted(rng.choice(valid, n_samples, replace=False).tolist())

    # ── Build ANN indices once per KV head (full key set) ──────────────────────
    print(f"  Building indices for {H_kv} KV heads (N={N}) ...")
    keys_np = keys.numpy()   # (H_kv, N, D) — CPU
    ann = {}
    for h_kv in tqdm(range(H_kv), desc="  build", leave=False):
        k_np = keys_np[h_kv]   # (N, D)
        ann[h_kv] = {
            "louver": build_louver(k_np),
            "hnsw":   build_hnsw(k_np),
            "ivf":    build_ivf(k_np),
            "pq":     build_pq(k_np),
            "lsh":    build_lsh(k_np),
        }

    # ── GPU: exact scores for all sample queries ────────────────────────────────
    keys_gpu = to_gpu(keys)   # (H_kv, N, D)

    methods = ["louver", "hnsw", "ivf", "pq", "lsh",
               "quest", "streamingllm", "twilight"]
    results = {m: {k: [] for k in K_VALUES} for m in methods}

    for t in tqdm(positions, desc="  queries", leave=True, unit="q"):
        q_all = to_gpu(queries[:, t, :])              # (H_q, D)
        scores_all, top_idx_all, _ = exact_scores_and_topk(
            q_all, keys_gpu, q_to_kv, k_max)

        for h_q in range(H_q):
            h_kv     = q_to_kv[h_q]
            q_np     = q_all[h_q].cpu().numpy()
            sc_np    = scores_all[h_q].cpu().numpy()  # (N,)
            ti_h     = top_idx_all[h_q]               # (k_max,)
            a        = ann[h_kv]

            for k in K_VALUES:
                if N < k:
                    continue
                results["louver"][k].append(
                    query_louver(a["louver"], q_np, sc_np, ti_h, k))
                results["hnsw"][k].append(
                    query_hnsw(a["hnsw"], q_np, ti_h, k))
                results["ivf"][k].append(
                    query_ivf(a["ivf"], q_np, ti_h, k))
                results["pq"][k].append(
                    query_pq(a["pq"], q_np, ti_h, k))
                results["lsh"][k].append(
                    query_lsh(a["lsh"], q_np, ti_h, k))
                results["quest"][k].append(
                    quest_recall(sc_np, ti_h, k))
                results["streamingllm"][k].append(
                    streamingllm_recall(ti_h, k, N))
                results["twilight"][k].append(
                    twilight_recall(sc_np, ti_h, k))

    return results


def merge(a: dict, b: dict) -> dict:
    if not a:
        return b
    return {m: {k: a[m][k] + b[m][k] for k in K_VALUES} for m in a}


# ── Output ─────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "louver":       "Louver (ours)",
    "hnsw":         "HNSW [RetrievalAttn]",
    "ivf":          "IVF  [InfLLM]",
    "pq":           "PQ   [PQCache]",
    "lsh":          "LSH  [MagicPIG]",
    "quest":        "Quest",
    "streamingllm": "StreamingLLM",
    "twilight":     "Twilight",
}

PHASE1 = ["louver", "hnsw", "ivf", "pq", "lsh"]
PHASE2 = ["louver", "quest", "streamingllm", "twilight"]


def print_table(results, methods, title):
    def mu(lst): return sum(lst) / len(lst) if lst else float("nan")
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")
    hdr = f"  {'Method':<26}" + "".join(f"  recall@{k:<5}" for k in K_VALUES)
    print(hdr)
    for m in methods:
        row = f"  {METHOD_LABELS[m]:<26}"
        for k in K_VALUES:
            row += f"  {mu(results.get(m, {}).get(k, [])):.3f}       "
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-qkv", type=Path, nargs="+", required=True)
    p.add_argument("--n-samples", type=int, default=100,
                   help="Random decode queries per capture (default 100).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.RandomState(args.seed)
    print(f"Device: {DEVICE}  n_samples={args.n_samples}")

    all_res: dict = {}
    stems = []

    for pt in args.input_qkv:
        print(f"\n=== {pt.name} ===")
        cap   = CaptureState.load(pt)
        layers = cap.layer_ids()
        layer  = layers[len(layers) // 2]
        print(f"Layer {layer}  H_q={cap.to_layer_tensors(layer)[0].shape[0]}")
        res   = run_capture(cap, layer, args.n_samples, rng)
        all_res = merge(all_res, res)
        stems.append(pt.stem[:16])

    if not all_res:
        print("No results."); return

    print_table(all_res, PHASE1, "Phase 1 — Index Recall")
    print_table(all_res, PHASE2, "Phase 2 — Sparse-Attention Recall (budget = k)")

    stem_str = "_".join(stems)
    out = args.output_csv or (
        Path(__file__).parent / "reports" / f"recall_{stem_str}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    def mu(lst): return round(sum(lst) / len(lst), 4) if lst else float("nan")

    rows = []
    for m in (PHASE1 + [m for m in PHASE2 if m not in PHASE1]):
        phase = "ann" if m in PHASE1[1:] else ("sparse" if m in PHASE2[1:] else "louver")
        for k in K_VALUES:
            vals = all_res.get(m, {}).get(k, [])
            rows.append({"method": m, "phase": phase, "k": k,
                         "recall_mean": mu(vals), "n": len(vals)})

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "phase", "k", "recall_mean", "n"])
        w.writeheader(); w.writerows(rows)
    print(f"\nCSV → {out}")


if __name__ == "__main__":
    main()
