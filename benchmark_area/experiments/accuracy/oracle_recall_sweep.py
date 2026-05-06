"""
Oracle recall sweep: compare oracle configs vs fixed budget=10% on real captured keys.

Metric: recall@top-10% = fraction of true top-10% keys (by score) that
        the oracle threshold retrieves.
Also reports: retrieval fraction = total fraction of keys retrieved.

Usage:
    python oracle_recall_sweep.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from hira.benchmark_area.quick_pruning.pruning_bench_utils import CaptureState

CAPTURE = Path(__file__).resolve().parents[2] / "quick_pruning" / \
    "capture_qkv_12000_meta-llama_Llama-3.2-3B-Instruct.pt"

BUDGET_FRAC = 0.10      # reference budget
SAMPLE_SIZE = 256       # reservoir sample size
N_DECODE_STEPS = 200    # steps to average over (subsampled from available)
LAYERS = [5, 10, 15, 20]   # transformer layers to aggregate over

# ──────────────────────────────────────────────────────────────────────────────

def reservoir_sample(keys_so_far: torch.Tensor, sample_size: int, seed: int = 0) -> torch.Tensor:
    """keys_so_far: (H_kv, N, D). Returns (H_kv, M, D) reservoir sample."""
    H_kv, N, D = keys_so_far.shape
    M = min(sample_size, N)
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(N, generator=g)[:M]
    return keys_so_far[:, idx, :]


def true_scores(q: torch.Tensor, keys: torch.Tensor, q_head_to_kv: torch.Tensor) -> torch.Tensor:
    """
    q: (H_q, D), keys: (H_kv, N, D)
    Returns (H_q, N) dot products.
    """
    H_q, D = q.shape
    H_kv, N, _ = keys.shape
    k_for_q = keys[q_head_to_kv]  # (H_q, N, D)
    return torch.einsum("hd,hnd->hn", q.float(), k_for_q.float())


def sample_scores(q: torch.Tensor, sample: torch.Tensor, q_head_to_kv: torch.Tensor) -> torch.Tensor:
    """
    q: (H_q, D), sample: (H_kv, M, D)
    Returns (H_q, M) dot products.
    """
    H_kv, M, D = sample.shape
    s_for_q = sample[q_head_to_kv]  # (H_q, M, D)
    return torch.einsum("hd,hnd->hn", q.float(), s_for_q.float())


def threshold_mean_max(scores: torch.Tensor) -> torch.Tensor:
    """scores: (H_q, M) → (H_q,) threshold."""
    return (scores.max(dim=-1).values + scores.mean(dim=-1)) / 2



def recall_and_frac(true_sc: torch.Tensor, threshold: torch.Tensor, budget_frac: float):
    """
    true_sc: (H_q, N), threshold: (H_q,)
    Returns (recall, retrieval_frac) averaged over heads.
    """
    H_q, N = true_sc.shape
    k_budget = max(1, int(budget_frac * N))
    recalls, fracs = [], []
    for h in range(H_q):
        sc = true_sc[h]
        tau = threshold[h].item()
        # True top-k budget set
        top_idx = sc.topk(k_budget).indices
        # Oracle retrieved set
        retrieved = sc >= tau
        n_retrieved = retrieved.sum().item()
        # Recall: how many of top-k are retrieved
        n_correct = retrieved[top_idx].sum().item()
        recalls.append(n_correct / k_budget)
        fracs.append(n_retrieved / N)
    return sum(recalls) / len(recalls), sum(fracs) / len(fracs)


# ──────────────────────────────────────────────────────────────────────────────

def threshold_sample_max(scores: torch.Tensor) -> torch.Tensor:
    """scores: (H_q, M) → (H_q,) threshold = max sample score."""
    return scores.max(dim=-1).values


def threshold_budget(scores: torch.Tensor, budget_frac: float) -> torch.Tensor:
    """scores: (H_q, M) → (H_q,) threshold = k-th score from sample."""
    k = max(1, int(budget_frac * scores.shape[-1]))
    return scores.topk(k, dim=-1).values[:, -1]


CONFIGS = [
    # (name, fn)  fn(sample_sc) → (H_q,) threshold
    ("budget    f=0.10", lambda sc: threshold_budget(sc, 0.10)),
    ("sample_max      ", lambda sc: threshold_sample_max(sc)),
    ("mean_max        ", lambda sc: threshold_mean_max(sc)),
]

SAMPLE_SIZES = [32, 64, 128, 256, 512, 1024]


def run_sweep(cap: CaptureState, layers: list[int], n_steps: int, sample_size: int,
              device: torch.device):
    """Returns dict config_name → (mean_recall, mean_frac)."""
    results = {name: ([], []) for name, _ in CONFIGS}

    available_steps = cap.generated_token_count()
    step_stride = max(1, available_steps // n_steps)
    steps_to_eval = list(range(0, available_steps, step_stride))[:n_steps]

    for layer in layers:
        if layer not in cap.prefill_keys:
            continue
        prefill_k = cap.prefill_keys[layer].to(device=device, dtype=torch.float32)  # (H_kv, N_pre, D)
        gen_q_raw = cap.generated_queries[layer]
        gen_k_raw = cap.generated_keys[layer]
        H_kv = prefill_k.shape[0]

        # Pre-stack decode keys on GPU
        if gen_k_raw:
            gen_k_gpu = torch.stack(gen_k_raw, dim=1).to(device=device, dtype=torch.float32)  # (H_kv, T, D)
        else:
            gen_k_gpu = None

        for step in steps_to_eval:
            if step >= len(gen_q_raw):
                break
            q = gen_q_raw[step].to(device=device, dtype=torch.float32)  # (H_q, D)
            H_q = q.shape[0]
            q_head_to_kv = torch.arange(H_q, device=device) // (H_q // H_kv)

            # Build full key set up to this step
            if step > 0 and gen_k_gpu is not None:
                all_keys = torch.cat([prefill_k, gen_k_gpu[:, :step, :]], dim=1)
            else:
                all_keys = prefill_k

            N = all_keys.shape[1]
            if N < 10:
                continue

            true_sc = true_scores(q, all_keys, q_head_to_kv)

            # Build reservoir sample from all keys (on GPU)
            M = min(sample_size, N)
            idx = torch.randperm(N, device=device)[:M]
            sample = all_keys[:, idx, :]
            samp_sc = sample_scores(q, sample, q_head_to_kv)

            for name, fn in CONFIGS:
                tau = fn(samp_sc)
                r, f = recall_and_frac(true_sc, tau, BUDGET_FRAC)
                results[name][0].append(r)
                results[name][1].append(f)

    return {
        name: (
            sum(recalls) / len(recalls) if recalls else 0.0,
            sum(fracs) / len(fracs) if fracs else 0.0,
        )
        for name, (recalls, fracs) in results.items()
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {CAPTURE.name} ...")
    cap = CaptureState.load(CAPTURE)
    layers_avail = cap.layer_ids()
    layers = [l for l in LAYERS if l in layers_avail]
    if not layers:
        layers = layers_avail[::max(1, len(layers_avail) // 4)][:4]
    print(f"Layers: {layers}  decode_steps_available: {cap.generated_token_count()}")
    print(f"Evaluating {N_DECODE_STEPS} steps × {len(layers)} layers per config\n")

    # Run all sample sizes at once
    all_res = {}
    for ss in SAMPLE_SIZES:
        print(f"  sample_size={ss}...", flush=True)
        all_res[ss] = run_sweep(cap, layers, N_DECODE_STEPS, ss, device)

    # Print table: all configs at default sample size, then mean_max+budget across sample sizes
    print(f"\n── All configs, sample_size={SAMPLE_SIZE} ──")
    print(f"{'Config':<22}  {'Recall@top10%':>14}  {'Retrieved%':>10}")
    print("-" * 50)
    for name, _ in CONFIGS:
        r, f = all_res[SAMPLE_SIZE][name]
        marker = " ★" if r >= 0.80 else ""
        print(f"{name:<22}  {r*100:>13.1f}%  {f*100:>9.1f}%{marker}")

    print(f"\n── mean_max and budget vs sample_size ──")
    print(f"{'Config':<22}  {'sample_size':>11}  {'Recall@top10%':>14}  {'Retrieved%':>10}")
    print("-" * 65)
    for ss in SAMPLE_SIZES:
        for name, _ in CONFIGS:
            if "mean_max" not in name and "budget" not in name:
                continue
            r, f = all_res[ss][name]
            marker = " ★" if r >= 0.80 else ""
            print(f"{name:<22}  {ss:>11}  {r*100:>13.1f}%  {f*100:>9.1f}%{marker}")


if __name__ == "__main__":
    main()
