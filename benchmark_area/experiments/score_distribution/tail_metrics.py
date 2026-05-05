"""Fast batched per-step metrics. One matmul per layer, no per-step Python loop.

For each generated token t: build (H_total, T_t) softmax across selected layers,
compute top-k mass, eff-size, entropy, frac-above-uniform.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
FIXED_K_CHAL = ROOT.parent.parent / "fixed_k_chal"
sys.path.insert(0, str(FIXED_K_CHAL))

from helpers import ObserveAttentionHelper  # noqa: E402


def build_layer_tensors(helper, layers, device):
    """Returns list of (Q_layer, K_layer, ratio) on device.
    Q_layer: (Hq, G, D), K_layer: (KV, T_total, D)."""
    out = []
    for L in layers:
        kvh_keys = sorted(helper.keys[L].keys())
        Tmax = max(max(helper.keys[L][k].keys()) for k in kvh_keys) + 1
        D = next(iter(helper.keys[L][kvh_keys[0]].values())).shape[0]
        K = torch.zeros(len(kvh_keys), Tmax, D, dtype=torch.float32)
        for i, kvh in enumerate(kvh_keys):
            for pos, vec in helper.keys[L][kvh].items():
                K[i, pos] = vec.float()

        qh_keys = sorted(helper.queries[L].keys())
        G = max(max(helper.queries[L][q].keys()) for q in qh_keys) + 1
        Q = torch.zeros(len(qh_keys), G, D, dtype=torch.float32)
        for i, qh in enumerate(qh_keys):
            for tt, vec in helper.queries[L][qh].items():
                Q[i, tt] = vec.float()

        ratio = len(qh_keys) // len(kvh_keys)
        out.append((Q.to(device), K.to(device), ratio))
    return out


def collect_fast(helper, layers, device, topks=(8, 32, 128), g_chunk=512):
    """Layer-by-layer + g-chunked. Per-token metrics aggregated across heads of all layers."""
    bundles = build_layer_tensors(helper, layers, device)
    G = bundles[0][0].shape[1]
    prompt_len = helper.prompt_length

    # Per-token accumulators across layers: keep per-head min/max via running reduce.
    # For min-across-heads / max-across-heads: track running min/max across all heads of all layers.
    # For mean-across-heads: track sum and head count.
    init = lambda: torch.full((G,), float("nan"))
    metrics = {}
    H_total = 0

    def alloc(name, init_val):
        if name not in metrics:
            metrics[name] = torch.full((G,), init_val, dtype=torch.float32, device=device)

    # init keys
    for k in topks:
        alloc(f"topk_mass_{k}_min", float("inf"))
        alloc(f"topk_mass_{k}_max", float("-inf"))
        alloc(f"topk_mass_{k}_sum", 0.0)
    alloc("eff_size_max", float("-inf"))
    alloc("eff_size_sum", 0.0)
    alloc("entropy_max", float("-inf"))
    alloc("entropy_sum", 0.0)
    alloc("above_max", float("-inf"))
    alloc("above_sum", 0.0)
    # NEW: 95% coverage k — number of tokens needed to cover 95% mass (per-step).
    # Computed twice: weight (after softmax) + score (raw scores normalized to a distribution).
    for tag in ("weight", "score"):
        alloc(f"cov50_{tag}_min", float("inf"))
        alloc(f"cov50_{tag}_max", float("-inf"))
        alloc(f"cov50_{tag}_sum", 0.0)

    T_arr = torch.tensor([prompt_len + t for t in range(G)], dtype=torch.float32)

    for li, (Q, K, ratio) in enumerate(tqdm(bundles, desc="layers")):
        K_exp = K.repeat_interleave(ratio, dim=0)        # (Hq, T_total, D)
        D = Q.shape[-1]
        Hq = Q.shape[0]
        T_total = K_exp.shape[1]
        H_total += Hq

        # build a global causal mask once (reuse across chunks) on device
        # mask[g, t] = -inf if t >= prompt_len + g
        # For each g, valid prefix = [0, prompt_len + g)
        for g0 in tqdm(range(0, G, g_chunk), desc=f"layer{li}", leave=False):
            g1 = min(G, g0 + g_chunk)
            Q_chunk = Q[:, g0:g1, :]                      # (Hq, gc, D)
            scores = torch.einsum("hgd,htd->hgt", Q_chunk, K_exp) / (D ** 0.5)
            # mask: invalid positions -> -inf
            t_idx = torch.arange(T_total, device=device).view(1, 1, -1)
            valid_len = (prompt_len + torch.arange(g0, g1, device=device)).view(1, -1, 1)
            mask = t_idx >= valid_len
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)         # (Hq, gc, T_total)

            # per-head metrics over T_total (zeros on invalid positions don't contribute)
            for k in topks:
                topk_v = torch.topk(probs, min(k, T_total), dim=-1).values.sum(-1)  # (Hq, gc)
                cur_min = topk_v.min(0).values
                cur_max = topk_v.max(0).values
                cur_sum = topk_v.sum(0)
                metrics[f"topk_mass_{k}_min"][g0:g1] = torch.minimum(metrics[f"topk_mass_{k}_min"][g0:g1], cur_min)
                metrics[f"topk_mass_{k}_max"][g0:g1] = torch.maximum(metrics[f"topk_mass_{k}_max"][g0:g1], cur_max)
                metrics[f"topk_mass_{k}_sum"][g0:g1] += cur_sum

            eff = 1.0 / (probs.pow(2).sum(-1) + 1e-12)    # (Hq, gc)
            metrics["eff_size_max"][g0:g1] = torch.maximum(metrics["eff_size_max"][g0:g1], eff.max(0).values)
            metrics["eff_size_sum"][g0:g1] += eff.sum(0)

            ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)
            metrics["entropy_max"][g0:g1] = torch.maximum(metrics["entropy_max"][g0:g1], ent.max(0).values)
            metrics["entropy_sum"][g0:g1] += ent.sum(0)

            # ----- 95% coverage k computations -----
            valid_mask = ~mask  # (1, gc, T_total) broadcast
            valid_mask_b = valid_mask.expand(Hq, -1, -1)

            # weight version: probs already mask invalid → 0
            valid_len_f = valid_len.squeeze(-1).float()  # (1, gc)
            sorted_p, _ = probs.sort(dim=-1, descending=True)
            cum_p = sorted_p.cumsum(dim=-1)
            k50_w = ((cum_p < 0.5).sum(-1) + 1).float()  # (Hq, gc)
            ratio_w = k50_w / valid_len_f                # ratio of keys
            metrics["cov50_weight_min"][g0:g1] = torch.minimum(metrics["cov50_weight_min"][g0:g1], ratio_w.min(0).values.float())
            metrics["cov50_weight_max"][g0:g1] = torch.maximum(metrics["cov50_weight_max"][g0:g1], ratio_w.max(0).values.float())
            metrics["cov50_weight_sum"][g0:g1] += ratio_w.sum(0).float()

            # score version: shift raw scores to non-negative on valid positions, normalize, coverage
            scores_for_norm = torch.where(valid_mask_b, scores,
                                          torch.full_like(scores, float("inf")))
            score_min = scores_for_norm.min(dim=-1, keepdim=True).values  # (Hq, gc, 1) finite
            shifted = (scores - score_min).masked_fill(~valid_mask_b, 0.0)
            mass_s = shifted / shifted.sum(-1, keepdim=True).clamp_min(1e-12)
            sorted_s, _ = mass_s.sort(dim=-1, descending=True)
            cum_s = sorted_s.cumsum(dim=-1)
            k50_s = ((cum_s < 0.5).sum(-1) + 1).float()
            ratio_s = k50_s / valid_len_f
            metrics["cov50_score_min"][g0:g1] = torch.minimum(metrics["cov50_score_min"][g0:g1], ratio_s.min(0).values.float())
            metrics["cov50_score_max"][g0:g1] = torch.maximum(metrics["cov50_score_max"][g0:g1], ratio_s.max(0).values.float())
            metrics["cov50_score_sum"][g0:g1] += ratio_s.sum(0).float()
            del sorted_p, cum_p, sorted_s, cum_s, mass_s, shifted, scores_for_norm, k50_w, k50_s, ratio_w, ratio_s

            unif_thr = (1.0 / T_arr[g0:g1].to(device)).view(1, -1, 1)
            above = (probs > unif_thr).float()
            # divide by valid count (= prompt_len + g)
            above = above.sum(-1) / valid_len.squeeze(-1).float()  # (Hq, gc)
            metrics["above_max"][g0:g1] = torch.maximum(metrics["above_max"][g0:g1], above.max(0).values)
            metrics["above_sum"][g0:g1] += above.sum(0)

            del scores, probs
        del K_exp
        torch.cuda.empty_cache() if device == "cuda" else None
        print(f"  layer {li+1}/{len(bundles)} done", flush=True)

    # bring metrics to CPU once for the final Python row build
    metrics = {k: v.cpu() for k, v in metrics.items()}

    # is_special: detect EOS / chat-end tokens by id.
    eos_ids = set()
    if helper.tokenizer is not None:
        if helper.tokenizer.eos_token_id is not None:
            eos_ids.add(helper.tokenizer.eos_token_id)
        # Qwen-specific
        for tok_str in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = helper.tokenizer.convert_tokens_to_ids(tok_str)
                if isinstance(tid, int) and tid >= 0:
                    eos_ids.add(tid)
            except Exception:
                pass

    rows = []
    for t in range(G):
        T_t = int(prompt_len + t)
        tok_id = helper.generated_tokens[t]
        tok_str = helper.get_token_string(t)
        is_special = int(tok_id in eos_ids or "<|" in tok_str)
        r = {"T": T_t,
             "token_index": t,
             "is_special": is_special,
             "token_str": tok_str.replace("\n", "\\n").replace(",", " ")}
        for k in topks:
            r[f"topk_mass_{k}_min"] = float(metrics[f"topk_mass_{k}_min"][t])
            r[f"topk_mass_{k}_max"] = float(metrics[f"topk_mass_{k}_max"][t])
            r[f"topk_mass_{k}_mean"] = float(metrics[f"topk_mass_{k}_sum"][t]) / H_total
        r["eff_size_max"] = float(metrics["eff_size_max"][t])
        r["eff_size_mean"] = float(metrics["eff_size_sum"][t]) / H_total
        r["entropy_max"] = float(metrics["entropy_max"][t])
        r["entropy_mean"] = float(metrics["entropy_sum"][t]) / H_total
        r["frac_above_unif_max"] = float(metrics["above_max"][t])
        r["frac_above_unif_mean"] = float(metrics["above_sum"][t]) / H_total
        for tag in ("weight", "score"):
            r[f"cov50_{tag}_min"] = float(metrics[f"cov50_{tag}_min"][t])
            r[f"cov50_{tag}_max"] = float(metrics[f"cov50_{tag}_max"][t])
            r[f"cov50_{tag}_mean"] = float(metrics[f"cov50_{tag}_sum"][t]) / H_total
        rows.append(r)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default=str(ROOT / "snapshots" / "snap_qwen2k.pt"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "tail_metrics.csv"))
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    helper = ObserveAttentionHelper.from_file(args.snap)
    num_layers = len(helper.queries)
    if args.layers is None:
        start = (num_layers * 3) // 4
        layers = list(range(start, num_layers))
    else:
        layers = args.layers
    print(f"layers used: {layers}  device: {args.device}")

    rows = collect_fast(helper, layers, args.device)
    fields = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}  ({len(rows)} rows)")
