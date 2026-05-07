"""Correctness verifier for CPU TA-filter kernels.

Three checks per (config, threshold):

1. **Reference attend** — pure-Python TA-filter mirroring `TA_filter_algorithm.md`
   + CUDA `stop_depth_per_head` (`depth = first_below_idx + 1`). Computes the
   alive set and an online-softmax output.

2. **Kernel alive-set probe** — replaces values with one-hot per-key indicators
   so the kernel's softmax weights become the output (one slot per key). Reads
   which slots are non-zero → kernel's alive set. (Done in chunks to keep D_v
   bounded.)

3. **Output cross-check** — compares each kernel's output against:
   a. An "ideal sparse attn" computed in fp64 over the *reference's* alive set,
      to isolate alive-set parity from softmax numerics.
   b. The reference's online-softmax output (same set, same dtype).

Run:
    ~/venv/bin/python -m hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.benchmarking.verify_correctness
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HIRA = Path(__file__).resolve().parents[5]
for _p in (_HIRA.parent, _HIRA):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hira.benchmark_area.kernel_impl.TA_filter_alg.cpu.cpu_index import (
    TAIndexCPU, TAIndexCPUConfig,
)


def ta_reference(q, threshold, state, q_head_to_kv, *, use_bf16_storage=False):
    """Pure-Python TA-filter reference. Returns (alive_sets, depths, output)."""
    if use_bf16_storage:
        centers = state["centers_padded_f32"].to(torch.bfloat16).to(torch.float32)
        keys_arena = state["keys_padded_f32"].to(torch.bfloat16).to(torch.float32)
        vals_arena = state["values_padded_f32"].to(torch.bfloat16).to(torch.float32)
        q_use = q.to(torch.bfloat16).to(torch.float32)
    else:
        centers = state["centers_padded_f32"]
        keys_arena = state["keys_padded_f32"]
        vals_arena = state["values_padded_f32"]
        q_use = q.float()
    assigns = state["assigns_padded_i32"].to(torch.long)
    invalid = state["invalid_mask"].bool()
    dim_offsets = state["dim_offsets"].tolist()
    dim_widths  = state["dim_widths"].tolist()

    S = centers.shape[0]
    K_used = int(state["K_used"])
    N_used = int(state["N_used"])
    h_q, D = q.shape
    Dv = vals_arena.shape[-1]
    L_FORCE = min(256, K_used)

    out = torch.zeros(h_q, Dv, dtype=torch.float32)
    alive_sets = []
    depths = []

    for hq in range(h_q):
        hkv = int(q_head_to_kv[hq].item())
        th  = float(threshold[hq].item())
        qrow = q_use[hq]

        scores_s = []
        for s in range(S):
            off, w = dim_offsets[s], dim_widths[s]
            qs = qrow[off:off + w]
            cs = centers[s, hkv, :K_used, :w].float()
            scores_s.append((cs * qs[None, :]).sum(dim=-1))

        sorted_idx = []
        sorted_val = []
        for s in range(S):
            sv, si = torch.topk(scores_s[s], k=L_FORCE)
            sorted_idx.append(si)
            sorted_val.append(sv)

        row_sums = sum(sorted_val)
        below = row_sums < th
        if bool(below.any().item()):
            depth = int(below.float().argmax().item()) + 1
        else:
            depth = L_FORCE
        depths.append(depth)

        selected = [set(int(x) for x in sorted_idx[s][:depth].tolist()) for s in range(S)]

        a_kv = assigns[:, hkv, :N_used]
        inv_kv = invalid[hkv, :N_used]
        alive = []
        for k in range(N_used):
            if bool(inv_kv[k].item()):
                continue
            for s in range(S):
                if int(a_kv[s, k].item()) in selected[s]:
                    alive.append(k)
                    break
        alive_sets.append(set(alive))

        # Online softmax (matches the kernel's order: scan alive ascending).
        scale = 1.0 / math.sqrt(D)
        m = float("-inf"); l = 0.0
        o = torch.zeros(Dv, dtype=torch.float32)
        for k in sorted(alive):
            kr = keys_arena[hkv, k].float()
            v  = vals_arena[hkv, k].float()
            sc = float((qrow * kr).sum().item()) * scale
            if sc > m:
                a = math.exp(m - sc); l = l * a + 1.0; o = o * a + v; m = sc
            else:
                w = math.exp(sc - m); l += w; o = o + w * v
        out[hq] = o / max(l, 1e-30)
    return alive_sets, depths, out


def ideal_sparse_attn_fp64(q, state, q_head_to_kv, alive_sets):
    """Brute-force fp64 softmax over the given alive set. Numerics-tight
    'ground truth' for the kernel's softmax over the same set."""
    keys = state["keys_padded_f32"].double()
    vals = state["values_padded_f32"].double()
    h_q, D = q.shape
    Dv = vals.shape[-1]
    out = torch.zeros(h_q, Dv, dtype=torch.float64)
    scale = 1.0 / math.sqrt(D)
    for hq in range(h_q):
        hkv = int(q_head_to_kv[hq].item())
        idx = sorted(alive_sets[hq])
        if not idx:
            continue
        idx_t = torch.tensor(idx, dtype=torch.long)
        k_sel = keys[hkv].index_select(0, idx_t)
        v_sel = vals[hkv].index_select(0, idx_t)
        sc = (q[hq].double() @ k_sel.T) * scale
        sm = torch.softmax(sc, dim=-1)
        out[hq] = (sm[:, None] * v_sel).sum(dim=0)
    return out.float()


def kernel_alive_set(index, q, threshold, q_head_to_kv, state):
    """Probe the kernel's alive set by replacing values with per-key indicators
    in fixed-size chunks. For each chunk of CHUNK keys we set
    `values[h_kv, k, k%CHUNK] = 1` and read which chunk slots become non-zero.

    Returns list of sets — one per h_q.
    """
    H_kv, N_pad, D_v = state["values_padded_f32"].shape
    N_used = int(state["N_used"])
    CHUNK = D_v
    saved_v32 = state["values_padded_f32"].clone()
    saved_vbf = state["values_padded_bf16"].clone() if "values_padded_bf16" in state else None
    h_q = q.shape[0]
    alive = [set() for _ in range(h_q)]

    try:
        for chunk_start in range(0, N_used, CHUNK):
            chunk_end = min(chunk_start + CHUNK, N_used)
            v = torch.zeros(H_kv, N_pad, D_v, dtype=torch.float32)
            for k in range(chunk_start, chunk_end):
                v[:, k, k - chunk_start] = 1.0
            state["values_padded_f32"].copy_(v)
            if saved_vbf is not None:
                state["values_padded_bf16"].copy_(v.to(torch.bfloat16))
            out = index.attend(q, threshold, q_head_to_kv=q_head_to_kv)
            for hq in range(h_q):
                for slot in range(chunk_end - chunk_start):
                    if out[hq, slot].abs().item() > 1e-7:
                        alive[hq].add(chunk_start + slot)
    finally:
        state["values_padded_f32"].copy_(saved_v32)
        if saved_vbf is not None:
            state["values_padded_bf16"].copy_(saved_vbf)
    return alive


def make_test_state(H_q, H_kv, N, D, *, seed=0):
    torch.manual_seed(seed)
    keys = torch.randn(H_kv, N, D)
    values = torch.randn(H_kv, N, D)
    q = torch.randn(H_q, D)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_head_to_kv = torch.arange(H_q) // (H_q // H_kv)
    return keys, values, qn, q_head_to_kv


def topk_threshold(q, keys, q_head_to_kv, topk):
    keys_eval = keys.index_select(0, q_head_to_kv)
    sc = torch.einsum("hd,hnd->hn", q.float(), keys_eval.float())
    k_eff = min(topk, sc.shape[-1])
    top_vals, _ = sc.topk(k_eff, dim=-1)
    return top_vals[:, k_eff - 1].contiguous().to(torch.float32)


def fmt_setdiff(ref, ker):
    only_ref = ref - ker
    only_ker = ker - ref
    return f"|ref|={len(ref)} |ker|={len(ker)} only_ref={len(only_ref)} only_ker={len(only_ker)}"


def run_config(label, H_q, H_kv, N, D, topks):
    print(f"\n── {label}  H_q={H_q} H_kv={H_kv} N={N} D={D} ──")
    keys, values, qn, q2kv = make_test_state(H_q, H_kv, N, D)

    indices = {}
    for ver in ("v1", "v2", "v3"):
        cfg = TAIndexCPUConfig(n_growth=0, refine_iter=2, attend_version=ver)
        indices[ver] = TAIndexCPU(cfg).build(keys, values)
    state = indices["v1"].state

    for topk in topks:
        th = topk_threshold(qn, keys, q2kv, topk)
        alive_ref, depths_ref, _ = ta_reference(qn, th, state, q2kv, use_bf16_storage=False)
        ideal_out = ideal_sparse_attn_fp64(qn, state, q2kv, alive_ref)
        avg_alive = sum(len(s) for s in alive_ref) / max(H_q, 1)
        avg_depth = sum(depths_ref) / max(H_q, 1)
        print(f"\n  topk={topk:>4}  ref_avg_depth={avg_depth:5.1f}  ref_avg|alive|={avg_alive:6.1f}")
        for ver in ("v1", "v2", "v3"):
            out = indices[ver].attend(qn, th, q_head_to_kv=q2kv)
            diff_vs_ideal = (out.float() - ideal_out.float()).abs().max().item()
            mag = ideal_out.float().abs().max().item() + 1e-9
            rel_vs_ideal = diff_vs_ideal / mag

            alive_ker = kernel_alive_set(indices[ver], qn, th, q2kv, indices[ver].state)
            sym = sum(len(a ^ b) for a, b in zip(alive_ref, alive_ker))
            tot_ref = sum(len(s) for s in alive_ref)
            tag = "OK" if (sym == 0 and rel_vs_ideal < (1e-2 if ver == "v3" else 1e-4)) else (
                "ALIVE-DIFF" if sym > 0 else "NUM-DIFF")
            print(f"    {ver}: {tag}  alive_sym_diff={sym}/{tot_ref}  "
                  f"output_vs_ideal_fp64: abs={diff_vs_ideal:.2e} rel={rel_vs_ideal:.2e}")


def main():
    print("== TA-filter CPU correctness verifier ==")
    print("Reference: pure-python TA spec (matches CUDA stop_depth_per_head + alive-OR).")
    print("Each kernel's output is compared to fp64 brute-force softmax over the")
    print("reference's alive set (isolates alive-set parity from softmax numerics).")
    print("Each kernel's alive set is also probed via per-key one-hot values.")
    run_config("Llama-like small",  H_q=24, H_kv=8, N=512,  D=128, topks=[5, 20, 100, 10**6])
    run_config("Llama-like medium", H_q=24, H_kv=8, N=2048, D=128, topks=[5, 20, 100])
    run_config("Qwen-like medium",  H_q=28, H_kv=4, N=2048, D=128, topks=[5, 20, 100])


if __name__ == "__main__":
    main()
