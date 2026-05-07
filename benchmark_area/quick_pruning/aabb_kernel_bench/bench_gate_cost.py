#!/usr/bin/env python3
"""
Benchmark actual gate cost (g) of AABB gating vs dot product.

Measures wall-clock time of different gating implementations relative to
a simple dot product (which defines g=1.0). All measurements at D=128.

Implementations tested:
1. dot_product:       q @ keys^T  (baseline, g=1.0 by definition)
2. ball_gate:         q·c + r     (g_theoretical=1.0)
3. aabb_naive:        max(q*lo, q*hi).sum()  (g_theoretical=2.0)
4. aabb_midpoint:     q·mid + |q|·half       (g_theoretical=2.0, same math)
5. aabb_fused_triton: Triton kernel fusing the AABB gate
6. aabb_int8:         Quantized lo/hi to int8, dequant in kernel
7. ball_then_aabb:    Ball gate first, AABB only for survivors

Output: measured g values (time relative to dot product) for each method.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─── Torch implementations ───────────────────────────────────────────


def dot_product_gate(q: torch.Tensor, keys: torch.Tensor, threshold: torch.Tensor):
    """Baseline: full dot product q @ keys^T, then compare to threshold.
    q: (H, D), keys: (H, N, D) -> scores: (H, N) -> mask: (H, N) bool
    """
    scores = torch.einsum("hd,hnd->hn", q, keys)
    return scores > threshold.unsqueeze(-1)


def ball_gate(q: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, threshold: torch.Tensor):
    """Ball gate: UB = q·c + r. q: (H,D), centers: (H,K,D), radii: (H,K)"""
    scores = torch.einsum("hd,hkd->hk", q, centers)
    ub = scores + radii
    return ub > threshold.unsqueeze(-1)


def aabb_naive_gate(q: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, threshold: torch.Tensor):
    """AABB gate: UB = sum_d max(q_d*lo_d, q_d*hi_d).
    q: (H,D), lo/hi: (H,K,D)
    """
    q_exp = q.unsqueeze(1)  # (H,1,D)
    ub = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)  # (H,K)
    return ub > threshold.unsqueeze(-1)


def aabb_midpoint_gate(q: torch.Tensor, mid: torch.Tensor, half: torch.Tensor, threshold: torch.Tensor):
    """AABB via midpoint form: UB = q·mid + |q|·half.
    q: (H,D), mid: (H,K,D), half: (H,K,D)
    """
    q_mid = torch.einsum("hd,hkd->hk", q, mid)
    q_abs = q.abs()
    q_half = torch.einsum("hd,hkd->hk", q_abs, half)
    ub = q_mid + q_half
    return ub > threshold.unsqueeze(-1)


def ball_then_aabb_gate(
    q: torch.Tensor,
    centers: torch.Tensor, radii: torch.Tensor,
    lo: torch.Tensor, hi: torch.Tensor,
    threshold: torch.Tensor,
):
    """Two-stage: ball gate first (cheap), then AABB only on survivors."""
    # Stage 1: ball
    c_scores = torch.einsum("hd,hkd->hk", q, centers)
    ball_ub = c_scores + radii
    ball_pass = ball_ub > threshold.unsqueeze(-1)

    # Stage 2: AABB only on ball survivors
    # For benchmarking, we compute full AABB but mask — in a real kernel
    # we'd skip non-survivors entirely
    q_exp = q.unsqueeze(1)
    aabb_ub = torch.maximum(q_exp * lo, q_exp * hi).sum(dim=-1)

    # Final: pass only if both ball and aabb pass
    # (ball is looser, so ball_pass is superset — aabb refines)
    return ball_pass & (aabb_ub > threshold.unsqueeze(-1))


def aabb_int8_gate(
    q: torch.Tensor,
    lo_q: torch.Tensor, hi_q: torch.Tensor,  # int8 quantized
    lo_scale: torch.Tensor, hi_scale: torch.Tensor,  # per-cluster scales (H,K)
    lo_zp: torch.Tensor, hi_zp: torch.Tensor,  # per-cluster zero points (H,K)
    threshold: torch.Tensor,
):
    """AABB with int8-quantized bounds. Dequantize on-the-fly."""
    # Dequantize: val = (int8_val - zp) * scale
    lo_f = (lo_q.float() - lo_zp.unsqueeze(-1)) * lo_scale.unsqueeze(-1)
    hi_f = (hi_q.float() - hi_zp.unsqueeze(-1)) * hi_scale.unsqueeze(-1)
    q_exp = q.unsqueeze(1)
    ub = torch.maximum(q_exp * lo_f, q_exp * hi_f).sum(dim=-1)
    return ub > threshold.unsqueeze(-1)


# ─── Triton kernel implementations ───────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _aabb_gate_kernel(
        Q_ptr, Lo_ptr, Hi_ptr, Out_ptr, Th_ptr,
        K: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused AABB gate kernel.
        Q: (H, D), Lo/Hi: (H, K, D), Th: (H,), Out: (H, K) bool
        Each program handles one head and a block of K clusters.
        """
        h = tl.program_id(0)
        k_start = tl.program_id(1) * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load query for this head
        q_base = h * D
        d_offsets = tl.arange(0, D)

        # Load threshold
        th = tl.load(Th_ptr + h)

        # Accumulate UB for each cluster in block
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        for d in range(D):
            q_val = tl.load(Q_ptr + q_base + d)
            lo_val = tl.load(
                Lo_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            hi_val = tl.load(
                Hi_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            prod_lo = q_val * lo_val
            prod_hi = q_val * hi_val
            acc += tl.maximum(prod_lo, prod_hi)

        # Write result
        out_offsets = h * K + k_offsets
        tl.store(Out_ptr + out_offsets, (acc > th).to(tl.int8), mask=k_mask)

    def aabb_triton_gate(q, lo, hi, threshold):
        """Launch the Triton AABB gate kernel."""
        H, K, D = lo.shape
        out = torch.empty(H, K, dtype=torch.int8, device=q.device)
        BLOCK_K = min(64, triton.next_power_of_2(K))
        grid = (H, triton.cdiv(K, BLOCK_K))
        _aabb_gate_kernel[grid](
            q, lo, hi, out, threshold,
            K=K, D=D, BLOCK_K=BLOCK_K,
        )
        return out.bool()

    @triton.jit
    def _aabb_midpoint_kernel(
        Q_ptr, Mid_ptr, Half_ptr, Out_ptr, Th_ptr,
        K: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused AABB midpoint kernel: UB = q·mid + |q|·half."""
        h = tl.program_id(0)
        k_start = tl.program_id(1) * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        th = tl.load(Th_ptr + h)
        q_base = h * D

        acc_mid = tl.zeros([BLOCK_K], dtype=tl.float32)
        acc_half = tl.zeros([BLOCK_K], dtype=tl.float32)

        for d in range(D):
            q_val = tl.load(Q_ptr + q_base + d)
            q_abs = tl.abs(q_val)

            mid_val = tl.load(
                Mid_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            half_val = tl.load(
                Half_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            acc_mid += q_val * mid_val
            acc_half += q_abs * half_val

        ub = acc_mid + acc_half
        out_offsets = h * K + k_offsets
        tl.store(Out_ptr + out_offsets, (ub > th).to(tl.int8), mask=k_mask)

    def aabb_midpoint_triton_gate(q, mid, half, threshold):
        H, K, D = mid.shape
        out = torch.empty(H, K, dtype=torch.int8, device=q.device)
        BLOCK_K = min(64, triton.next_power_of_2(K))
        grid = (H, triton.cdiv(K, BLOCK_K))
        _aabb_midpoint_kernel[grid](
            q, mid, half, out, threshold,
            K=K, D=D, BLOCK_K=BLOCK_K,
        )
        return out.bool()

    @triton.jit
    def _dot_product_kernel(
        Q_ptr, Keys_ptr, Out_ptr, Th_ptr,
        N: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Baseline dot product kernel for fair comparison."""
        h = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        th = tl.load(Th_ptr + h)
        q_base = h * D

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(D):
            q_val = tl.load(Q_ptr + q_base + d)
            k_val = tl.load(
                Keys_ptr + h * N * D + n_offsets * D + d,
                mask=n_mask, other=0.0
            )
            acc += q_val * k_val

        out_offsets = h * N + n_offsets
        tl.store(Out_ptr + out_offsets, (acc > th).to(tl.int8), mask=n_mask)

    def dot_triton_gate(q, keys, threshold):
        H, N, D = keys.shape
        out = torch.empty(H, N, dtype=torch.int8, device=q.device)
        BLOCK_N = min(64, triton.next_power_of_2(N))
        grid = (H, triton.cdiv(N, BLOCK_N))
        _dot_product_kernel[grid](
            q, keys, out, threshold,
            N=N, D=D, BLOCK_N=BLOCK_N,
        )
        return out.bool()

    @triton.jit
    def _ball_then_aabb_kernel(
        Q_ptr, Centers_ptr, Radii_ptr, Lo_ptr, Hi_ptr, Out_ptr, Th_ptr,
        K: tl.constexpr, D: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused ball-then-AABB: compute ball UB, skip AABB if ball prunes."""
        h = tl.program_id(0)
        k_start = tl.program_id(1) * BLOCK_K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        th = tl.load(Th_ptr + h)
        q_base = h * D

        # Load radii
        radii = tl.load(Radii_ptr + h * K + k_offsets, mask=k_mask, other=0.0)

        # Stage 1: Ball gate (q·c + r)
        ball_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for d in range(D):
            q_val = tl.load(Q_ptr + q_base + d)
            c_val = tl.load(
                Centers_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            ball_acc += q_val * c_val
        ball_ub = ball_acc + radii
        ball_pass = ball_ub > th

        # Stage 2: AABB only for ball survivors
        # In Triton we can't truly branch per-element, but we can skip
        # the entire block if no element passes ball
        any_pass = tl.sum(ball_pass.to(tl.int32), axis=0)

        # Always compute AABB (Triton SIMT — can't skip individual lanes)
        # but only store result where ball passed
        aabb_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for d in range(D):
            q_val = tl.load(Q_ptr + q_base + d)
            lo_val = tl.load(
                Lo_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            hi_val = tl.load(
                Hi_ptr + h * K * D + k_offsets * D + d,
                mask=k_mask, other=0.0
            )
            aabb_acc += tl.maximum(q_val * lo_val, q_val * hi_val)

        final = ball_pass & (aabb_acc > th)
        out_offsets = h * K + k_offsets
        tl.store(Out_ptr + out_offsets, final.to(tl.int8), mask=k_mask)

    def ball_then_aabb_triton_gate(q, centers, radii, lo, hi, threshold):
        H, K, D = lo.shape
        out = torch.empty(H, K, dtype=torch.int8, device=q.device)
        BLOCK_K = min(64, triton.next_power_of_2(K))
        grid = (H, triton.cdiv(K, BLOCK_K))
        _ball_then_aabb_kernel[grid](
            q, centers, radii, lo, hi, out, threshold,
            K=K, D=D, BLOCK_K=BLOCK_K,
        )
        return out.bool()


# ─── Quantization helpers ─────────────────────────────────────────────

def quantize_per_cluster(tensor: torch.Tensor):
    """Quantize (H, K, D) tensor to int8 with per-cluster scale/zp.
    Returns: int8_tensor, scale (H,K), zero_point (H,K)
    """
    H, K, D = tensor.shape
    tmin = tensor.amin(dim=-1)  # (H, K)
    tmax = tensor.amax(dim=-1)  # (H, K)
    scale = (tmax - tmin) / 255.0
    scale = scale.clamp_min(1e-12)
    zp = (-tmin / scale).round().clamp(0, 255)
    quant = ((tensor - tmin.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, 255).to(torch.int8)
    return quant, scale, zp


# ─── Benchmark harness ────────────────────────────────────────────────

def benchmark_fn(fn, warmup=50, repeat=200):
    """Benchmark a function, return median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times.sort()
    # Return median
    return times[len(times) // 2]


def run_benchmark(H=8, D=128, N=1024, bf=2):
    """Run all gate benchmarks and report g values."""
    K = N // bf
    device = "cuda"
    dtype = torch.float32

    print(f"{'='*80}")
    print(f"Gate Cost Benchmark: H={H}, D={D}, N={N}, bf={bf}, K={K}")
    print(f"{'='*80}\n")

    # Generate random data
    torch.manual_seed(42)
    q = torch.randn(H, D, device=device, dtype=dtype)
    keys = torch.randn(H, N, D, device=device, dtype=dtype)
    centers = torch.randn(H, K, D, device=device, dtype=dtype)
    radii = torch.rand(H, K, device=device, dtype=dtype) * 5
    lo = torch.randn(H, K, D, device=device, dtype=dtype) - 1
    hi = lo + torch.rand(H, K, D, device=device, dtype=dtype) * 2
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    threshold = torch.randn(H, device=device, dtype=dtype)

    # Quantized bounds
    lo_q, lo_s, lo_z = quantize_per_cluster(lo)
    hi_q, hi_s, hi_z = quantize_per_cluster(hi)

    # ── PyTorch implementations ──

    results = {}

    # 1. Dot product baseline (N points)
    t_dot_N = benchmark_fn(lambda: dot_product_gate(q, keys, threshold))
    results["dot_product (N pts)"] = {"time_us": t_dot_N, "n_items": N}

    # 2. Dot product on K clusters (to normalize g)
    keys_K = keys[:, :K, :]
    t_dot_K = benchmark_fn(lambda: dot_product_gate(q, keys_K, threshold))
    results["dot_product (K cls)"] = {"time_us": t_dot_K, "n_items": K}

    # 3. Ball gate
    t_ball = benchmark_fn(lambda: ball_gate(q, centers, radii, threshold))
    results["ball_gate"] = {"time_us": t_ball, "n_items": K}

    # 4. AABB naive
    t_aabb = benchmark_fn(lambda: aabb_naive_gate(q, lo, hi, threshold))
    results["aabb_naive"] = {"time_us": t_aabb, "n_items": K}

    # 5. AABB midpoint
    t_mid = benchmark_fn(lambda: aabb_midpoint_gate(q, mid, half, threshold))
    results["aabb_midpoint"] = {"time_us": t_mid, "n_items": K}

    # 6. Ball then AABB (PyTorch)
    t_ball_aabb = benchmark_fn(
        lambda: ball_then_aabb_gate(q, centers, radii, lo, hi, threshold)
    )
    results["ball_then_aabb (torch)"] = {"time_us": t_ball_aabb, "n_items": K}

    # 7. AABB int8
    t_int8 = benchmark_fn(
        lambda: aabb_int8_gate(q, lo_q, hi_q, lo_s, hi_s, lo_z, hi_z, threshold)
    )
    results["aabb_int8"] = {"time_us": t_int8, "n_items": K}

    # 8. AABB fp16 (bounds stored as fp16, compute in fp32)
    lo_f16 = lo.half()
    hi_f16 = hi.half()
    t_fp16 = benchmark_fn(
        lambda: aabb_naive_gate(q, lo_f16.float(), hi_f16.float(), threshold)
    )
    results["aabb_fp16_bounds"] = {"time_us": t_fp16, "n_items": K}

    # 9. AABB fp16 (bounds stored as fp16, compute directly in fp16)
    q_f16 = q.half()
    th_f16 = threshold.half()
    def aabb_fp16_direct():
        q_exp = q_f16.unsqueeze(1)
        ub = torch.maximum(q_exp * lo_f16, q_exp * hi_f16).sum(dim=-1)
        return ub > th_f16.unsqueeze(-1)
    t_fp16d = benchmark_fn(aabb_fp16_direct)
    results["aabb_fp16_direct"] = {"time_us": t_fp16d, "n_items": K}

    # ── Triton implementations ──
    if HAS_TRITON:
        # Triton dot product baseline
        t_tdot = benchmark_fn(lambda: dot_triton_gate(q, keys_K, threshold))
        results["triton_dot (K cls)"] = {"time_us": t_tdot, "n_items": K}

        # Triton AABB
        lo_c = lo.contiguous()
        hi_c = hi.contiguous()
        t_taabb = benchmark_fn(lambda: aabb_triton_gate(q, lo_c, hi_c, threshold))
        results["triton_aabb"] = {"time_us": t_taabb, "n_items": K}

        # Triton AABB midpoint
        mid_c = mid.contiguous()
        half_c = half.contiguous()
        t_tmid = benchmark_fn(lambda: aabb_midpoint_triton_gate(q, mid_c, half_c, threshold))
        results["triton_aabb_midpoint"] = {"time_us": t_tmid, "n_items": K}

        # Triton ball+AABB
        centers_c = centers.contiguous()
        radii_c = radii.contiguous()
        t_tba = benchmark_fn(
            lambda: ball_then_aabb_triton_gate(q, centers_c, radii_c, lo_c, hi_c, threshold)
        )
        results["triton_ball_then_aabb"] = {"time_us": t_tba, "n_items": K}

    # ── Report ──

    # Normalize: g = time_per_item / time_per_item_of_dot_product
    # Use dot_product(K) as reference since gates operate on K clusters
    ref_time = t_dot_K
    ref_per_item = ref_time / K

    print(f"{'Method':<30s} {'Time(us)':>10s} {'per-item(ns)':>14s} {'g (measured)':>14s} {'Note':>20s}")
    print("-" * 92)

    for name, info in results.items():
        t = info["time_us"]
        n = info["n_items"]
        per_item_ns = (t / n) * 1000
        g = (t / n) / (ref_per_item) if ref_per_item > 0 else float("inf")
        note = ""
        if "dot" in name and "N" in name:
            note = f"(bf*K items)"
        elif "dot" in name:
            note = "(reference g=1.0)"
        results[name]["g"] = g
        print(f"{name:<30s} {t:>10.1f} {per_item_ns:>14.1f} {g:>14.2f} {note:>20s}")

    # ── Speedup ratio calculation ──

    print(f"\n{'='*80}")
    print(f"Speedup Ratio Analysis (bf={bf})")
    print(f"{'='*80}")
    print(f"\nFormula: ratio = g/bf + scanned_fraction")
    print(f"Speedup when ratio < 1.0\n")

    gate_methods = {
        "ball_gate": results.get("ball_gate", {}).get("g", 1.0),
        "aabb_naive": results.get("aabb_naive", {}).get("g", 2.0),
        "aabb_midpoint": results.get("aabb_midpoint", {}).get("g", 2.0),
        "aabb_int8": results.get("aabb_int8", {}).get("g", 2.0),
        "aabb_fp16_bounds": results.get("aabb_fp16_bounds", {}).get("g", 2.0),
        "aabb_fp16_direct": results.get("aabb_fp16_direct", {}).get("g", 2.0),
    }
    if HAS_TRITON:
        gate_methods["triton_aabb"] = results.get("triton_aabb", {}).get("g", 2.0)
        gate_methods["triton_aabb_midpoint"] = results.get("triton_aabb_midpoint", {}).get("g", 2.0)
        gate_methods["triton_ball_then_aabb"] = results.get("triton_ball_then_aabb", {}).get("g", 2.0)

    scanned_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    print(f"{'Method':<30s}", end="")
    for s in scanned_values:
        print(f" {'s='+str(s):>8s}", end="")
    print()
    print("-" * (30 + 9 * len(scanned_values)))

    for name, g in gate_methods.items():
        print(f"{name:<30s}", end="")
        for s in scanned_values:
            ratio = g / bf + s
            if ratio < 1.0:
                print(f" {ratio:>7.3f}*", end="")
            else:
                print(f" {ratio:>8.3f}", end="")
        print(f"  (g={g:.2f})")

    # ── Critical scanned fraction ──
    print(f"\nMax scanned fraction for speedup (ratio < 1.0):")
    print(f"{'Method':<30s} {'g':>6s} {'g/bf':>8s} {'max_scanned':>12s}")
    print("-" * 60)
    for name, g in gate_methods.items():
        overhead = g / bf
        max_s = 1.0 - overhead
        marker = " ✓" if max_s > 0 else " ✗"
        print(f"{name:<30s} {g:>6.2f} {overhead:>8.3f} {max_s:>12.3f}{marker}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--bf", type=int, default=2)
    args = parser.parse_args()

    print("Testing multiple configurations...\n")

    for bf in [2, 3, 4]:
        results = run_benchmark(H=args.H, D=args.D, N=args.N, bf=bf)
        print("\n\n")
