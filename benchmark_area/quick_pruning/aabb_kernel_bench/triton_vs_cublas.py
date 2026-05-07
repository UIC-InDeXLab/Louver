#!/usr/bin/env python3
"""Compare Triton AABB kernel against cuBLAS bmm (best dot product baseline)."""

import time
import torch
import triton
import triton.language as tl

H, D = 8, 128


@triton.jit
def _aabb_fused(Q, Lo, Hi, Out, Th, K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr):
    h = tl.program_id(0)
    k0 = tl.program_id(1) * BK
    ks = k0 + tl.arange(0, BK)
    mask = ks < K
    th = tl.load(Th + h)
    acc = tl.zeros([BK], dtype=tl.float32)
    qb = h * D
    for d in range(D):
        qv = tl.load(Q + qb + d)
        lv = tl.load(Lo + h * K * D + ks * D + d, mask=mask, other=0.0)
        hv = tl.load(Hi + h * K * D + ks * D + d, mask=mask, other=0.0)
        acc += tl.maximum(qv * lv, qv * hv)
    tl.store(Out + h * K + ks, (acc > th).to(tl.int8), mask=mask)


@triton.jit
def _dot_fused(Q, Keys, Out, Th, N: tl.constexpr, D: tl.constexpr, BN: tl.constexpr):
    h = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    ns = n0 + tl.arange(0, BN)
    mask = ns < N
    th = tl.load(Th + h)
    acc = tl.zeros([BN], dtype=tl.float32)
    qb = h * D
    for d in range(D):
        qv = tl.load(Q + qb + d)
        kv = tl.load(Keys + h * N * D + ns * D + d, mask=mask, other=0.0)
        acc += qv * kv
    tl.store(Out + h * N + ns, (acc > th).to(tl.int8), mask=mask)


@triton.jit
def _aabb_coalesced(
    Q, Lo, Hi, Out, Th,
    K: tl.constexpr, D: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
):
    """AABB with coalesced memory access pattern using blocked D."""
    h = tl.program_id(0)
    k0 = tl.program_id(1) * BK
    ks = k0 + tl.arange(0, BK)
    k_mask = ks < K
    th = tl.load(Th + h)

    acc = tl.zeros([BK], dtype=tl.float32)
    qb = h * D

    # Process D in blocks
    for d_start in range(0, D, BD):
        d_offs = d_start + tl.arange(0, BD)
        d_mask = d_offs < D

        # Load query block
        q_block = tl.load(Q + qb + d_offs, mask=d_mask, other=0.0)  # (BD,)

        # For each dim in block, accumulate
        for di in range(BD):
            if d_start + di < D:
                qv = tl.load(Q + qb + d_start + di)
                lv = tl.load(Lo + h * K * D + ks * D + d_start + di, mask=k_mask, other=0.0)
                hv = tl.load(Hi + h * K * D + ks * D + d_start + di, mask=k_mask, other=0.0)
                acc += tl.maximum(qv * lv, qv * hv)

    tl.store(Out + h * K + ks, (acc > th).to(tl.int8), mask=k_mask)


def bench(fn, n=500):
    for _ in range(100):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2] * 1e6


def main():
    print("=" * 100)
    print("AABB Gate Cost: Triton kernels vs cuBLAS bmm vs PyTorch ops")
    print(f"H={H}, D={D}")
    print("=" * 100)

    for K in [256, 512, 1024]:
        print(f"\n{'─'*100}")
        print(f"K={K}")
        print(f"{'─'*100}")

        q = torch.randn(H, D, device="cuda").contiguous()
        data = torch.randn(H, K, D, device="cuda").contiguous()
        lo = (data - 0.5).contiguous()
        hi = (data + 0.5).contiguous()
        th = torch.randn(H, device="cuda").contiguous()
        data_t = data.transpose(-2, -1).contiguous()
        q_mm = q.unsqueeze(1).contiguous()

        # ── Dot product baselines ──

        # cuBLAS bmm
        t_bmm = bench(lambda: torch.bmm(q_mm, data_t).squeeze(1))

        # Triton dot
        out_t = torch.empty(H, K, dtype=torch.int8, device="cuda")
        best_tdot = float("inf")
        best_tdot_bk = 0
        for BK in [32, 64, 128]:
            def run_tdot(bk=BK):
                _dot_fused[(H, triton.cdiv(K, bk))](q, data, out_t, th, N=K, D=D, BN=bk)
            t = bench(run_tdot)
            if t < best_tdot:
                best_tdot = t
                best_tdot_bk = BK

        # PyTorch manual
        t_manual = bench(lambda: (q.unsqueeze(1) * data).sum(-1))

        # einsum
        t_einsum = bench(lambda: torch.einsum("hd,hkd->hk", q, data))

        print(f"  DOT PRODUCT baselines:")
        print(f"    cuBLAS bmm:        {t_bmm:7.1f} us")
        print(f"    Triton dot (BK={best_tdot_bk}): {best_tdot:7.1f} us")
        print(f"    torch manual:      {t_manual:7.1f} us")
        print(f"    torch einsum:      {t_einsum:7.1f} us")

        # ── AABB variants ──

        # PyTorch maximum
        t_aabb_max = bench(lambda: torch.maximum(q.unsqueeze(1) * lo, q.unsqueeze(1) * hi).sum(-1))

        # Triton AABB
        best_taabb = float("inf")
        best_taabb_bk = 0
        for BK in [32, 64, 128]:
            def run_taabb(bk=BK):
                _aabb_fused[(H, triton.cdiv(K, bk))](q, lo, hi, out_t, th, K=K, D=D, BK=bk)
            t = bench(run_taabb)
            if t < best_taabb:
                best_taabb = t
                best_taabb_bk = BK

        # PyTorch midpoint: einsum(q,mid) + einsum(|q|,half)
        mid = ((lo + hi) / 2).contiguous()
        half = ((hi - lo).abs() / 2).contiguous()
        q_abs = q.abs().contiguous()
        mid_t = mid.transpose(-2, -1).contiguous()
        half_t = half.transpose(-2, -1).contiguous()
        q_abs_mm = q_abs.unsqueeze(1).contiguous()

        t_mid_einsum = bench(
            lambda: torch.einsum("hd,hkd->hk", q, mid) + torch.einsum("hd,hkd->hk", q_abs, half)
        )
        t_mid_bmm = bench(
            lambda: torch.bmm(q_mm, mid_t).squeeze(1) + torch.bmm(q_abs_mm, half_t).squeeze(1)
        )

        print(f"\n  AABB variants:")
        print(f"    torch.maximum:     {t_aabb_max:7.1f} us")
        print(f"    Triton (BK={best_taabb_bk}):    {best_taabb:7.1f} us")
        print(f"    mid einsum:        {t_mid_einsum:7.1f} us")
        print(f"    mid bmm:           {t_mid_bmm:7.1f} us")

        # ── g values relative to each baseline ──

        best_dot = min(t_bmm, best_tdot, t_manual, t_einsum)
        best_dot_name = ["bmm", "triton", "manual", "einsum"][
            [t_bmm, best_tdot, t_manual, t_einsum].index(best_dot)
        ]
        best_aabb = min(t_aabb_max, best_taabb, t_mid_einsum, t_mid_bmm)
        best_aabb_name = ["torch.max", "triton", "mid_einsum", "mid_bmm"][
            [t_aabb_max, best_taabb, t_mid_einsum, t_mid_bmm].index(best_aabb)
        ]

        print(f"\n  GATE COST (g = aabb_time / dot_time):")
        print(f"    vs cuBLAS bmm:     g = {best_aabb/t_bmm:.3f}  (best aabb={best_aabb_name})")
        print(f"    vs Triton dot:     g = {best_aabb/best_tdot:.3f}")
        print(f"    vs best dot ({best_dot_name}):  g = {best_aabb/best_dot:.3f}")

        # Fair comparison: same implementation family
        g_torch = t_aabb_max / t_manual  # both use elementwise PyTorch
        g_triton = best_taabb / best_tdot  # both Triton
        g_cublas = t_mid_bmm / t_bmm  # both cuBLAS-based

        print(f"\n  FAIR (same-family) comparisons:")
        print(f"    PyTorch elementwise: g = {g_torch:.3f}  (aabb_max vs manual_dot)")
        print(f"    Triton kernels:      g = {g_triton:.3f}  (triton_aabb vs triton_dot)")
        print(f"    cuBLAS (2x bmm):     g = {g_cublas:.3f}  (mid_bmm vs bmm)")

        # Speedup table
        print(f"\n  SPEEDUP TABLE (using Triton g={g_triton:.2f}):")
        print(f"    {'bf':>4s}  {'scanned':>8s}  {'ratio':>8s}  {'speedup':>8s}")
        for bf in [2, 3, 4]:
            for s in [0.15, 0.20, 0.30]:
                ratio = g_triton / bf + s
                sp = f"{1/ratio:.2f}x" if ratio < 1 else "—"
                mark = " *" if ratio < 1 else ""
                print(f"    {bf:>4d}  {s:>8.2f}  {ratio:>8.3f}  {sp:>8s}{mark}")


if __name__ == "__main__":
    main()
