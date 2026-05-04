// Shared helpers for the CPU TA-filter pipeline kernels.
// Fp32 storage, AVX-512 specialised on D=128 hot loops.
#pragma once

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
#define HIRA_HAS_AVX512 1
#else
#define HIRA_HAS_AVX512 0
#endif

namespace ta_cpu {

inline std::vector<int64_t> resolve_q_head_to_kv(
    torch::Tensor q_head_to_kv, int64_t h_q, int64_t h_kv) {
    std::vector<int64_t> q2kv(static_cast<size_t>(h_q), 0);
    if (q_head_to_kv.defined() && q_head_to_kv.numel() > 0) {
        auto m = q_head_to_kv.contiguous().to(torch::kInt64);
        TORCH_CHECK(m.dim() == 1 && m.size(0) == h_q,
                    "q_head_to_kv shape mismatch");
        const int64_t* p = m.data_ptr<int64_t>();
        for (int64_t h = 0; h < h_q; ++h) q2kv[(size_t)h] = p[h];
    } else if (h_q == h_kv) {
        for (int64_t h = 0; h < h_q; ++h) q2kv[(size_t)h] = h;
    } else {
        TORCH_CHECK(h_q % h_kv == 0,
                    "H_q must be divisible by H_kv when no q_head_to_kv given");
        int64_t g = h_q / h_kv;
        for (int64_t h = 0; h < h_q; ++h) q2kv[(size_t)h] = h / g;
    }
    return q2kv;
}

#if HIRA_HAS_AVX512

inline float reduce_zmm(__m512 v) { return _mm512_reduce_add_ps(v); }

inline float dot_d128(const float* __restrict a, const float* __restrict b) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(),
           a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    for (int i = 0; i < 128; i += 64) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i +  0), _mm512_loadu_ps(b + i +  0), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), a2);
        a3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), a3);
    }
    return reduce_zmm(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
}

inline float dot_generic(const float* __restrict a, const float* __restrict b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float r = reduce_zmm(acc);
    for (; i < n; ++i) r += a[i] * b[i];
    return r;
}

inline void axpy_d128(float* __restrict dst, const float* __restrict src, float w) {
    const __m512 vw = _mm512_set1_ps(w);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_fmadd_ps(_mm512_loadu_ps(src + i +  0), vw, _mm512_loadu_ps(dst + i +  0)));
        _mm512_storeu_ps(dst + i + 16, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), vw, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), vw, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48, _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), vw, _mm512_loadu_ps(dst + i + 48)));
    }
}

inline void scale_d128(float* __restrict dst, float a) {
    const __m512 va = _mm512_set1_ps(a);
    for (int i = 0; i < 128; i += 64) {
        _mm512_storeu_ps(dst + i +  0, _mm512_mul_ps(_mm512_loadu_ps(dst + i +  0), va));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), va));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), va));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), va));
    }
}

#if defined(__AVX512BF16__)
// AVX-512 BF16 dot product. Accumulates 32 bf16 multiplies into 16 fp32 lanes
// per instruction (vdpbf16ps). a, b: bf16 arrays.
inline float dot_bf16_d128(const uint16_t* __restrict a, const uint16_t* __restrict b) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512bh va0 = (__m512bh)_mm512_loadu_si512((const __m512i*)(a +  0));
    __m512bh vb0 = (__m512bh)_mm512_loadu_si512((const __m512i*)(b +  0));
    __m512bh va1 = (__m512bh)_mm512_loadu_si512((const __m512i*)(a + 32));
    __m512bh vb1 = (__m512bh)_mm512_loadu_si512((const __m512i*)(b + 32));
    __m512bh va2 = (__m512bh)_mm512_loadu_si512((const __m512i*)(a + 64));
    __m512bh vb2 = (__m512bh)_mm512_loadu_si512((const __m512i*)(b + 64));
    __m512bh va3 = (__m512bh)_mm512_loadu_si512((const __m512i*)(a + 96));
    __m512bh vb3 = (__m512bh)_mm512_loadu_si512((const __m512i*)(b + 96));
    acc0 = _mm512_dpbf16_ps(acc0, va0, vb0);
    acc1 = _mm512_dpbf16_ps(acc1, va1, vb1);
    acc2 = _mm512_dpbf16_ps(acc2, va2, vb2);
    acc3 = _mm512_dpbf16_ps(acc3, va3, vb3);
    return reduce_zmm(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

// w=32 bf16 dot (centroid). One vdpbf16ps over 32 bf16 pairs.
inline float dot_bf16_w32(const uint16_t* __restrict a, const uint16_t* __restrict b) {
    __m512 acc = _mm512_setzero_ps();
    __m512bh va = (__m512bh)_mm512_loadu_si512((const __m512i*)a);
    __m512bh vb = (__m512bh)_mm512_loadu_si512((const __m512i*)b);
    acc = _mm512_dpbf16_ps(acc, va, vb);
    return reduce_zmm(acc);
}
#endif

#else
inline float dot_d128(const float* a, const float* b) {
    float s = 0.f;
    for (int i = 0; i < 128; ++i) s += a[i] * b[i];
    return s;
}
inline float dot_generic(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
inline void axpy_d128(float* dst, const float* src, float w) {
    for (int i = 0; i < 128; ++i) dst[i] += w * src[i];
}
inline void scale_d128(float* dst, float a) {
    for (int i = 0; i < 128; ++i) dst[i] *= a;
}
#endif

}  // namespace ta_cpu
