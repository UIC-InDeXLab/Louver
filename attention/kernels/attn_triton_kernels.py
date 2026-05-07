"""
Fused HIRA Attention Triton Kernel
===================================

Single-pass fused kernel for HIRA decoding attention that combines:
  1. Online softmax over pre-computed index weights (sparse, zeros = pruned)
  2. On-the-fly queued QK dot-product + softmax
  3. Weighted value accumulation for both indexed and queued KV pairs

Avoids:
  - masked_fill scan to filter zeros
  - torch.cat concatenations of weights and values
  - repeat_kv memory copy (handles GQA via q_head_to_kv mapping)
  - Two-pass softmax (uses online / streaming softmax à la FlashAttention)

Shapes (decoding, batch=1, seq_len=1):
  index_weights : (H_q, N)       — from searcher, 0.0 for pruned/padded
  index_values  : (H_kv, N, D)   — from indexer
  query         : (H_q, D)       — original (un-normalized) query
  queued_keys   : (H_kv, Q, D)   — keys not yet indexed
  queued_values : (H_kv, Q, D)   — values not yet indexed
  q_head_to_kv  : (H_q,)  int64  — maps each query head to its KV head
  output        : (H_q, D)
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton JIT kernel
# ---------------------------------------------------------------------------

# Sentinel value used instead of -inf to avoid NaN from exp(-inf - (-inf)).
# Any real attention score is vastly larger than this.
_NEG_INF_SENTINEL = tl.constexpr(-1e30)


@triton.jit
def _fused_hira_attention_kernel(
    # ---- Index weights (H_q, N) — already scaled by searcher ----
    IW_ptr,
    IW_h_stride,
    IW_n_stride,
    # ---- Index values (H_kv, N, D) ----
    IV_ptr,
    IV_h_stride,
    IV_n_stride,
    IV_d_stride,
    # ---- Query (H_q, D) ----
    Q_ptr,
    Q_h_stride,
    Q_d_stride,
    # ---- Queued keys (H_kv, Q_LEN, D) ----
    QK_ptr,
    QK_h_stride,
    QK_n_stride,
    QK_d_stride,
    # ---- Queued values (H_kv, Q_LEN, D) ----
    QV_ptr,
    QV_h_stride,
    QV_n_stride,
    QV_d_stride,
    # ---- q_head → kv_head mapping (H_q,) ----
    H2KV_ptr,
    # ---- Output (H_q, D) ----
    OUT_ptr,
    OUT_h_stride,
    OUT_d_stride,
    # ---- Scalars ----
    scaling,  # float – attention scaling factor (1/sqrt(d_k))
    N,  # int – number of indexed keys  (runtime)
    Q_LEN,  # int – number of queued keys  (runtime)
    # ---- Compile-time constants ----
    D: tl.constexpr,  # head dimension (e.g. 128)
    BLOCK_N: tl.constexpr,  # tile size for indexed keys
    BLOCK_Q: tl.constexpr,  # tile size for queued keys
):
    """One program per query head.  Streams over all KV pairs with online softmax."""

    h_q = tl.program_id(0)

    # ── Resolve GQA mapping ──────────────────────────────────────────────
    kv_h = tl.load(H2KV_ptr + h_q).to(tl.int64)

    # ── Load query vector (D,) ───────────────────────────────────────────
    d_offs = tl.arange(0, D)
    q = tl.load(Q_ptr + h_q * Q_h_stride + d_offs * Q_d_stride).to(tl.float32)

    # ── Online softmax state ─────────────────────────────────────────────
    m = _NEG_INF_SENTINEL  # running max
    l = 0.0  # running Σ exp(w − m)
    acc = tl.zeros([D], dtype=tl.float32)  # running Σ p·v

    # ==================================================================
    # Phase 1 – Indexed keys (pre-computed weights from searcher)
    # ==================================================================
    iv_h_off = kv_h * IV_h_stride  # base offset for this KV head in values

    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # Load pre-computed scaled weights --------------------------------
        w = tl.load(
            IW_ptr + h_q * IW_h_stride + n_offs * IW_n_stride,
            mask=n_mask,
            other=0.0,
        ).to(tl.float32)

        # Zero → pruned / padded; replace with sentinel for softmax -------
        valid = w != 0.0
        w_safe = tl.where(valid, w, _NEG_INF_SENTINEL)

        # Online softmax update -------------------------------------------
        m_j = tl.max(w_safe, axis=0)
        m_new = tl.maximum(m, m_j)

        alpha = tl.exp(m - m_new)
        l = l * alpha
        acc = acc * alpha

        p = tl.where(valid, tl.exp(w - m_new), 0.0)
        l = l + tl.sum(p, axis=0)

        # Load values (BLOCK_N, D) and accumulate -------------------------
        v = tl.load(
            IV_ptr
            + iv_h_off
            + n_offs[:, None] * IV_n_stride
            + d_offs[None, :] * IV_d_stride,
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc = acc + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    # ==================================================================
    # Phase 2 – Queued keys (weights computed on-the-fly)
    # ==================================================================
    qk_h_off = kv_h * QK_h_stride
    qv_h_off = kv_h * QV_h_stride

    for q_start in range(0, Q_LEN, BLOCK_Q):
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < Q_LEN

        # Load queued keys (BLOCK_Q, D) -----------------------------------
        qk = tl.load(
            QK_ptr
            + qk_h_off
            + q_offs[:, None] * QK_n_stride
            + d_offs[None, :] * QK_d_stride,
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Compute attention weight: q · k * scaling -----------------------
        w = tl.sum(q[None, :] * qk, axis=1) * scaling
        w = tl.where(q_mask, w, _NEG_INF_SENTINEL)

        # Online softmax update -------------------------------------------
        m_j = tl.max(w, axis=0)
        m_new = tl.maximum(m, m_j)

        alpha = tl.exp(m - m_new)
        l = l * alpha
        acc = acc * alpha

        p = tl.where(q_mask, tl.exp(w - m_new), 0.0)
        l = l + tl.sum(p, axis=0)

        # Load queued values (BLOCK_Q, D) and accumulate ------------------
        qv = tl.load(
            QV_ptr
            + qv_h_off
            + q_offs[:, None] * QV_n_stride
            + d_offs[None, :] * QV_d_stride,
            mask=q_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc = acc + tl.sum(p[:, None] * qv, axis=0)
        m = m_new

    # ==================================================================
    # Normalize and store
    # ==================================================================
    out = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(
        OUT_ptr + h_q * OUT_h_stride + d_offs * OUT_d_stride,
        out.to(OUT_ptr.dtype.element_ty),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_hira_attention(
    index_weights: torch.Tensor,  # (H_q, N)  — from searcher, 0 = pruned
    index_values: torch.Tensor,  # (H_kv, N, D)
    query: torch.Tensor,  # (H_q, D)
    queued_keys: torch.Tensor,  # (H_kv, Q, D)
    queued_values: torch.Tensor,  # (H_kv, Q, D)
    q_head_to_kv: torch.Tensor,  # (H_q,)  int64
    scaling: float,
    BLOCK_N: int = 64,
    BLOCK_Q: int = 64,
) -> torch.Tensor:
    """Launch the fused HIRA attention kernel.

    Returns
    -------
    output : (H_q, D)  – attention output for a single decoding step.
    """
    H_q, D = query.shape
    N = index_weights.shape[1]
    Q_LEN = queued_keys.shape[1]  # may be 0

    output = torch.empty((H_q, D), device=query.device, dtype=query.dtype)

    grid = (H_q,)

    _fused_hira_attention_kernel[grid](
        # Index weights
        index_weights,
        index_weights.stride(0),
        index_weights.stride(1),
        # Index values
        index_values,
        index_values.stride(0),
        index_values.stride(1),
        index_values.stride(2),
        # Query
        query,
        query.stride(0),
        query.stride(1),
        # Queued keys
        queued_keys,
        queued_keys.stride(0),
        queued_keys.stride(1),
        queued_keys.stride(2),
        # Queued values
        queued_values,
        queued_values.stride(0),
        queued_values.stride(1),
        queued_values.stride(2),
        # Head mapping
        q_head_to_kv,
        # Output
        output,
        output.stride(0),
        output.stride(1),
        # Scalars
        scaling,
        N,
        Q_LEN,
        # Compile-time constants
        D=D,
        BLOCK_N=BLOCK_N,
        BLOCK_Q=BLOCK_Q,
    )

    return output
