# Key-norm insight report

- shape (L, H, T) = (28, 4, 193)
- prompt length = 65

## Q1: Are key norms the same?

**No.** The L2 norm of $k_{\ell,h,t}$ varies across every axis (layer, head, position). Aggregate stats over all non-NaN entries:

| min | max | mean | std | max/min |
|---|---|---|---|---|
| 0.372 | 923.234 | 39.984 | 100.074 | 2484.49× |

## Q2: Do they vary in a meaningful way?

**Yes — variation is structured.** Marginalising one axis at a time we still see large spread:

| marginal | min | max | mean | max/min |
|---|---|---|---|---|
| per-layer mean | 15.960 | 273.969 | 39.984 | 17.17× |
| per-head mean | 25.050 | 61.777 | 39.984 | 2.47× |
| per-position mean | 29.738 | 42.147 | 39.984 | 1.42× |

Interpretation: the per-layer ratio shows the **vertical** structure (deeper layers tend to have systematically different scale); the per-head ratio shows that **even within one layer**, different KV heads operate at different scales; the per-position ratio captures the **positional** trend, including a typically larger norm at the very first tokens (BOS / system header) and at the decode boundary.

## Implication for sparse attention

Attention scores $s_{t,j} = q_t^\top k_j / \sqrt{d}$ scale linearly with $\|k_j\|$, so a fixed score / cosine threshold translates into different effective angular thresholds at different positions and heads. Methods that normalise away $\|k\|$ (cosine / inner-product MIPS indices) discard a real signal: the norm itself encodes how strongly a key wants to be retrieved. Range searching on the raw key vectors retains this magnitude.
