"""attention_v2.19 — v2.18 (optimized aux) with num_splits=170."""
from .attention_v2_18 import attend as _attend
KERNEL_VERSION = "v2.19"
def attend(q, th_per_subspace, state, buffer_keys, buffer_values,
           keys_children, q_head_to_kv=None, scale=None):
    return _attend(q, th_per_subspace, state, buffer_keys, buffer_values,
                   keys_children, q_head_to_kv, scale, num_splits=170)
KERNEL = attend
