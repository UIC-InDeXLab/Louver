"""attention_v2.17 — v2.15 with num_splits=42."""
from .attention_v2_15 import attend as _attend
KERNEL_VERSION = "v2.17"
def attend(q, th_per_subspace, state, buffer_keys, buffer_values,
           keys_children, q_head_to_kv=None, scale=None):
    return _attend(q, th_per_subspace, state, buffer_keys, buffer_values,
                   keys_children, q_head_to_kv, scale, num_splits=42)
KERNEL = attend
