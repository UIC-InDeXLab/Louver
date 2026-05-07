"""CUDA filtering kernels for TA-filter stage-1 parent/key masking."""

from .ta_filter_v_8_0 import KERNEL_VERSION as V8_0_VERSION
from .ta_filter_v_8_0 import ta_filter_v8_0

__all__ = [
    "ta_filter_v8_0",
    "V8_0_VERSION",
]
