#!/usr/bin/env python3
# this_file: src/uubed/native_wrapper.py
"""Wrapper for native module with fallback to pure Python."""

try:
    # Try to import native module
    from . import _native
    q64_encode_native = _native.q64_encode_native
    q64_decode_native = _native.q64_decode_native
    simhash_q64_native = _native.simhash_q64_native
    top_k_q64_native = _native.top_k_q64_native
    z_order_q64_native = _native.z_order_q64_native
    HAS_NATIVE = True
except ImportError:
    # Fall back to pure Python
    HAS_NATIVE = False

    # Import pure Python implementations
    from .encoders.q64 import q64_encode as q64_encode_native
    from .encoders.q64 import q64_decode as q64_decode_native
    from .encoders.shq64 import simhash_q64 as simhash_q64_native
    from .encoders.t8q64 import top_k_q64 as top_k_q64_native
    from .encoders.zoq64 import z_order_q64 as z_order_q64_native


def is_native_available() -> bool:
    """Check if native acceleration is available."""
    return HAS_NATIVE