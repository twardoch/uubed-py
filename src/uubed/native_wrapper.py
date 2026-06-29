#!/usr/bin/env python3
# this_file: src/uubed/native_wrapper.py
"""Wrapper for the optional Rust-accelerated extension module with pure-Python fallback.

At import time this module attempts to load ``uubed._native``, a compiled
extension built by maturin/PyO3 (``uubed-rs``).  If the ``.so``/``.pyd``
file is absent — e.g. in a sdist-only install or a development environment
without a Rust toolchain — the import raises ``ImportError`` and we fall back
transparently to the pure-Python implementations under ``uubed.encoders.*``.

The public names exported here (``q64_encode_native``, etc.) are therefore
always callable regardless of whether the native extension is present.
Callers can query ``HAS_NATIVE`` or ``is_native_available()`` to find out
which code path is active, but for correctness they should not need to.

Performance note: the native Rust path is typically 10-100× faster for
large embeddings.  The pure-Python path is correct but slower; it is
suitable for testing, CI, and environments where building Rust is not
feasible.
"""

try:
    # Attempt to import the Rust extension module compiled by maturin.
    # ``_native`` is a PyO3 cdylib; its presence depends on whether the
    # wheel was built with ``maturin develop`` / ``pip install uubed[native]``.
    from . import _native
    q64_encode_native = _native.q64_encode_native
    q64_decode_native = _native.q64_decode_native
    simhash_q64_native = _native.simhash_q64_native
    top_k_q64_native = _native.top_k_q64_native
    z_order_q64_native = _native.z_order_q64_native
    mq64_encode_native = _native.mq64_encode_native
    mq64_decode_native = _native.mq64_decode_native
    # HAS_NATIVE=True signals callers that Rust acceleration is active.
    HAS_NATIVE = True
except ImportError:
    # Native extension not available — bind the pure-Python implementations
    # under the same names so the rest of the package is unaffected.
    HAS_NATIVE = False

    from .encoders.mq64 import mq64_decode as mq64_decode_native
    from .encoders.mq64 import mq64_encode as mq64_encode_native
    from .encoders.q64 import q64_decode as q64_decode_native
    from .encoders.q64 import q64_encode as q64_encode_native
    from .encoders.shq64 import simhash_q64 as simhash_q64_native
    from .encoders.t8q64 import top_k_q64 as top_k_q64_native
    from .encoders.zoq64 import z_order_q64 as z_order_q64_native


def is_native_available() -> bool:
    """Return True if the Rust-accelerated extension module is loaded.

    Use this to emit diagnostics or choose optimised code paths.
    All ``*_native`` names in this module are callable regardless of the
    return value — the pure-Python fallback has the same interface.
    """
    return HAS_NATIVE
