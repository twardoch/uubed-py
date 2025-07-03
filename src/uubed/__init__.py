#!/usr/bin/env python3
# this_file: src/uubed/__init__.py
"""
uubed: High-performance encoding for embedding vectors.

Solves the "substring pollution" problem in search systems by using
position-dependent alphabets that prevent false matches.
"""

from .__version__ import __version__
from .api import encode, decode
from .streaming import (
    encode_stream,
    encode_file_stream,
    decode_stream,
    batch_encode,
    StreamingEncoder,
)
from .exceptions import (
    UubedError,
    UubedValidationError,
    UubedEncodingError,
    UubedDecodingError,
    UubedResourceError,
    UubedConnectionError,
    UubedConfigurationError,
)
from .validation import (
    validate_encoding_method,
    validate_embedding_input,
    validate_method_parameters,
    validate_file_path,
    validate_memory_usage,
    estimate_memory_usage,
)
from .config import (
    get_config,
    load_config,
    get_setting,
    set_setting,
    create_default_config,
)

# Optional imports with graceful fallback
try:
    from .gpu import (
        is_gpu_available,
        get_gpu_info,
        gpu_encode_batch,
        GPUStreamingEncoder,
        benchmark_gpu_vs_cpu,
    )
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False

try:
    from .matryoshka import (
        MatryoshkaEncoder,
        MatryoshkaSearchIndex,
        create_adaptive_matryoshka_encoder,
    )
    _MATRYOSHKA_AVAILABLE = True
except ImportError:
    _MATRYOSHKA_AVAILABLE = False

__all__ = [
    # Core API
    "encode",
    "decode",
    "__version__",
    # Streaming API
    "encode_stream",
    "encode_file_stream",
    "decode_stream",
    "batch_encode",
    "StreamingEncoder",
    # Exception classes
    "UubedError",
    "UubedValidationError",
    "UubedEncodingError",
    "UubedDecodingError",
    "UubedResourceError",
    "UubedConnectionError",
    "UubedConfigurationError",
    # Validation functions
    "validate_encoding_method",
    "validate_embedding_input",
    "validate_method_parameters",
    "validate_file_path",
    "validate_memory_usage",
    "estimate_memory_usage",
    # Configuration functions
    "get_config",
    "load_config",
    "get_setting",
    "set_setting",
    "create_default_config",
]

# Add GPU functions if available
if _GPU_AVAILABLE:
    __all__.extend([
        "is_gpu_available",
        "get_gpu_info",
        "gpu_encode_batch",
        "GPUStreamingEncoder",
        "benchmark_gpu_vs_cpu",
    ])

# Add Matryoshka functions if available
if _MATRYOSHKA_AVAILABLE:
    __all__.extend([
        "MatryoshkaEncoder",
        "MatryoshkaSearchIndex",
        "create_adaptive_matryoshka_encoder",
    ])