#!/usr/bin/env python3
# this_file: src/uubed/__init__.pyi
"""Type stubs for uubed package."""

from .__version__ import __version__
from .api import decode, encode
from .exceptions import (
    UubedConfigurationError,
    UubedConnectionError,
    UubedDecodingError,
    UubedEncodingError,
    UubedError,
    UubedResourceError,
    UubedValidationError,
)
from .streaming import (
    StreamingEncoder,
    batch_encode,
    decode_stream,
    encode_file_stream,
    encode_stream,
)
from .validation import (
    estimate_memory_usage,
    validate_embedding_input,
    validate_encoding_method,
    validate_file_path,
    validate_memory_usage,
    validate_method_parameters,
)

# GPU functions (if available)
def is_gpu_available() -> bool: ...
def get_gpu_info() -> dict: ...
def gpu_encode_batch(embeddings: list, method: str = "auto", **kwargs) -> list: ...

class GPUStreamingEncoder:
    def __init__(self, method: str = "auto", batch_size: int = 100, **kwargs) -> None: ...

def benchmark_gpu_vs_cpu(
    embeddings: list,
    method: str = "auto",
    iterations: int = 5,
    **kwargs
) -> dict: ...

# Matryoshka functions (if available)
class MatryoshkaEncoder:
    def __init__(self, dimensions: list, strategies: dict = None) -> None: ...

class MatryoshkaSearchIndex:
    def __init__(self, encoder: MatryoshkaEncoder) -> None: ...

def create_adaptive_matryoshka_encoder(
    max_dimensions: int,
    levels: int = 5,
    strategy: str = "adaptive"
) -> MatryoshkaEncoder: ...
