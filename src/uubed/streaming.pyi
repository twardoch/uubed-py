#!/usr/bin/env python3
# this_file: src/uubed/streaming.pyi
"""Type stubs for uubed.streaming module."""

from typing import Iterator, List, Union, Optional, Any, BinaryIO
import numpy as np
from .api import EncodingMethod

def encode_stream(
    embeddings: Iterator[Union[List[int], np.ndarray, bytes]],
    method: EncodingMethod = "auto",
    batch_size: int = 100,
    progress: bool = False,
    **kwargs
) -> Iterator[str]:
    """
    Stream encode embeddings with batching and optional progress tracking.

    Args:
        embeddings: Iterator of embedding vectors
        method: Encoding method to use
        batch_size: Number of embeddings to process in each batch
        progress: Whether to show progress bar
        **kwargs: Method-specific parameters

    Yields:
        Encoded strings
    """
    ...

def encode_file_stream(
    file_path: str,
    embedding_size: int,
    method: EncodingMethod = "auto",
    batch_size: int = 100,
    progress: bool = False,
    **kwargs
) -> Iterator[str]:
    """
    Stream encode embeddings directly from binary file.

    Args:
        file_path: Path to binary file containing embeddings
        embedding_size: Size of each embedding in bytes
        method: Encoding method to use
        batch_size: Number of embeddings to process in each batch
        progress: Whether to show progress bar
        **kwargs: Method-specific parameters

    Yields:
        Encoded strings
    """
    ...

def decode_stream(
    encoded_strings: Iterator[str],
    method: Optional[EncodingMethod] = None,
    batch_size: int = 100,
    progress: bool = False
) -> Iterator[bytes]:
    """
    Stream decode encoded strings.

    Args:
        encoded_strings: Iterator of encoded strings
        method: Encoding method used for encoding
        batch_size: Number of strings to process in each batch
        progress: Whether to show progress bar

    Yields:
        Decoded bytes
    """
    ...

def batch_encode(
    embeddings: List[Union[List[int], np.ndarray, bytes]],
    method: EncodingMethod = "auto",
    **kwargs
) -> List[str]:
    """
    Batch encode multiple embeddings efficiently.

    Args:
        embeddings: List of embedding vectors
        method: Encoding method to use
        **kwargs: Method-specific parameters

    Returns:
        List of encoded strings
    """
    ...

class StreamingEncoder:
    """Context manager for streaming encoding operations."""
    
    def __init__(
        self,
        method: EncodingMethod = "auto",
        batch_size: int = 100,
        progress: bool = False,
        **kwargs
    ) -> None: ...
    
    def __enter__(self) -> "StreamingEncoder": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    
    def encode_stream(
        self,
        embeddings: Iterator[Union[List[int], np.ndarray, bytes]]
    ) -> Iterator[str]: ...
    
    def encode_file_stream(
        self,
        file_path: str,
        embedding_size: int
    ) -> Iterator[str]: ...