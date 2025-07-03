#!/usr/bin/env python3
# this_file: src/uubed/api.pyi
"""Type stubs for uubed.api module."""

from typing import Union, List, Literal, Optional
import numpy as np

EncodingMethod = Literal["eq64", "shq64", "t8q64", "zoq64", "auto"]

def encode(
    embedding: Union[List[int], np.ndarray, bytes],
    method: EncodingMethod = "auto",
    **kwargs
) -> str:
    """
    Encode embedding vector using specified method with comprehensive validation.

    Args:
        embedding: Vector to encode (list/array of 0-255 integers, bytes, or float arrays in [0,1] or [0,255])
        method: Encoding method or "auto" for automatic selection
        **kwargs: Method-specific parameters
            - planes (int): Number of hash planes for shq64 (default: 64, must be multiple of 8)
            - k (int): Number of top elements to keep for t8q64 (default: 8)

    Returns:
        Encoded string in q64 format

    Raises:
        UubedValidationError: If input parameters are invalid
        UubedEncodingError: If encoding operation fails
    """
    ...

def decode(encoded: str, method: Optional[EncodingMethod] = None) -> bytes:
    """
    Decode encoded string back to bytes with comprehensive validation.

    Args:
        encoded: Encoded string in q64 format
        method: Encoding method used for encoding (auto-detected if None)

    Returns:
        Original bytes (only for eq64 method)

    Raises:
        UubedValidationError: If input parameters are invalid
        UubedDecodingError: If decoding operation fails
        NotImplementedError: If trying to decode lossy compression methods
    """
    ...