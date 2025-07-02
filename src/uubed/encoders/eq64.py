#!/usr/bin/env python3
# this_file: src/uubed/encoders/eq64.py
"""Eq64: Full embedding encoder with visual dots for readability."""

from .q64 import q64_encode, q64_decode
from typing import Union, List


def eq64_encode(data: Union[bytes, List[int]]) -> str:
    """
    Encode full embedding with dots every 8 characters.

    Example: "ABCDEFGh.ijklmnop.qrstuvwx"

    Why dots? Makes it easier to visually compare embeddings
    and spot patterns during debugging.
    """
    base_encoded = q64_encode(data)

    # Insert dots for readability
    result = []
    for i, char in enumerate(base_encoded):
        if i > 0 and i % 8 == 0:
            result.append(".")
        result.append(char)

    return "".join(result)


def eq64_decode(encoded: str) -> bytes:
    """Decode Eq64 by removing dots and using standard q64 decode."""
    return q64_decode(encoded.replace(".", ""))