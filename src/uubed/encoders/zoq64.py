#!/usr/bin/env python3
# this_file: src/uubed/encoders/zoq64.py
"""Zoq64: Z-order (Morton code) encoder for spatial locality."""

import struct
from .q64 import q64_encode
from typing import List


def z_order_q64(embedding: List[int]) -> str:
    """
    Encode using Z-order (Morton) curve.

    Why Z-order? Space-filling curves preserve spatial locality:
    nearby points in high-dimensional space get similar prefixes,
    enabling efficient prefix searches and range queries.

    Args:
        embedding: List of byte values (0-255)

    Returns:
        8-character q64 string
    """
    # Quantize to 2 bits per dimension (take top 2 bits)
    quantized = [(b >> 6) & 0b11 for b in embedding]

    # Interleave bits for first 16 dimensions
    result = 0
    for i, val in enumerate(quantized[:16]):
        for bit_pos in range(2):
            bit = (val >> bit_pos) & 1
            result |= bit << (i * 2 + bit_pos)

    # Pack as 4 bytes
    packed = struct.pack(">I", result)
    return q64_encode(packed)