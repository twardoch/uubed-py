#!/usr/bin/env python3
# this_file: src/uubed/encoders/shq64.py
"""Shq64: SimHash encoder for similarity-preserving compact codes."""

import numpy as np
from .q64 import q64_encode
from typing import List


def simhash_q64(embedding: List[int], planes: int = 64) -> str:
    """
    Generate position-safe SimHash code.

    How it works:
    1. Project embedding onto 64 random hyperplanes
    2. Store sign bit (which side of hyperplane)
    3. Similar embeddings → similar bit patterns → similar codes

    Args:
        embedding: List of byte values (0-255)
        planes: Number of random projections (must be multiple of 8)

    Returns:
        16-character q64 string (for 64 planes)
    """
    # Use fixed seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate random projection matrix
    rand_vectors = rng.normal(size=(planes, len(embedding)))

    # Convert bytes to centered floats
    vec = np.array(embedding, dtype=float)
    vec = (vec - 128) / 128  # Center around 0

    # Project and get sign bits
    projections = rand_vectors @ vec
    bits = (projections > 0).astype(int)

    # Pack bits into bytes
    byte_data = []
    for i in range(0, len(bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bits):
                byte_val |= int(bits[i + j]) << (7 - j)
        byte_data.append(byte_val)

    return q64_encode(bytes(byte_data))