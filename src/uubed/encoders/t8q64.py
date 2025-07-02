#!/usr/bin/env python3
# this_file: src/uubed/encoders/t8q64.py
"""T8q64: Top-k indices encoder for sparse representation."""

import numpy as np
from .q64 import q64_encode
from typing import List


def top_k_q64(embedding: List[int], k: int = 8) -> str:
    """
    Encode top-k highest magnitude indices.

    Why this works: Important features tend to have extreme values.
    By storing only the indices of the k largest values, we get
    a sparse but effective representation.

    Args:
        embedding: List of byte values (0-255)
        k: Number of top indices to keep

    Returns:
        16-character q64 string (for k=8)
    """
    # Get indices of k largest values
    indices = np.argsort(np.array(embedding))[-k:]

    # Sort indices for consistent encoding
    indices = sorted(indices.tolist())

    # Clamp indices to fit in a byte (max 255)
    # For embeddings larger than 256, we lose precision
    indices = [min(idx, 255) for idx in indices]

    # Ensure we have exactly k indices (pad with 255 if needed)
    while len(indices) < k:
        indices.append(255)

    return q64_encode(bytes(indices))