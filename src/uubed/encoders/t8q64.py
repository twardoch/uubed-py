#!/usr/bin/env python3
# this_file: src/uubed/encoders/t8q64.py
"""T8q64: Top-k indices encoder for sparse representation.

This module implements the T8q64 (Top-K QuadB64) encoding algorithm.
T8q64 is designed for creating compact, position-safe representations of
sparse or high-dimensional embedding vectors by encoding only the indices
of the `k` highest magnitude (or most important) values. This approach
is highly effective when only a few dimensions carry most of the information.

**Key Characteristics:**
- **Lossy Compression:** Achieves significant compression by discarding all but the `k` most significant values.
- **Sparsity-Aware:** Particularly well-suited for sparse embeddings where many values are zero or near-zero.
- **Compact Output:** The length of the encoded string is directly proportional to `k`, making it predictable and often very short.
- **Position-Safe:** The encoding preserves the original positional information of the selected top-k indices.

**Use Cases:**
- Encoding sparse vectors from models like Bag-of-Words, TF-IDF, or certain types of neural network activations.
- Creating compact representations for approximate nearest neighbor search where only dominant features are relevant.
- Reducing storage and transmission costs for high-dimensional data with inherent sparsity.

**Limitations:**
- **Information Loss:** All information outside the top `k` values is discarded, making it unsuitable for applications requiring full precision or reconstruction.
- **Sensitivity to `k`:** The choice of `k` is crucial and depends on the data's sparsity and the application's tolerance for information loss.
- **Magnitude-Based Selection:** Relies on the magnitude of values to determine importance, which might not always align with semantic importance in all embedding types.
"""

from typing import List, Union

import numpy as np

from .q64 import q64_encode


def top_k_q64(
    embedding: bytes,
    k: int = 8
) -> str:
    """
    Encodes an embedding vector by selecting and encoding the indices of its top-k highest magnitude values.

    This method is particularly useful for sparse representations where the absolute
    values of the embedding components indicate their importance. By focusing on the
    most significant dimensions, T8q64 creates a highly compact yet informative code.

    Args:
        embedding (bytes): The input embedding vector as a byte sequence (values 0-255).
                           Each byte represents a dimension's value.
        k (int): The number of top indices to keep. This determines the sparsity
                 of the representation. Must be a positive integer and not greater
                 than the embedding size. Defaults to 8.

    Returns:
        str: A position-safe q64 encoded string representing the top-k indices.
             The length of the output string depends on `k` (2 characters per index).

    Raises:
        ValueError: If `k` is non-positive or greater than the embedding size.

    Example:
        >>> from uubed.encoders.t8q64 import top_k_q64
        >>> import numpy as np

        >>> # Example 1: Basic usage with a sample embedding
        >>> emb = np.array([10, 200, 5, 150, 20, 255, 1, 100], dtype=np.uint8).tobytes()
        >>> encoded = top_k_q64(emb, k=3)
        >>> print(f"Original: {list(emb)}, Encoded (k=3): {encoded}")
        # Expected output (indices of 255, 200, 150): 5, 1, 3 -> q64_encode(bytes([1,3,5]))

        >>> # Example 2: When k is equal to embedding size
        >>> emb_small = np.array([10, 20, 30], dtype=np.uint8).tobytes()
        >>> encoded_small = top_k_q64(emb_small, k=3)
        >>> print(f"Original: {list(emb_small)}, Encoded (k=3): {encoded_small}")

        >>> # Example 3: Handling an embedding with fewer elements than k (will raise ValueError)
        >>> try:
        ...     top_k_q64(b'\x01\x02', k=5)
        ... except ValueError as e:
        ...     print(f"Error: {e}")
    """
    # Convert the input bytes to a NumPy array of unsigned 8-bit integers.
    # This allows for efficient numerical operations on the embedding values.
    embedding_array: np.ndarray = np.frombuffer(embedding, dtype=np.uint8)

    # Validate the 'k' parameter to ensure it's within acceptable bounds.
    if k <= 0:
        raise ValueError("'k' must be a positive integer.")
    if k > embedding_array.size:
        # If 'k' is larger than the actual number of elements in the embedding,
        # it's a logical error as we cannot select more indices than available.
        raise ValueError(f"'k' ({k}) cannot be greater than the embedding size ({embedding_array.size}).")

    # Find the indices of the k largest values in the embedding_array.
    # `np.argpartition` is an efficient way to find the indices of the k-th smallest/largest elements.
    # We use -k to get the indices of the k largest values.
    top_k_indices: np.ndarray = np.argpartition(embedding_array, -k)[-k:]

    # Sort the selected indices to ensure a consistent and reproducible order.
    # This is crucial for the encoded string to be deterministic and for proper decoding/comparison.
    sorted_indices: list[int] = sorted(top_k_indices.tolist())

    # Clamp indices to fit within a single byte (0-255).
    # This step is a safeguard, as indices should naturally be within this range
    # if the embedding size is <= 256. If an original index is > 255, information
    # about its exact position is lost, but its presence in the top-k is retained.
    clamped_indices: list[int] = [min(idx, 255) for idx in sorted_indices]

    # Ensure the list of indices has exactly `k` elements.
    # This padding is typically not needed if `k` is validated against `embedding_array.size`,
    # but it acts as a robustness measure. Padding with 255 (max uint8 value) is a convention.
    while len(clamped_indices) < k:
        clamped_indices.append(255)

    # Encode the list of clamped and padded indices into a q64 string.
    # The `bytes()` constructor converts the list of integers (0-255) into a byte sequence.
    return q64_encode(bytes(clamped_indices))
