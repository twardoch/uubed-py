#!/usr/bin/env python3
# this_file: src/uubed/encoders/shq64.py
"""Shq64: SimHash encoder for similarity-preserving compact codes.

This module implements the Shq64 (SimHash QuadB64) encoding algorithm.
Shq64 generates a compact, fixed-length, position-safe code that preserves
the similarity of the original embedding vectors. It achieves this by
projecting the embedding onto a set of randomly generated hyperplanes and
encoding the resulting sign bits. The core idea is that if two vectors are
close in the original space, their projections onto random hyperplanes are
likely to have the same sign, leading to similar (or identical) SimHash codes.

**Key Characteristics:**
- **Lossy Compression:** SimHash is a lossy compression technique, meaning the
  original embedding cannot be perfectly reconstructed from the SimHash code.
- **Similarity Preservation:** Designed to produce similar hash codes for similar
  input embeddings, making it suitable for approximate nearest neighbor (ANN)
  search, deduplication, and clustering.
- **Fixed Length Output:** The length of the generated SimHash code (and thus
  the q64 string) is determined by the `planes` parameter, providing a predictable
  and compact representation.
- **Position-Safe:** The resulting q64 string is position-safe, ensuring that
  minor changes in the input embedding lead to small, predictable changes in the hash.

**Use Cases:**
- Near-duplicate detection in large datasets (e.g., images, documents, vectors).
- Fast similarity search in vector databases where exact matches are not required.
- Reducing the dimensionality of high-dimensional data while retaining semantic similarity.

**Limitations:**
- **Information Loss:** Not suitable for applications requiring perfect reconstruction
  of the original embedding.
- **Sensitivity to `planes`:** The number of `planes` directly impacts the hash
  length and the granularity of similarity detection. A higher number of planes
  provides more precision but results in a longer hash.
- **Randomness:** The effectiveness of SimHash depends on the quality and randomness
  of the generated hyperplanes. A fixed seed is used for reproducibility.
"""

import numpy as np
from .q64 import q64_encode
from typing import List, Union


def simhash_q64(
    embedding: bytes,
    planes: int = 64,
    seed: int = 42
) -> str:
    """
    Generates a position-safe SimHash code from an embedding vector.

    This function implements the core SimHash logic: it projects the input
    embedding onto a set of random hyperplanes and encodes the sign of each
    projection as a bit. These bits are then packed into bytes and converted
    into a q64 string. The resulting SimHash code is compact and designed
    to preserve the similarity of the original embeddings.

    Args:
        embedding (bytes): The input embedding vector as a byte sequence.
                           Each byte is treated as a dimension with a value from 0-255.
        planes (int): The number of random projection planes to use. This parameter
                      directly determines the length of the SimHash code (number of bits).
                      It must be a multiple of 8, as the bits are packed into bytes.
                      Defaults to 64, which results in an 8-byte (16-character q64) hash.
        seed (int): An integer seed for the random number generator used to create
                    the hyperplanes. Using a fixed seed ensures that the SimHash
                    codes are reproducible for the same input embedding and `planes`
                    value across different runs.
                    Defaults to 42.

    Returns:
        str: A position-safe q64 encoded string representing the SimHash of the embedding.
             The length of the string will be `planes / 4` characters (since each byte
             encodes to 2 q64 characters, and `planes` bits form `planes / 8` bytes).

    Raises:
        ValueError: If `planes` is not a multiple of 8.

    Example:
        >>> from uubed.encoders.shq64 import simhash_q64
        >>> import numpy as np

        >>> # Example 1: Basic usage with a 768-dimensional embedding
        >>> emb = np.random.randint(0, 256, 768, dtype=np.uint8).tobytes()
        >>> encoded_64_planes = simhash_q64(emb, planes=64)
        >>> print(f"Encoded (64 planes): {encoded_64_planes}")

        >>> # Example 2: Using more planes for a longer, potentially more precise hash
        >>> encoded_128_planes = simhash_q64(emb, planes=128)
        >>> print(f"Encoded (128 planes): {encoded_128_planes}")

        >>> # Example 3: Ensuring reproducibility with a fixed seed
        >>> emb2 = np.random.randint(0, 256, 128, dtype=np.uint8).tobytes()
        >>> hash1 = simhash_q64(emb2, seed=100)
        >>> hash2 = simhash_q64(emb2, seed=100)
        >>> print(f"Reproducible hashes: {hash1 == hash2}")

        >>> # Example 4: Handling invalid `planes` value
        >>> try:
        ...     simhash_q64(emb, planes=60) # Not a multiple of 8
        ... except ValueError as e:
        ...     print(f"Error: {e}")
    """
    # Validate the `planes` parameter. Although `validate_method_parameters` in `api.py`
    # handles this for top-level calls, it's good practice to keep this check for direct
    # calls to `simhash_q64`.
    if planes % 8 != 0:
        raise ValueError("'planes' must be a multiple of 8.")

    # Initialize a NumPy random number generator with the specified seed.
    # This ensures that the random hyperplanes generated are consistent across runs
    # for the same seed, leading to reproducible SimHash codes.
    rng: np.random.Generator = np.random.default_rng(seed)

    # Convert the input `bytes` embedding into a NumPy array of unsigned 8-bit integers.
    # This allows for efficient numerical operations. `embedding_dim` stores the size
    # of the input embedding.
    embedding_array: np.ndarray = np.frombuffer(embedding, dtype=np.uint8)
    embedding_dim: int = embedding_array.shape[0]

    # Generate a matrix of random hyperplanes. Each row of `rand_vectors` represents
    # a hyperplane. The values are drawn from a normal distribution.
    rand_vectors: np.ndarray = rng.normal(size=(planes, embedding_dim))

    # Center the embedding values around zero and convert them to float.
    # SimHash typically works best with values centered around 0. The original
    # byte values (0-255) are mapped to a range of approximately -1 to 1.
    vec_centered: np.ndarray = (embedding_array.astype(float) - 128) / 128

    # Project the centered embedding onto each random hyperplane.
    # The dot product (`@` operator) of the embedding with each hyperplane vector
    # gives the projection. The sign of this projection determines the SimHash bit.
    projections: np.ndarray = rand_vectors @ vec_centered

    # Convert the signs of the projections into binary bits (0 or 1).
    # If a projection is positive, the bit is 1; otherwise, it's 0.
    bits: np.ndarray = (projections > 0).astype(int)

    # Pack the individual bits into 8-bit bytes.
    # This loop iterates through the `bits` array, taking 8 bits at a time
    # to construct each byte. The bits are arranged from most significant (MSB) to least significant (LSB).
    byte_data: List[int] = []
    for i in range(0, len(bits), 8):
        byte_val: int = 0
        # For each group of 8 bits, construct a single byte.
        for j in range(8):
            # Ensure we don't go out of bounds if the total number of bits is not a multiple of 8.
            # This handles cases where `planes` might not be perfectly divisible by 8 (though validated).
            if i + j < len(bits):
                # Shift the bit to its correct position within the byte (MSB first).
                # `(7 - j)` ensures the first bit goes to position 7, second to 6, etc.
                byte_val |= int(bits[i + j]) << (7 - j)
        byte_data.append(byte_val)

    # Encode the resulting list of bytes into a position-safe q64 string.
    # The `bytes()` constructor converts the list of integers (0-255) into a byte sequence.
    return q64_encode(bytes(byte_data))
