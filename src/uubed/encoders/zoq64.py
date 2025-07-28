#!/usr/bin/env python3
# this_file: src/uubed/encoders/zoq64.py
"""Zoq64: Z-order (Morton code) encoder for spatial locality.

This module implements the Zoq64 (Z-order QuadB64) encoding algorithm.
Zoq64 leverages the Z-order curve (also known as Morton code) to transform
multi-dimensional embedding vectors into a single-dimensional, position-safe
code. The primary benefit of Z-order curves is their ability to preserve
spatial locality: points that are close in the original multi-dimensional
space tend to have Z-order codes with similar prefixes. This property is
invaluable for efficient spatial indexing, range queries, and approximate
nearest neighbor searches in vector databases.

**Key Characteristics:**
- **Lossy Compression:** Achieves high compression by quantizing each dimension
  to a very small number of bits (currently 2 bits per dimension) and processing
  only a limited number of dimensions (currently the first 16).
- **Spatial Locality Preservation:** Designed to maintain the relative proximity
  of original embedding vectors in the encoded space, making it suitable for
  approximate nearest neighbor (ANN) searches and spatial indexing.
- **Compact Output:** Produces a very short, fixed-length q64 string (8 characters)
  regardless of the original embedding's size (up to 16 dimensions).

**Use Cases:**
- Generating compact, searchable representations of high-dimensional data.
- Indexing large datasets for fast approximate spatial queries.
- Scenarios where extreme compression and spatial proximity are prioritized
  over perfect reconstruction or full precision.

**Limitations:**
- **Fixed Dimensionality:** Currently processes only the first 16 dimensions.
  Embeddings with more dimensions will have their additional dimensions ignored.
- **High Information Loss:** Due to aggressive quantization, this method is not
  suitable for applications requiring high precision or exact reconstruction
  of the original embedding.
"""

import struct
from typing import List, Union

from .q64 import q64_encode


def z_order_q64(embedding: bytes) -> str:
    """
    Encodes an embedding vector into a Z-order (Morton) code and then to a q64 string.

    This function transforms a multi-dimensional embedding into a single Z-order
    integer by quantizing each dimension and interleaving their bits. The resulting
    integer is then packed into bytes and encoded into a compact q64 string.
    This method is highly lossy but effective for preserving spatial locality.

    Args:
        embedding (bytes): The input embedding vector as a byte sequence. Each byte
                           is treated as a dimension with a value from 0-255.
                           Only the first 16 dimensions of the embedding are used for encoding.
                           Any dimensions beyond the 16th are ignored.

    Returns:
        str: An 8-character q64 encoded string representing the Z-order code.
             The fixed length is derived from 16 dimensions * 2 bits/dimension = 32 bits,
             which packs into 4 bytes, and 4 bytes encode to 8 q64 characters.

    Note:
        This implementation applies aggressive quantization and truncation:
        - Each dimension is quantized to only 2 bits (effectively mapping 0-255 to 0-3).
        - Only the first 16 dimensions of the input embedding are considered.
        This makes the encoding highly compact but results in significant information loss.
        It is best suited for scenarios where approximate spatial relationships are
        more critical than precise value reconstruction.
    """
    # Step 1: Quantize each byte (dimension) of the embedding to 2 bits.
    # This is achieved by right-shifting each 8-bit value by 6 positions
    # (effectively taking the top 2 bits) and then masking with 0b11 to ensure
    # only these 2 bits are retained.
    # Example: For a byte value of 213 (0b11010101):
    #   (213 >> 6) -> 0b00000011 (3)
    #   (3 & 0b11) -> 0b00000011 (3)
    quantized: list[int] = [(b >> 6) & 0b11 for b in embedding]

    # Step 2: Interleave bits for the first 16 dimensions to form a single Z-order integer.
    # The Z-order curve (Morton code) is constructed by interleaving the bits of the
    # coordinates. For 16 dimensions, each contributing 2 bits, the resulting Z-order
    # code will be 32 bits long (16 * 2 = 32).
    result: int = 0
    # Iterate only over the first 16 quantized dimensions to adhere to the fixed dimensionality.
    for i, val in enumerate(quantized[:16]):
        # For each quantized dimension (val), iterate over its 2 bits.
        for bit_pos in range(2):
            # Extract the current bit (0 or 1) from the quantized value.
            bit: int = (val >> bit_pos) & 1
            # Place the extracted bit into the `result` integer at its interleaved position.
            # The position is calculated to ensure bits from different dimensions are interleaved:
            # (dimension_index * bits_per_dimension + bit_within_dimension).
            # For example, for dimension 0, bits go to positions 0, 1. For dimension 1, positions 2, 3, etc.
            result |= bit << (i * 2 + bit_pos)

    # Step 3: Pack the 32-bit Z-order integer into 4 bytes.
    # ">I" specifies big-endian (network byte order) for an unsigned integer.
    packed: bytes = struct.pack(">I", result)

    # Step 4: Encode the 4 resulting bytes into an 8-character q64 string.
    # The q64 encoding converts each byte into two base64-like characters.
    return q64_encode(packed)
