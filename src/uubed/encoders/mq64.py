#!/usr/bin/env python3
"""Matryoshka QuadB64 (Mq64) pure Python fallback encoder.

This module provides a pure Python implementation of the Matryoshka QuadB64 (Mq64)
encoding and decoding. Mq64 is a hierarchical encoding scheme designed to represent
embedding vectors at multiple levels of granularity or precision. Each level in the
Matryoshka structure encodes a prefix of the original data, allowing for efficient
retrieval and comparison at different resolutions.

**Key Characteristics:**
- **Hierarchical Encoding:** Allows an embedding to be represented by multiple
  encoded strings, where each string corresponds to a specific prefix length
  of the original data.
- **Lossy (typically):** While the underlying `q64_encode` is lossless for each
  segment, the concept of Matryoshka encoding often implies using shorter prefixes
  for coarser representations, which are inherently lossy with respect to the full data.
- **Flexible Levels:** Supports custom definition of encoding levels, allowing users
  to specify the lengths of the prefixes to be encoded.
- **Pure Python Fallback:** This implementation serves as a pure Python fallback
  for environments where the native Rust implementation is not available.

**Use Cases:**
- Multi-resolution search in vector databases, where initial coarse searches can
  be performed on shorter Mq64 codes, followed by finer-grained searches on longer codes.
- Progressive loading or streaming of embedding data.
- Applications where different levels of detail or precision are required for embeddings.

**Limitations:**
- **Performance:** As a pure Python implementation, its performance will be
  significantly slower than the native Rust version.
- **Memory Usage:** May consume more memory compared to optimized native implementations
  for very large embeddings or numerous levels.
"""
from typing import List, Optional, Union
from .q64 import q64_encode, q64_decode

def mq64_encode(
    data: Union[bytes, bytearray],
    levels: Optional[List[int]] = None
) -> str:
    """
    Encodes input data into a Matryoshka QuadB64 (Mq64) string.

    Mq64 creates a hierarchical representation of the input data by encoding
    successive prefixes of the data at different specified `levels`. This allows
    for a multi-resolution representation, where shorter encoded strings (earlier levels)
    provide a coarser, more compact view, and longer strings (later levels) offer
    finer detail.

    Args:
        data (Union[bytes, bytearray]): The input data to encode. This should be
                                        the full byte sequence of the embedding.
        levels (Optional[List[int]]): A list of integers specifying the lengths
                                      of the prefixes of `data` to encode. Each integer
                                      represents a byte length. If `None`, default levels
                                      are derived based on the length of the input `data`,
                                      typically powers of two multiples of 64 up to the data length.

    Returns:
        str: The Mq64 encoded string. Different levels are separated by a colon (`:`).
             Example: "level1_q64_string:level2_q64_string:level3_q64_string"

    Example:
        >>> from uubed.encoders.mq64 import mq64_encode
        >>>
        >>> # Example 1: Encoding with default levels
        >>> data = bytes(range(256)) # A 256-byte embedding
        >>> encoded_default = mq64_encode(data)
        >>> print(f"Encoded (default levels): {encoded_default[:50]}...")

        >>> # Example 2: Encoding with custom levels
        >>> custom_levels_data = bytes(range(128))
        >>> custom_encoded = mq64_encode(custom_levels_data, levels=[16, 64, 128])
        >>> print(f"Encoded (custom levels): {custom_encoded}")

        >>> # Example 3: Encoding an empty data string
        >>> empty_encoded = mq64_encode(b"")
        >>> print(f"Encoded (empty data): '{empty_encoded}'")
    """
    # Derive default levels if not provided.
    # The default strategy is to create levels at powers of two multiples of 64,
    # up to and including the full length of the input data.
    length: int = len(data)
    if levels is None:
        levels = []
        lvl = 64
        while lvl < length:
            levels.append(lvl)
            lvl *= 2
        levels.append(length) # Always include the full length as the last level.

    # Encode each specified level (prefix) of the data.
    parts: List[str] = []
    for lvl in levels:
        # Only encode levels that are less than or equal to the actual data length.
        if lvl <= length:
            # Encode the prefix of the data up to the current level's length.
            # q64_encode handles the conversion of bytes to q64 string.
            parts.append(q64_encode(data[:lvl]))
    
    # Join the encoded parts with a colon to form the final Mq64 string.
    return ":".join(parts)

def mq64_decode(
    encoded: str
) -> bytes:
    """
    Decodes an Mq64 encoded string back into its original byte sequence.

    This function extracts the last (most detailed) level from the Mq64 string
    (which is typically the full data representation) and decodes it using the
    standard q64 decoder.

    Args:
        encoded (str): The Mq64 encoded string to decode. This string is expected
                       to be composed of multiple q64 segments separated by colons (`:`).

    Returns:
        bytes: The original byte sequence corresponding to the last (most detailed)
               level of the Mq64 encoding. Returns an empty `bytes` object if the
               input `encoded` string is empty or contains no decodable segments.

    Example:
        >>> from uubed.encoders.mq64 import mq64_encode, mq64_decode
        >>>
        >>> # Encode some data with multiple levels
        >>> data = bytes(range(128))
        >>> encoded_mq64 = mq64_encode(data, levels=[16, 64, 128])
        >>> print(f"Encoded: {encoded_mq64}")

        >>> # Decode the Mq64 string back to bytes
        >>> decoded_bytes = mq64_decode(encoded_mq64)
        >>> print(f"Decoded: {decoded_bytes[:20]}...")
        >>> assert decoded_bytes == data

        >>> # Decoding an empty string
        >>> empty_decoded = mq64_decode("")
        >>> print(f"Decoded (empty): '{empty_decoded}'")
        >>> assert empty_decoded == b''

        >>> # Decoding a string with only one level
        >>> single_level_encoded = mq64_encode(b"\x00\x01\x02\x03", levels=[4])
        >>> decoded_single_level = mq64_decode(single_level_encoded)
        >>> print(f"Decoded (single level): {decoded_single_level}")
        >>> assert decoded_single_level == b"\x00\x01\x02\x03"
    """
    # Split the encoded string by the colon delimiter to get individual q64 segments.
    # If the input string is empty, `segments` will be an empty list.
    segments: List[str] = encoded.split(':') if encoded else []

    # If there are any segments, decode the last one. The last segment typically
    # represents the most detailed or full encoding of the original data.
    if segments:
        return q64_decode(segments[-1])  # Decode the last (full) level using the base q64 decoder.
    
    # If no segments are found (e.g., empty input string), return an empty bytes object.
    return b''