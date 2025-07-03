#!/usr/bin/env python3
# this_file: src/uubed/encoders/eq64.py
"""Eq64: Full embedding encoder with visual dots for readability.

This module implements the Eq64 (Enhanced QuadB64) encoding algorithm.
Eq64 is a variant of QuadB64 (q64) designed for full, lossless encoding of
embedding vectors. Its primary distinguishing feature is the insertion of
dot characters at regular intervals within the encoded string. These dots
serve purely as visual separators to enhance readability and aid in debugging
or manual inspection of encoded embeddings, without affecting the underlying
encoded data or the ability to perfectly reconstruct the original bytes.

**Key Characteristics:**
- **Lossless Encoding:** Ensures that the original byte sequence can be perfectly
  reconstructed from the encoded string without any loss of information.
- **Human Readability:** The periodic insertion of dots breaks up long encoded
  strings into more manageable segments, improving visual parsing and reducing
  the likelihood of errors during manual transcription or inspection.
- **Position-Safe:** Like standard q64, Eq64 maintains position safety, meaning
  that each character in the encoded string corresponds to a specific position
  in the original byte sequence, making it robust to single-character errors.
- **Compatibility:** Built on top of the base q64 encoding, ensuring compatibility
  with existing q64 infrastructure once the dots are removed.

**Use Cases:**
- Storing and transmitting embedding vectors where perfect fidelity is required.
- Debugging and logging of encoded embeddings in human-readable formats.
- Applications where visual inspection of the encoded data is beneficial.

**Limitations:**
- **Increased Length:** The addition of dot characters increases the length of the
  encoded string compared to a pure q64 encoding, which might be a consideration
  for extremely space-constrained environments.
- **No Compression:** Eq64 does not provide any data compression; it is purely
  an encoding scheme for representation.
"""

from .q64 import q64_encode, q64_decode
from typing import Union, List


def eq64_encode(data: Union[bytes, List[int]]) -> str:
    """
    Encodes a byte sequence or a list of integers into an Eq64 string with dot separators.

    This function first converts the input `data` into a standard q64 encoded string.
    Then, for enhanced readability, it inserts a dot (`.`) character every 8 characters
    into the resulting q64 string. This visual segmentation does not affect the
    decodability of the string, as the dots are simply removed during decoding.

    Args:
        data (Union[bytes, List[int]]): The input data to be encoded. It can be either:
                                        - A `bytes` object: A sequence of bytes.
                                        - A `List[int]`: A list of integers, where each integer
                                          is expected to be in the range 0-255 (representing a byte).

    Returns:
        str: The Eq64 encoded string, featuring dot separators for improved readability.
             Example: "ABCDEFGH.IJKLMNOP.QRSTUVWX"

    Example:
        >>> from uubed.encoders.eq64 import eq64_encode
        >>>
        >>> # Encoding a byte string
        >>> encoded_bytes = eq64_encode(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f")
        >>> print(encoded_bytes)
        # Expected: "AAAAAAAABBBBBBBB.CCCCCCCCDDDDDDDD"

        >>> # Encoding a list of integers
        >>> encoded_list = eq64_encode([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        >>> print(encoded_list)
        # Expected: "AAAAAAAABBBBBBBB.CCCCCCCCDDDDDDDD"
    """
    # Step 1: Encode the input data using the base q64 encoder.
    # This converts the bytes or list of integers into a raw q64 string.
    base_encoded: str = q64_encode(data)

    # Step 2: Insert dots into the base_encoded string for visual readability.
    # The dots are inserted every 8 characters. A list of characters is used
    # for efficient string building, then joined at the end.
    result: List[str] = []
    for i, char in enumerate(base_encoded):
        # Insert a dot before every 8th character, but not at the very beginning of the string.
        if i > 0 and i % 8 == 0:
            result.append(".")
        result.append(char)

    # Join the list of characters and dots to form the final Eq64 string.
    return "".join(result)


def eq64_decode(encoded: str) -> bytes:
    """
    Decodes an Eq64 string back into its original byte sequence.

    This function first removes all dot characters that were inserted for readability
    from the `encoded` string. After cleaning, it uses the standard q64 decoder
    to convert the pure q64 string back into the original byte sequence.

    Args:
        encoded (str): The Eq64 encoded string, which may or may not contain dot separators.

    Returns:
        bytes: The original byte sequence that was encoded.

    Example:
        >>> from uubed.encoders.eq64 import eq64_decode
        >>>
        >>> # Decoding a string with dots
        >>> decoded_with_dots = eq64_decode("AAAAAAAABBBBBBBB.CCCCCCCCDDDDDDDD")
        >>> print(decoded_with_dots)
        # Expected: b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'

        >>> # Decoding a string without dots (behaves like standard q64_decode)
        >>> decoded_no_dots = eq64_decode("AAAAAAAABBBBBBBBCCCCCCCCDDDDDDDD")
        >>> print(decoded_no_dots)
        # Expected: b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
    """
    # Step 1: Remove all dot characters from the encoded string.
    # The `str.replace()` method is efficient for this operation, creating a clean q64 string.
    clean_encoded: str = encoded.replace(".", "")

    # Step 2: Decode the cleaned string using the base q64 decoder.
    # This converts the pure q64 string back into the original byte sequence.
    return q64_decode(clean_encoded)
