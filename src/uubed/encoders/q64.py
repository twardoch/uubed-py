#!/usr/bin/env python3
# this_file: src/uubed/encoders/q64.py
"""QuadB64: Position-safe base encoding that prevents substring pollution."""

from typing import Union, List

# Position-dependent alphabets
ALPHABETS = [
    "ABCDEFGHIJKLMNOP",  # pos ≡ 0 (mod 4)
    "QRSTUVWXYZabcdef",  # pos ≡ 1
    "ghijklmnopqrstuv",  # pos ≡ 2
    "wxyz0123456789-_",  # pos ≡ 3
]

# Pre-compute reverse lookup for O(1) decode
REV_LOOKUP = {}
for idx, alphabet in enumerate(ALPHABETS):
    for char_idx, char in enumerate(alphabet):
        REV_LOOKUP[char] = (idx, char_idx)


def q64_encode(data: Union[bytes, List[int]]) -> str:
    """
    Encode bytes into q64 positional alphabet.

    Why this matters: Regular base64 allows "abc" to match anywhere.
    Q64 ensures "abc" can only match at specific positions, eliminating
    false positives in substring searches.

    Args:
        data: Bytes or list of integers to encode

    Returns:
        Position-safe encoded string (2 chars per byte)
    """
    if isinstance(data, list):
        data = bytes(data)

    result = []
    pos = 0

    for byte in data:
        # Split byte into two 4-bit nibbles
        hi_nibble = (byte >> 4) & 0xF
        lo_nibble = byte & 0xF

        # Encode each nibble with position-dependent alphabet
        for nibble in (hi_nibble, lo_nibble):
            alphabet = ALPHABETS[pos & 3]  # pos mod 4
            result.append(alphabet[nibble])
            pos += 1

    return "".join(result)


def q64_decode(encoded: str) -> bytes:
    """
    Decode q64 string back to bytes.

    Args:
        encoded: Q64 encoded string

    Returns:
        Original bytes

    Raises:
        ValueError: If string is malformed
    """
    if len(encoded) & 1:
        raise ValueError("q64 length must be even (2 chars per byte)")

    nibbles = []
    for pos, char in enumerate(encoded):
        try:
            expected_alphabet_idx, nibble_value = REV_LOOKUP[char]
        except KeyError:
            raise ValueError(f"Invalid q64 character {char!r}") from None

        if expected_alphabet_idx != (pos & 3):
            raise ValueError(
                f"Character {char!r} illegal at position {pos}. "
                f"Expected alphabet {expected_alphabet_idx}"
            )
        nibbles.append(nibble_value)

    # Combine nibbles back into bytes
    iterator = iter(nibbles)
    return bytes((hi << 4) | lo for hi, lo in zip(iterator, iterator))