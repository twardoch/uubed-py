#!/usr/bin/env python3
# this_file: src/uubed/encoders/q64.py
"""QuadB64: Position-safe base encoding that prevents substring pollution.

This module implements the core QuadB64 (q64) encoding and decoding algorithms.
Q64 is a position-safe base encoding scheme designed to prevent substring pollution.
This means that a given sequence of characters will only match if it appears at a
specific position modulo 4 within the encoded string. This property is crucial
for applications like vector databases where exact substring matching can lead
to false positives, ensuring that a substring match is only valid if it occurs
at the correct positional alignment.

**Key Characteristics:**
- **Position-Safe:** The most distinctive feature of Q64. It uses four different
  alphabets, with the choice of alphabet depending on the character's position
  (0, 1, 2, or 3) modulo 4 within the encoded string. This prevents accidental
  matches of substrings that are not aligned with their original encoding positions.
- **Base-16 Encoding (effectively):** Each byte of input data is represented by two
  characters in the output string. Each character encodes a 4-bit nibble (0-15),
  similar to hexadecimal encoding, but with position-dependent alphabets.
- **Compact:** Provides a relatively compact representation, with a 2:1 character-to-byte ratio.
- **Deterministic:** For a given input, the output Q64 string is always the same.

**Use Cases:**
- Encoding binary data (like embedding vectors) into a text-safe format.
- Storing and transmitting data where substring pollution is a concern.
- Applications requiring a robust, position-aware encoding scheme.

**Limitations:**
- **Not Human-Readable:** While text-safe, the output is not easily human-readable
  due to the varying alphabets.
- **No Compression:** Q64 is an encoding scheme, not a compression algorithm. It does
  not reduce the size of the data; in fact, it doubles the size (2 characters per byte).
"""

from typing import Union, List, Dict, Tuple

# Define the four position-dependent alphabets for Q64 encoding.
# Each alphabet is used based on the position of the character modulo 4.
ALPHABETS: List[str] = [
    "ABCDEFGHIJKLMNOP",  # Used when position % 4 == 0
    "QRSTUVWXYZabcdef",  # Used when position % 4 == 1
    "ghijklmnopqrstuv",  # Used when position % 4 == 2
    "wxyz0123456789-_",  # Used when position % 4 == 3
]

# Pre-compute a reverse lookup table for efficient decoding.
# This maps each character to a tuple: (expected_alphabet_index, nibble_value).
# This allows for O(1) lookup during decoding and helps validate character positions.
REV_LOOKUP: Dict[str, Tuple[int, int]] = {}
for idx, alphabet in enumerate(ALPHABETS):
    for char_idx, char in enumerate(alphabet):
        REV_LOOKUP[char] = (idx, char_idx)


def q64_encode(
    data: Union[bytes, List[int]]
) -> str:
    """
    Encodes a byte sequence or a list of integers into a position-safe q64 string.

    The fundamental principle of Q64 encoding is to use different alphabets for
    characters based on their position within the output string (modulo 4). This
    design prevents "substring pollution," ensuring that a substring match is
    only considered valid if it occurs at the correct positional alignment.
    Each byte of input data is converted into two 4-bit nibbles, and each nibble
    is then encoded using the appropriate alphabet.

    Args:
        data (Union[bytes, List[int]]): The input data to be encoded. It can be either:
                                        - A `bytes` object: A sequence of raw bytes.
                                        - A `List[int]`: A list of integers, where each integer
                                          is expected to be in the range 0-255 (representing a byte).

    Returns:
        str: The position-safe q64 encoded string. The length of the output string
             will be twice the length of the input byte sequence.

    Example:
        >>> from uubed.encoders.q64 import q64_encode
        >>>
        >>> # Encoding a byte string
        >>> q64_encode(b"\x00\x10\x20\x30")
        # Expected: 'AQUgkw'

        >>> # Encoding a list of integers
        >>> q64_encode([0, 16, 32, 48])
        # Expected: 'AQUgkw'

        >>> # Encoding an empty byte string
        >>> q64_encode(b"")
        # Expected: ''
    """
    # Step 1: Ensure the input data is in `bytes` format for consistent processing.
    # If a list of integers is provided, convert it to bytes.
    if isinstance(data, list):
        data = bytes(data)

    result: List[str] = []
    pos: int = 0  # Initialize the current character position in the output string.

    # Step 2: Iterate through each byte in the input data.
    for byte in data:
        # Step 3: Split each byte into two 4-bit nibbles (high and low).
        # A byte (8 bits) can be thought of as two hexadecimal digits, each 4 bits.
        hi_nibble: int = (byte >> 4) & 0xF  # Extract the most significant 4 bits (0-15).
        lo_nibble: int = byte & 0xF        # Extract the least significant 4 bits (0-15).

        # Step 4: Encode each nibble using the position-dependent alphabet.
        # Each nibble will be converted into one character in the output string.
        for nibble in (hi_nibble, lo_nibble):
            # Determine which of the four alphabets to use based on the current
            # character position modulo 4. `pos & 3` is an efficient way to calculate `pos % 4`.
            alphabet: str = ALPHABETS[pos & 3]
            # Append the character corresponding to the nibble value from the selected alphabet.
            result.append(alphabet[nibble])
            pos += 1  # Increment the position for the next character.

    # Step 5: Join the list of characters to form the final q64 encoded string.
    return "".join(result)


def q64_decode(
    encoded: str
) -> bytes:
    """
    Decodes a q64 encoded string back into its original byte sequence.

    This function rigorously validates the positional correctness of each character
    in the encoded string using a pre-computed reverse lookup table (`REV_LOOKUP`).
    It reconstructs the original bytes by combining pairs of 4-bit nibbles obtained
    from the decoded characters.

    Args:
        encoded (str): The q64 encoded string to decode.

    Returns:
        bytes: The original byte sequence.

    Raises:
        ValueError: If the encoded string is malformed, which can include:
                    - Having an odd length (as each byte is represented by two characters).
                    - Containing invalid q64 characters (characters not found in any alphabet).
                    - Containing characters that are in the wrong positional alphabet,
                      violating the position-safe property of Q64.

    Example:
        >>> from uubed.encoders.q64 import q64_decode
        >>>
        >>> # Decoding a valid q64 string
        >>> q64_decode("AQUgkw")
        # Expected: b'\x00\x10\x20\x30'

        >>> # Decoding an empty string
        >>> q64_decode("")
        # Expected: b''

        >>> # Example of an invalid length (will raise ValueError)
        >>> try:
        ...     q64_decode("ABC")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        # Expected: Error: q64 encoded string length must be even (2 characters per byte).

        >>> # Example of an invalid character (will raise ValueError)
        >>> try:
        ...     q64_decode("A!CDEFGH")
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        # Expected: Error: Invalid q64 character '!' found in the encoded string.

        >>> # Example of a character in the wrong position (will raise ValueError)
        >>> try:
        ...     q64_decode("BAAAAAAAAAAAAAAA") # 'B' is from alphabet 0, but at position 1
        ... except ValueError as e:
        ...     print(f"Error: {e}")
        # Expected: Error: Character 'B' is illegal at position 1. It belongs to alphabet 0, but position 1 requires alphabet 1.
    """
    # Step 1: Validate that the encoded string has an even length.
    # Since each byte is represented by two q64 characters, an odd length indicates a malformed string.
    if len(encoded) % 2 != 0:
        raise ValueError("q64 encoded string length must be even (2 characters per byte).")

    nibbles: List[int] = []
    # Step 2: Iterate through each character in the encoded string with its position.
    for pos, char in enumerate(encoded):
        try:
            # Step 3: Look up the character in the pre-computed reverse lookup table (`REV_LOOKUP`).
            # This table provides two pieces of information for each valid q64 character:
            #   - `expected_alphabet_idx`: The index of the alphabet (0-3) to which this character belongs.
            #   - `nibble_value`: The 4-bit integer value that this character represents.
            expected_alphabet_idx, nibble_value = REV_LOOKUP[char]
        except KeyError:
            # If a character is not found in `REV_LOOKUP`, it means it's not a valid q64 character.
            raise ValueError(f"Invalid q64 character {char!r} found in the encoded string.") from None

        # Step 4: Validate the positional correctness of the character.
        # The actual alphabet index for the current position (`pos & 3`) must match
        # the `expected_alphabet_idx` retrieved from the lookup table. This is the core
        # of Q64's position-safety.
        if expected_alphabet_idx != (pos & 3):
            raise ValueError(
                f"Character {char!r} is illegal at position {pos}. "
                f"It belongs to alphabet {expected_alphabet_idx}, but position {pos} requires alphabet {(pos & 3)}."
            )
        nibbles.append(nibble_value)

    # Step 5: Combine pairs of 4-bit nibbles back into 8-bit bytes.
    # The `iter(nibbles)` creates an iterator over the list of nibble values.
    # `zip(iterator, iterator)` effectively pairs consecutive nibbles (high and low).
    # For each pair, the high nibble is shifted left by 4 bits and OR-ed with the low nibble
    # to reconstruct the original byte. The `bytes()` constructor then converts this generator
    # expression into a byte sequence.
    iterator = iter(nibbles)
    return bytes((hi << 4) | lo for hi, lo in zip(iterator, iterator))
