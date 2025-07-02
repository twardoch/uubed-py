#!/usr/bin/env python3
# this_file: tests/test_encoders.py
"""Test suite for uubed encoders."""

import pytest
import numpy as np
from uubed import encode, decode
from uubed.encoders import q64


class TestQ64:
    """Test the base Q64 codec."""

    def test_encode_decode_roundtrip(self):
        """Test that encode->decode returns original data."""
        data = bytes([0, 127, 255, 42, 100])
        encoded = q64.q64_encode(data)
        decoded = q64.q64_decode(encoded)
        assert decoded == data

    def test_position_safety(self):
        """Test that characters are position-dependent."""
        data1 = bytes([0, 0, 0, 0])
        data2 = bytes([255, 255, 255, 255])

        enc1 = q64.q64_encode(data1)
        enc2 = q64.q64_encode(data2)

        # Check that different positions use different alphabets
        for i in range(len(enc1)):
            alphabet_idx = i & 3
            assert enc1[i] in q64.ALPHABETS[alphabet_idx]
            assert enc2[i] in q64.ALPHABETS[alphabet_idx]

    def test_invalid_decode(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="even"):
            q64.q64_decode("ABC")  # Odd length

        with pytest.raises(ValueError, match="Invalid q64 character"):
            q64.q64_decode("!@")  # Invalid characters

        with pytest.raises(ValueError, match="illegal at position"):
            q64.q64_decode("QA")  # Q is from alphabet 1, but at position 0


class TestHighLevelAPI:
    """Test the high-level encode/decode API."""

    def test_auto_encode(self):
        """Test automatic method selection."""
        small_embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
        large_embedding = np.random.randint(0, 256, 256, dtype=np.uint8)

        # Auto should pick shq64 for small, eq64 for large
        small_result = encode(small_embedding, method="auto")
        assert len(small_result) == 16  # SimHash is compact

        large_result = encode(large_embedding, method="auto")
        assert len(large_result) > 16  # Eq64 is longer than SimHash

    def test_all_methods(self):
        """Test all encoding methods."""
        embedding = list(range(32))

        eq64_result = encode(embedding, method="eq64")
        assert len(eq64_result) == 64  # Native version: 32 bytes = 64 chars

        shq64_result = encode(embedding, method="shq64")
        assert len(shq64_result) == 16  # 64 bits = 8 bytes = 16 chars

        t8q64_result = encode(embedding, method="t8q64", k=8)
        assert len(t8q64_result) == 16  # 8 indices = 16 chars

        zoq64_result = encode(embedding, method="zoq64")
        assert len(zoq64_result) == 8   # 4 bytes = 8 chars

    def test_decode_eq64(self):
        """Test decoding of eq64 format."""
        data = bytes(range(32))
        encoded = encode(data, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == data


class TestLocalityPreservation:
    """Test that similar embeddings produce similar codes."""

    def test_simhash_locality(self):
        """Test SimHash preserves similarity."""
        # Create two similar embeddings
        base = np.random.randint(0, 256, 32, dtype=np.uint8)
        similar = base.copy()
        similar[0] = (int(similar[0]) + 1) % 256  # Small change

        different = 255 - base  # Very different

        # Encode all three
        base_hash = encode(base, method="shq64")
        similar_hash = encode(similar, method="shq64")
        different_hash = encode(different, method="shq64")

        # Count character differences
        similar_diff = sum(a != b for a, b in zip(base_hash, similar_hash))
        different_diff = sum(a != b for a, b in zip(base_hash, different_hash))

        # Similar embeddings should have fewer differences
        assert similar_diff < different_diff

    def test_topk_locality(self):
        """Test Top-k preserves important features."""
        # Create embedding with clear top features
        embedding = np.zeros(256, dtype=np.uint8)
        top_indices = [10, 20, 30, 40, 50, 60, 70, 80]
        for idx in top_indices:
            embedding[idx] = 255

        # Add some noise
        similar = embedding.copy()
        similar[top_indices[0]] = 254  # Slightly reduce one top value
        similar[90] = 100  # Add medium value elsewhere

        # Encode both
        base_topk = encode(embedding, method="t8q64")
        similar_topk = encode(similar, method="t8q64")

        # Should have high overlap
        assert base_topk == similar_topk  # Top indices unchanged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])