#!/usr/bin/env python3
"""Test streaming API functionality."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uubed.api import decode, encode
from uubed.streaming import StreamingEncoder, batch_encode, decode_stream, encode_file_stream, encode_stream


class TestEncodeStream:
    """Test encode_stream functionality."""

    def test_encode_stream_basic(self):
        """Test basic streaming encoding."""
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8)
            for _ in range(10)
        ]

        encoded_list = list(encode_stream(embeddings, method="shq64"))
        assert len(encoded_list) == 10
        assert all(isinstance(enc, str) for enc in encoded_list)

    def test_encode_stream_generator(self):
        """Test streaming from generator."""
        def embedding_generator():
            for i in range(5):
                yield np.random.randint(0, 256, 16, dtype=np.uint8)

        encoded_list = list(encode_stream(embedding_generator(), method="t8q64", k=4))
        assert len(encoded_list) == 5

    def test_encode_stream_batch_size(self):
        """Test streaming with different batch sizes."""
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
            for _ in range(20)
        ]

        # Test with small batch size
        encoded_small = list(encode_stream(embeddings, method="eq64", batch_size=3))

        # Test with large batch size
        encoded_large = list(encode_stream(embeddings, method="eq64", batch_size=10))

        assert len(encoded_small) == len(encoded_large) == 20
        assert encoded_small == encoded_large  # Results should be identical

    def test_encode_stream_empty(self):
        """Test streaming with empty input."""
        encoded_list = list(encode_stream([], method="eq64"))
        assert len(encoded_list) == 0

    def test_encode_stream_different_methods(self):
        """Test streaming with different encoding methods."""
        embeddings = [
            np.random.randint(0, 256, 64, dtype=np.uint8)
            for _ in range(5)
        ]

        methods = ["eq64", "shq64", "t8q64", "zoq64"]
        for method in methods:
            if method == "t8q64":
                encoded_list = list(encode_stream(embeddings, method=method, k=8))
            else:
                encoded_list = list(encode_stream(embeddings, method=method))

            assert len(encoded_list) == 5
            assert all(isinstance(enc, str) for enc in encoded_list)


class TestEncodeFileStream:
    """Test encode_file_stream functionality."""

    def test_encode_file_stream_basic(self):
        """Test basic file streaming."""
        # Create temporary file with embeddings
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for _ in range(10):
                embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
                tmp.write(embedding.tobytes())
            tmp_path = tmp.name

        try:
            # Stream encode from file
            encoded_list = list(encode_file_stream(
                tmp_path,
                method="eq64",
                embedding_size=32
            ))

            assert len(encoded_list) == 10
            assert all(isinstance(enc, str) for enc in encoded_list)

        finally:
            os.unlink(tmp_path)

    def test_encode_file_stream_with_output(self):
        """Test file streaming with output file."""
        # Create input file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            for _ in range(5):
                embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
                tmp_in.write(embedding.tobytes())
            input_path = tmp_in.name

        # Create output file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            # Stream with output file
            encoded_list = list(encode_file_stream(
                input_path,
                output_path,
                method="shq64",
                embedding_size=64
            ))

            assert len(encoded_list) == 5

            # Check output file was written
            assert os.path.exists(output_path)
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 5

        finally:
            os.unlink(input_path)
            os.unlink(output_path)

    def test_encode_file_stream_incomplete_embedding(self):
        """Test handling of incomplete embeddings."""
        # Create file with incomplete last embedding
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write 2 complete embeddings
            for _ in range(2):
                embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
                tmp.write(embedding.tobytes())
            # Write incomplete embedding (only 16 bytes)
            incomplete = np.random.randint(0, 256, 16, dtype=np.uint8)
            tmp.write(incomplete.tobytes())
            tmp_path = tmp.name

        try:
            from uubed.exceptions import UubedResourceError
            with pytest.raises(UubedResourceError, match="not divisible by embedding_size"):
                list(encode_file_stream(tmp_path, embedding_size=32))
        finally:
            os.unlink(tmp_path)


class TestDecodeStream:
    """Test decode_stream functionality."""

    def test_decode_stream_basic(self):
        """Test basic decode streaming."""
        # Create encoded strings
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8)
            for _ in range(5)
        ]
        encoded_strings = [encode(emb, method="eq64") for emb in embeddings]

        # Decode stream
        decoded_list = list(decode_stream(encoded_strings, method="eq64"))

        assert len(decoded_list) == 5
        assert all(isinstance(dec, bytes) for dec in decoded_list)

        # Verify roundtrip
        for original, decoded in zip(embeddings, decoded_list, strict=False):
            assert np.frombuffer(decoded, dtype=np.uint8).tolist() == original.tolist()

    def test_decode_stream_with_newlines(self):
        """Test decode streaming with newline-terminated strings."""
        embeddings = [
            np.random.randint(0, 256, 16, dtype=np.uint8)
            for _ in range(3)
        ]
        encoded_strings = [encode(emb, method="eq64") + "\n" for emb in embeddings]

        decoded_list = list(decode_stream(encoded_strings, method="eq64"))
        assert len(decoded_list) == 3


class TestBatchEncode:
    """Test batch_encode functionality."""

    def test_batch_encode_basic(self):
        """Test basic batch encoding."""
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
            for _ in range(20)
        ]

        # Batch encode
        batch_results = batch_encode(embeddings, method="eq64")

        # Sequential encode for comparison
        sequential_results = [encode(emb, method="eq64") for emb in embeddings]

        assert len(batch_results) == len(sequential_results) == 20
        assert batch_results == sequential_results

    def test_batch_encode_different_methods(self):
        """Test batch encoding with different methods."""
        embeddings = [
            np.random.randint(0, 256, 64, dtype=np.uint8)
            for _ in range(10)
        ]

        methods = ["eq64", "shq64", "t8q64", "zoq64"]
        for method in methods:
            if method == "t8q64":
                results = batch_encode(embeddings, method=method, k=8)
            else:
                results = batch_encode(embeddings, method=method)

            assert len(results) == 10
            assert all(isinstance(r, str) for r in results)

    def test_batch_encode_empty(self):
        """Test batch encoding with empty input."""
        # Empty lists should raise validation error
        from uubed.exceptions import UubedValidationError
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            batch_encode([], method="eq64")


class TestStreamingEncoder:
    """Test StreamingEncoder context manager."""

    def test_streaming_encoder_basic(self):
        """Test basic StreamingEncoder usage."""
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8)
            for _ in range(5)
        ]

        with StreamingEncoder(method="shq64") as encoder:
            encoded_list = []
            for emb in embeddings:
                encoded = encoder.encode(emb)
                encoded_list.append(encoded)
                assert isinstance(encoded, str)

            assert encoder.count == 5

        assert len(encoded_list) == 5

    def test_streaming_encoder_with_file(self):
        """Test StreamingEncoder with file output."""
        embeddings = [
            np.random.randint(0, 256, 16, dtype=np.uint8)
            for _ in range(3)
        ]

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            output_path = tmp.name

        try:
            with StreamingEncoder(output_path, method="t8q64", k=4) as encoder:
                for emb in embeddings:
                    encoder.encode(emb)

            # Check file was written
            with open(output_path) as f:
                lines = f.readlines()
                assert len(lines) == 3
                assert all(line.strip() for line in lines)  # Non-empty lines

        finally:
            os.unlink(output_path)

    def test_streaming_encoder_count(self):
        """Test StreamingEncoder count tracking."""
        with StreamingEncoder(method="eq64") as encoder:
            assert encoder.count == 0

            encoder.encode(np.random.randint(0, 256, 16, dtype=np.uint8))
            assert encoder.count == 1

            encoder.encode(np.random.randint(0, 256, 16, dtype=np.uint8))
            assert encoder.count == 2


class TestStreamingPerformance:
    """Test streaming performance characteristics."""

    def test_memory_efficiency(self):
        """Test that streaming doesn't accumulate memory."""
        # This is a basic test - in practice, memory profiling tools would be better
        def large_embedding_generator():
            for i in range(100):
                yield np.random.randint(0, 256, 1024, dtype=np.uint8)

        # Should not accumulate memory
        encoded_count = 0
        for encoded in encode_stream(large_embedding_generator(), method="shq64"):
            encoded_count += 1
            assert isinstance(encoded, str)

        assert encoded_count == 100

    def test_streaming_vs_batch_consistency(self):
        """Test that streaming and batch produce same results."""
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8)
            for _ in range(10)
        ]

        # Streaming results
        streaming_results = list(encode_stream(embeddings, method="eq64"))

        # Batch results
        batch_results = batch_encode(embeddings, method="eq64")

        # Individual results
        individual_results = [encode(emb, method="eq64") for emb in embeddings]

        assert streaming_results == batch_results == individual_results


if __name__ == "__main__":
    pytest.main([__file__])
