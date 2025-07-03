#!/usr/bin/env python3
"""Test core API functionality."""

import pytest
import numpy as np
from uubed.api import encode, decode, EncodingMethod
from uubed.exceptions import UubedValidationError, UubedDecodingError


class TestEncodeDecode:
    """Test encoding and decoding functionality."""
    
    def test_encode_bytes(self):
        """Test encoding from bytes."""
        data = b"Hello, World!"
        encoded = encode(data, method="eq64")
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_encode_list(self):
        """Test encoding from list of integers."""
        data = [72, 101, 108, 108, 111]  # "Hello"
        encoded = encode(data, method="eq64")
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_encode_numpy_array(self):
        """Test encoding from numpy array."""
        data = np.array([72, 101, 108, 108, 111], dtype=np.uint8)
        encoded = encode(data, method="eq64")
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding produces original data."""
        original = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
        encoded = encode(original, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == original
    
    def test_all_encoding_methods(self):
        """Test all encoding methods work."""
        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        methods = ["eq64", "shq64", "t8q64", "zoq64"]
        for method in methods:
            if method == "t8q64":
                encoded = encode(data, method=method, k=8)
            elif method == "shq64":
                encoded = encode(data, method=method, planes=64)
            else:
                encoded = encode(data, method=method)
            
            assert isinstance(encoded, str)
            assert len(encoded) > 0
    
    def test_auto_method_selection(self):
        """Test automatic method selection."""
        small_data = np.random.randint(0, 256, 16, dtype=np.uint8)
        large_data = np.random.randint(0, 256, 128, dtype=np.uint8)
        
        small_encoded = encode(small_data, method="auto")
        large_encoded = encode(large_data, method="auto")
        
        assert isinstance(small_encoded, str)
        assert isinstance(large_encoded, str)
    
    def test_invalid_input_values(self):
        """Test validation of input values."""
        with pytest.raises(UubedValidationError, match="Values must be in range 0-255"):
            encode([0, 100, 300, 50], method="eq64")
    
    def test_decode_invalid_method(self):
        """Test decoding with lossy methods raises error."""
        data = np.random.randint(0, 256, 32, dtype=np.uint8)
        encoded = encode(data, method="shq64")
        
        with pytest.raises(UubedDecodingError, match="not supported for.*shq64"):
            decode(encoded, method="shq64")
    
    def test_method_specific_parameters(self):
        """Test method-specific parameters."""
        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        # Test t8q64 with different k values
        encoded_k4 = encode(data, method="t8q64", k=4)
        encoded_k8 = encode(data, method="t8q64", k=8)
        # k=4 produces 4 indices (8 chars), k=8 produces 8 indices (16 chars)
        assert len(encoded_k4) == 8  # 4 indices * 2 chars per index
        assert len(encoded_k8) == 16  # 8 indices * 2 chars per index
        
        # Test shq64 with different planes
        encoded_p32 = encode(data, method="shq64", planes=32)
        encoded_p64 = encode(data, method="shq64", planes=64)
        # planes=32 produces 32 bits (4 bytes = 8 chars), planes=64 produces 64 bits (8 bytes = 16 chars)
        assert len(encoded_p32) == 8   # 32 planes = 4 bytes = 8 chars
        assert len(encoded_p64) == 16  # 64 planes = 8 bytes = 16 chars


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            encode([], method="eq64")
    
    def test_invalid_method(self):
        """Test handling of invalid encoding method."""
        data = [1, 2, 3, 4, 5]
        with pytest.raises(UubedValidationError, match="Unknown encoding method"):
            encode(data, method="invalid_method")
    
    def test_decode_auto_detection(self):
        """Test automatic method detection in decode."""
        data = np.random.randint(0, 256, 32, dtype=np.uint8)
        encoded = encode(data, method="eq64")
        
        # Auto-detection should fail since eq64 doesn't have reliable pattern markers
        with pytest.raises(UubedDecodingError, match="Cannot auto-detect"):
            decode(encoded)
        
        # But explicit method should work
        decoded = decode(encoded, method="eq64")
        assert np.frombuffer(decoded, dtype=np.uint8).tolist() == data.tolist()
    
    def test_decode_auto_detection_failure(self):
        """Test decode fails when method cannot be auto-detected."""
        with pytest.raises(UubedDecodingError, match="Cannot auto-detect"):
            decode("SomeRandomString")


class TestDataTypes:
    """Test handling of different data types."""
    
    def test_numpy_float_to_uint8(self):
        """Test conversion from float arrays."""
        # Test normalized float data [0, 1] - validation requires this range
        float_data = np.random.rand(32)  # Changed from randn to rand for [0,1] range
        encoded = encode(float_data, method="eq64")
        assert isinstance(encoded, str)
    
    def test_numpy_different_dtypes(self):
        """Test different numpy dtypes."""
        data_uint8 = np.random.randint(0, 256, 32, dtype=np.uint8)
        data_int32 = np.random.randint(0, 256, 32, dtype=np.int32)
        data_float32 = np.random.randint(0, 256, 32).astype(np.float32)
        
        encoded_uint8 = encode(data_uint8, method="eq64")
        encoded_int32 = encode(data_int32, method="eq64")
        encoded_float32 = encode(data_float32, method="eq64")
        
        assert isinstance(encoded_uint8, str)
        assert isinstance(encoded_int32, str)
        assert isinstance(encoded_float32, str)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_byte(self):
        """Test encoding single byte."""
        encoded = encode([128], method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == bytes([128])
    
    def test_all_zeros(self):
        """Test encoding all zeros."""
        data = [0] * 32
        encoded = encode(data, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == bytes(data)
    
    def test_all_max_values(self):
        """Test encoding all max values."""
        data = [255] * 32
        encoded = encode(data, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert decoded == bytes(data)
    
    def test_large_embedding(self):
        """Test encoding large embedding."""
        data = np.random.randint(0, 256, 2048, dtype=np.uint8)
        encoded = encode(data, method="eq64")
        decoded = decode(encoded, method="eq64")
        assert np.frombuffer(decoded, dtype=np.uint8).tolist() == data.tolist()


class TestConsistency:
    """Test consistency across multiple runs."""
    
    def test_deterministic_encoding(self):
        """Test that encoding is deterministic."""
        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        # eq64 should be deterministic
        encoded1 = encode(data, method="eq64")
        encoded2 = encode(data, method="eq64")
        assert encoded1 == encoded2
        
        # shq64 should be deterministic (uses fixed seed)
        encoded1 = encode(data, method="shq64")
        encoded2 = encode(data, method="shq64")
        assert encoded1 == encoded2
    
    def test_method_independence(self):
        """Test that different methods produce different results."""
        data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        eq64_encoded = encode(data, method="eq64")
        shq64_encoded = encode(data, method="shq64")
        t8q64_encoded = encode(data, method="t8q64")
        zoq64_encoded = encode(data, method="zoq64")
        
        # All should be different (with high probability)
        encodings = [eq64_encoded, shq64_encoded, t8q64_encoded, zoq64_encoded]
        assert len(set(encodings)) == len(encodings)


if __name__ == "__main__":
    pytest.main([__file__])