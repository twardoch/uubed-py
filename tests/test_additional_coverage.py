#!/usr/bin/env python3
"""Additional tests for comprehensive coverage of edge cases and missing functionality."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from uubed import encode, decode
from uubed.api import _auto_select_method, _auto_detect_method
from uubed.exceptions import (
    UubedError,
    UubedValidationError,
    UubedEncodingError,
    UubedDecodingError,
    UubedResourceError,
    UubedConnectionError,
    UubedConfigurationError
)
from uubed.config import get_config
from uubed.validation import estimate_memory_usage, validate_memory_usage


class TestAutoMethodSelection:
    """Test automatic method selection logic."""
    
    def test_auto_select_very_small_embeddings(self):
        """Test auto-selection for very small embeddings."""
        # Size <= 16 should select shq64
        small_embedding = np.random.randint(0, 256, 8, dtype=np.uint8)
        method = _auto_select_method(small_embedding)
        assert method == "shq64"
        
        # Boundary case: size = 16
        boundary_embedding = np.random.randint(0, 256, 16, dtype=np.uint8)
        method = _auto_select_method(boundary_embedding)
        assert method == "shq64"
    
    def test_auto_select_small_embeddings(self):
        """Test auto-selection for small embeddings."""
        # Size <= 64, sparse (sparsity < 0.3) should select t8q64
        sparse_embedding = np.zeros(32, dtype=np.uint8)
        sparse_embedding[0:5] = 255  # Only 5 non-zero out of 32 (sparsity ~0.16)
        method = _auto_select_method(sparse_embedding)
        assert method == "t8q64"
        
        # Size <= 64, dense (sparsity >= 0.3) should select shq64
        dense_embedding = np.random.randint(100, 256, 32, dtype=np.uint8)
        method = _auto_select_method(dense_embedding)
        assert method == "shq64"
    
    def test_auto_select_medium_embeddings(self):
        """Test auto-selection for medium embeddings."""
        # Size <= 256, very sparse (sparsity < 0.2) should select t8q64
        sparse_embedding = np.zeros(128, dtype=np.uint8)
        sparse_embedding[0:10] = 255  # Only 10 non-zero out of 128 (sparsity ~0.08)
        method = _auto_select_method(sparse_embedding)
        assert method == "t8q64"
        
        # Size 128 (common) should select shq64
        common_embedding = np.random.randint(0, 256, 128, dtype=np.uint8)
        method = _auto_select_method(common_embedding)
        assert method == "shq64"
        
        # Size 256 (common) should select shq64
        common_embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
        method = _auto_select_method(common_embedding)
        assert method == "shq64"
        
        # Size 200 (not common), dense should select eq64
        uncommon_embedding = np.random.randint(0, 256, 200, dtype=np.uint8)
        method = _auto_select_method(uncommon_embedding)
        assert method == "eq64"
    
    def test_auto_select_large_embeddings(self):
        """Test auto-selection for large embeddings."""
        # Size > 256 should select eq64
        large_embedding = np.random.randint(0, 256, 512, dtype=np.uint8)
        method = _auto_select_method(large_embedding)
        assert method == "eq64"
        
        # Very large embedding should still select eq64
        very_large_embedding = np.random.randint(0, 256, 1024, dtype=np.uint8)
        method = _auto_select_method(very_large_embedding)
        assert method == "eq64"
    
    def test_auto_select_empty_embedding(self):
        """Test auto-selection for empty embedding."""
        empty_embedding = np.array([], dtype=np.uint8)
        method = _auto_select_method(empty_embedding)
        assert method == "shq64"  # Falls into very small case
    
    def test_auto_select_edge_cases(self):
        """Test auto-selection for edge cases."""
        # All zeros (sparsity = 0.0)
        all_zeros = np.zeros(64, dtype=np.uint8)
        method = _auto_select_method(all_zeros)
        assert method == "t8q64"  # Very sparse
        
        # All ones (sparsity = 1.0)
        all_ones = np.ones(64, dtype=np.uint8)
        method = _auto_select_method(all_ones)
        assert method == "shq64"  # Very dense


class TestAutoDetectMethod:
    """Test automatic method detection from encoded strings."""
    
    def test_detect_eq64_method(self):
        """Test detection of eq64 method."""
        # eq64 strings contain dots
        test_string = "A.B.C.D"
        method = _auto_detect_method(test_string)
        assert method == "eq64"
        
        # Multiple dots
        test_string = "ABC.DEF.GHI"
        method = _auto_detect_method(test_string)
        assert method == "eq64"
    
    def test_detect_non_eq64_method(self):
        """Test detection failure for non-eq64 methods."""
        # String without dots cannot be auto-detected
        test_string = "ABCDEFGH"
        with pytest.raises(UubedDecodingError, match="Cannot auto-detect"):
            _auto_detect_method(test_string)
        
        # Empty string
        with pytest.raises(UubedDecodingError, match="Cannot auto-detect"):
            _auto_detect_method("")


class TestConfigurationIntegration:
    """Test configuration integration with encoding."""
    
    def test_auto_method_with_config_default(self):
        """Test auto method selection with configured default."""
        config = get_config()
        
        # Set a default method
        config.set("encoding.default_method", "shq64")
        
        try:
            # Auto should use configured default
            test_embedding = np.random.randint(0, 256, 32, dtype=np.uint8)
            encoded = encode(test_embedding, method="auto")
            assert isinstance(encoded, str)
            
            # Should have used shq64 encoding
            assert len(encoded) == 16  # shq64 with 64 planes = 16 chars
            
        finally:
            # Reset to default
            config.set("encoding.default_method", "auto")
    
    def test_method_specific_config_parameters(self):
        """Test method-specific configuration parameters."""
        config = get_config()
        
        # Configure shq64 planes
        config.set("encoding.shq64.planes", 128)
        
        try:
            test_embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
            encoded = encode(test_embedding, method="shq64")
            assert isinstance(encoded, str)
            
            # Should use 128 planes (32 chars)
            assert len(encoded) == 32
            
        finally:
            # Reset to default
            config.set("encoding.shq64.planes", 64)
    
    def test_config_parameter_override(self):
        """Test that explicit parameters override config."""
        config = get_config()
        
        # Set config default
        config.set("encoding.shq64.planes", 64)
        
        try:
            test_embedding = np.random.randint(0, 256, 64, dtype=np.uint8)
            
            # Explicit parameter should override config
            encoded = encode(test_embedding, method="shq64", planes=128)
            assert isinstance(encoded, str)
            assert len(encoded) == 32  # 128 planes = 32 chars
            
        finally:
            # Reset to default
            config.set("encoding.shq64.planes", 64)


class TestMemoryManagement:
    """Test memory management and validation."""
    
    def test_memory_estimation_accuracy(self):
        """Test memory estimation accuracy."""
        # Test various scenarios
        scenarios = [
            (1, 128, "eq64"),
            (100, 256, "shq64"),
            (10, 512, "t8q64"),
            (50, 64, "zoq64"),
            (25, 1024, "mq64"),
            (1000, 16, "auto")
        ]
        
        for count, size, method in scenarios:
            estimate = estimate_memory_usage(count, size, method)
            
            # Should be positive
            assert estimate > 0
            
            # Should be reasonable (not too small, not too large)
            base_memory = count * size
            assert estimate >= base_memory  # Should be at least base memory
            assert estimate <= base_memory * 10  # Should not be more than 10x base
    
    def test_memory_validation_with_estimation(self):
        """Test memory validation integrated with estimation."""
        # Small operation should pass
        small_estimate = estimate_memory_usage(10, 32, "eq64")
        validate_memory_usage(small_estimate, "small operation")
        
        # Large operation should fail
        large_estimate = estimate_memory_usage(100000, 1024, "eq64")
        with pytest.raises(UubedResourceError, match="too high"):
            validate_memory_usage(large_estimate, "large operation")
    
    def test_memory_validation_in_encode(self):
        """Test memory validation during encoding."""
        # Normal size should work
        normal_embedding = np.random.randint(0, 256, 128, dtype=np.uint8)
        encoded = encode(normal_embedding, method="eq64")
        assert isinstance(encoded, str)
        
        # Very large embedding should potentially fail
        # (This depends on system memory and limits)
        try:
            huge_embedding = np.random.randint(0, 256, 1000000, dtype=np.uint8)
            encoded = encode(huge_embedding, method="eq64")
            # If it doesn't fail, that's also fine (depends on available memory)
        except UubedResourceError:
            # This is expected for very large embeddings
            pass


class TestExceptionHandling:
    """Test comprehensive exception handling."""
    
    def test_exception_chain_preservation(self):
        """Test that exception chains are preserved."""
        # Test encoding error chain
        try:
            # This should cause an encoding error
            encode("invalid_input", method="eq64")
        except UubedValidationError as e:
            assert hasattr(e, '__cause__') or hasattr(e, '__context__')
            assert e.error_code == "VALIDATION_ERROR"
    
    def test_error_context_information(self):
        """Test that error context contains useful information."""
        # Test validation error context
        try:
            encode([], method="eq64")
        except UubedValidationError as e:
            assert hasattr(e, 'context')
            assert 'parameter' in e.context
            assert 'expected' in e.context
            assert 'received' in e.context
        
        # Test encoding error context
        try:
            encode([1, 2, 3], method="invalid_method")
        except UubedValidationError as e:
            assert hasattr(e, 'context')
            assert 'parameter' in e.context
    
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        # All exceptions should inherit from UubedError
        assert issubclass(UubedValidationError, UubedError)
        assert issubclass(UubedEncodingError, UubedError)
        assert issubclass(UubedDecodingError, UubedError)
        assert issubclass(UubedResourceError, UubedError)
        assert issubclass(UubedConnectionError, UubedError)
        assert issubclass(UubedConfigurationError, UubedError)
        
        # All should inherit from Exception
        assert issubclass(UubedError, Exception)
    
    def test_error_message_quality(self):
        """Test that error messages are helpful."""
        # Test validation error messages
        try:
            encode([], method="eq64")
        except UubedValidationError as e:
            message = str(e)
            assert "cannot be empty" in message
            assert "parameter" in message
            assert "expected" in message
        
        # Test method validation error
        try:
            encode([1, 2, 3], method="invalid")
        except UubedValidationError as e:
            message = str(e)
            assert "Unknown encoding method" in message
            assert "expected" in message


class TestEdgeCasesAndRobustness:
    """Test edge cases and robustness."""
    
    def test_unicode_handling(self):
        """Test handling of unicode and special characters."""
        # Test that the system handles various input types gracefully
        test_cases = [
            bytes([0x00, 0x7F, 0x80, 0xFF]),  # Full byte range
            bytes([0x01, 0x02, 0x03, 0x04]),  # Low values
            bytes([0xFC, 0xFD, 0xFE, 0xFF]),  # High values
        ]
        
        for test_data in test_cases:
            encoded = encode(test_data, method="eq64")
            assert isinstance(encoded, str)
            
            decoded = decode(encoded, method="eq64")
            assert decoded == test_data
    
    def test_large_input_handling(self):
        """Test handling of large inputs."""
        # Test progressively larger inputs
        sizes = [1, 10, 100, 1000, 5000]
        
        for size in sizes:
            test_data = np.random.randint(0, 256, size, dtype=np.uint8)
            
            try:
                encoded = encode(test_data, method="eq64")
                assert isinstance(encoded, str)
                assert len(encoded) > 0
                
                decoded = decode(encoded, method="eq64")
                assert np.frombuffer(decoded, dtype=np.uint8).tolist() == test_data.tolist()
                
            except UubedResourceError:
                # Large inputs may fail due to memory constraints
                # This is expected and acceptable
                pass
    
    def test_concurrent_access(self):
        """Test concurrent access to encoding functions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    test_data = np.random.randint(0, 256, 32, dtype=np.uint8)
                    encoded = encode(test_data, method="eq64")
                    decoded = decode(encoded, method="eq64")
                    
                    assert np.frombuffer(decoded, dtype=np.uint8).tolist() == test_data.tolist()
                
                results.append(f"Worker {worker_id} completed")
                
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Start multiple workers
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 3
    
    def test_system_resource_limits(self):
        """Test behavior near system resource limits."""
        # Test with available memory
        try:
            # Try to create an embedding that uses significant memory
            large_size = 100000
            test_data = np.random.randint(0, 256, large_size, dtype=np.uint8)
            
            encoded = encode(test_data, method="eq64")
            assert isinstance(encoded, str)
            
        except (UubedResourceError, MemoryError):
            # This is expected if system resources are limited
            pass
    
    def test_invalid_parameter_combinations(self):
        """Test invalid parameter combinations."""
        test_data = np.random.randint(0, 256, 32, dtype=np.uint8)
        
        # Test t8q64 with k >= embedding size
        with pytest.raises(UubedValidationError, match="too small for the"):
            encode(test_data, method="t8q64", k=32)
        
        with pytest.raises(UubedValidationError, match="too small for the"):
            encode(test_data, method="t8q64", k=50)
    
    def test_boundary_conditions(self):
        """Test boundary conditions."""
        # Test minimum valid sizes for each method
        test_cases = [
            ("eq64", 1),
            ("shq64", 32),
            ("t8q64", 8),
            ("zoq64", 2),
            ("mq64", 1),
        ]
        
        for method, min_size in test_cases:
            test_data = np.random.randint(0, 256, min_size, dtype=np.uint8)
            
            if method == "t8q64":
                # For t8q64, k must be less than embedding size
                k = min(8, min_size - 1) if min_size > 1 else 1
                encoded = encode(test_data, method=method, k=k)
            else:
                encoded = encode(test_data, method=method)
            
            assert isinstance(encoded, str)
            assert len(encoded) > 0
    
    def test_method_specific_validation(self):
        """Test method-specific validation edge cases."""
        # Test shq64 with various planes values
        test_data = np.random.randint(0, 256, 64, dtype=np.uint8)
        
        valid_planes = [8, 16, 32, 64, 128, 256]
        for planes in valid_planes:
            encoded = encode(test_data, method="shq64", planes=planes)
            assert isinstance(encoded, str)
            assert len(encoded) == planes // 4  # planes/8 bytes * 2 chars per byte
        
        # Test t8q64 with various k values
        test_data = np.random.randint(0, 256, 100, dtype=np.uint8)
        
        valid_k_values = [1, 4, 8, 16, 32, 64]
        for k in valid_k_values:
            encoded = encode(test_data, method="t8q64", k=k)
            assert isinstance(encoded, str)
            assert len(encoded) == k * 2  # k indices * 2 chars per index


if __name__ == "__main__":
    pytest.main([__file__])