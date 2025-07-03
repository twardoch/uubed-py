"""
Tests for comprehensive error handling and validation system.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

# Import our new exception hierarchy and validation functions
from uubed.exceptions import (
    UubedError,
    UubedValidationError,
    UubedEncodingError,
    UubedDecodingError,
    UubedResourceError,
    UubedConnectionError,
    UubedConfigurationError,
)
from uubed.validation import (
    validate_encoding_method,
    validate_embedding_input,
    validate_method_parameters,
    validate_file_path,
    validate_memory_usage,
    estimate_memory_usage,
    validate_gpu_parameters,
)
from uubed import encode, decode


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""
    
    def test_base_exception(self):
        """Test base UubedError exception."""
        error = UubedError(
            "Test message",
            suggestion="Test suggestion",
            error_code="TEST_ERROR",
            context={"param": "value"}
        )
        
        assert "Test message" in str(error)
        assert "Test suggestion" in str(error)
        assert "TEST_ERROR" in str(error)
        assert error.suggestion == "Test suggestion"
        assert error.error_code == "TEST_ERROR"
        assert error.context["param"] == "value"
    
    def test_validation_error(self):
        """Test UubedValidationError with parameter details."""
        error = UubedValidationError(
            "Invalid input",
            parameter="embedding",
            expected="numpy array",
            received="string",
            suggestion="Convert to numpy array"
        )
        
        assert "Invalid input" in str(error)
        assert "embedding" in str(error)
        assert "numpy array" in str(error)
        assert "string" in str(error)
        assert error.context["parameter"] == "embedding"
    
    def test_encoding_error(self):
        """Test UubedEncodingError with method details."""
        error = UubedEncodingError(
            "Encoding failed",
            method="shq64",
            embedding_shape=(128,),
            suggestion="Check parameters"
        )
        
        assert "shq64" in str(error)
        assert "Encoding failed" in str(error)
        assert error.context["method"] == "shq64"
        assert error.context["embedding_shape"] == (128,)


class TestValidationFunctions:
    """Test the validation functions."""
    
    def test_validate_encoding_method(self):
        """Test encoding method validation."""
        # Valid methods
        assert validate_encoding_method("eq64") == "eq64"
        assert validate_encoding_method("EQ64") == "eq64"  # Case insensitive
        assert validate_encoding_method(" shq64 ") == "shq64"  # Strips whitespace
        
        # Invalid methods
        with pytest.raises(UubedValidationError) as exc_info:
            validate_encoding_method("invalid")
        assert "Unknown encoding method" in str(exc_info.value)
        
        # Invalid type
        with pytest.raises(UubedValidationError) as exc_info:
            validate_encoding_method(123)
        assert "must be a string" in str(exc_info.value)
    
    def test_validate_embedding_input_list(self):
        """Test embedding validation with lists."""
        # Valid list
        result = validate_embedding_input([1, 2, 3, 255], "eq64")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [1, 2, 3, 255])
        
        # Empty list
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input([], "eq64")
        assert "cannot be empty" in str(exc_info.value)
        
        # Out of range values
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input([1, 2, 256], "eq64")
        assert "overflow" in str(exc_info.value) or "cannot convert" in str(exc_info.value)
    
    def test_validate_embedding_input_numpy(self):
        """Test embedding validation with numpy arrays."""
        # Valid uint8 array
        arr = np.array([1, 2, 3], dtype=np.uint8)
        result = validate_embedding_input(arr, "eq64")
        np.testing.assert_array_equal(result, arr)
        
        # Float array in [0, 1] range (normalized)
        arr_float = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = validate_embedding_input(arr_float, "eq64")
        expected = np.array([0, 127, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Float array in [0, 255] range
        arr_float = np.array([0.0, 127.0, 255.0], dtype=np.float32)
        result = validate_embedding_input(arr_float, "eq64")
        expected = np.array([0, 127, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
        # Invalid float range
        arr_float = np.array([-1.0, 0.5, 2.0], dtype=np.float32)
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input(arr_float, "eq64")
        assert "range" in str(exc_info.value)
        
        # Empty array
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input(np.array([]), "eq64")
        assert "cannot be empty" in str(exc_info.value)
        
        # None input
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input(None, "eq64")
        assert "cannot be None" in str(exc_info.value)
    
    def test_validate_embedding_input_bytes(self):
        """Test embedding validation with bytes."""
        # Valid bytes
        data = bytes([1, 2, 3, 255])
        result = validate_embedding_input(data, "eq64")
        np.testing.assert_array_equal(result, [1, 2, 3, 255])
        
        # Empty bytes
        with pytest.raises(UubedValidationError) as exc_info:
            validate_embedding_input(b"", "eq64")
        assert "cannot be empty" in str(exc_info.value)
    
    def test_validate_method_parameters(self):
        """Test method-specific parameter validation."""
        # Valid shq64 parameters
        result = validate_method_parameters("shq64", planes=64)
        assert result["planes"] == 64
        
        # Invalid planes (not multiple of 8)
        with pytest.raises(UubedValidationError) as exc_info:
            validate_method_parameters("shq64", planes=63)
        assert "multiple of 8" in str(exc_info.value)
        
        # Invalid planes (too large)
        with pytest.raises(UubedValidationError) as exc_info:
            validate_method_parameters("shq64", planes=2048)
        assert "too large" in str(exc_info.value)
        
        # Valid t8q64 parameters
        result = validate_method_parameters("t8q64", k=8)
        assert result["k"] == 8
        
        # Invalid k (negative)
        with pytest.raises(UubedValidationError) as exc_info:
            validate_method_parameters("t8q64", k=-1)
        assert "must be positive" in str(exc_info.value)
        
        # Unknown parameter
        with pytest.raises(UubedValidationError) as exc_info:
            validate_method_parameters("eq64", unknown_param=123)
        assert "Unknown parameters" in str(exc_info.value)
    
    def test_validate_file_path(self):
        """Test file path validation."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        try:
            # Valid existing file
            result = validate_file_path(tmp_path, check_exists=True, check_readable=True)
            assert isinstance(result, Path)
            assert result.exists()
            
            # Non-existent file
            with pytest.raises(UubedResourceError) as exc_info:
                validate_file_path("/nonexistent/file.txt", check_exists=True)
            assert "does not exist" in str(exc_info.value)
            
            # Invalid type
            with pytest.raises(UubedValidationError) as exc_info:
                validate_file_path(123)
            assert "must be string or Path" in str(exc_info.value)
            
        finally:
            # Clean up
            os.unlink(tmp_path)
    
    def test_memory_validation(self):
        """Test memory usage validation."""
        # Valid memory usage
        validate_memory_usage(1024 * 1024, "test operation")  # 1MB
        
        # Excessive memory usage
        with pytest.raises(UubedResourceError) as exc_info:
            validate_memory_usage(2 * 1024 * 1024 * 1024, "test operation")  # 2GB
        assert "memory usage too high" in str(exc_info.value)
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test different methods
        eq64_memory = estimate_memory_usage(100, 128, "eq64")
        shq64_memory = estimate_memory_usage(100, 128, "shq64")
        
        assert eq64_memory > 0
        assert shq64_memory > 0
        assert eq64_memory != shq64_memory  # Different methods should have different estimates


class TestEnhancedAPIErrorHandling:
    """Test the enhanced error handling in the main API functions."""
    
    def test_encode_validation_errors(self):
        """Test that encode function properly validates inputs."""
        # Invalid method
        with pytest.raises(UubedValidationError) as exc_info:
            encode([1, 2, 3], method="invalid")
        assert "Unknown encoding method" in str(exc_info.value)
        
        # Empty embedding
        with pytest.raises(UubedValidationError) as exc_info:
            encode([], method="eq64")
        assert "cannot be empty" in str(exc_info.value)
        
        # Invalid k parameter for t8q64
        with pytest.raises(UubedValidationError) as exc_info:
            encode([1, 2, 3], method="t8q64", k=10)  # k >= embedding size
        assert "must be smaller than embedding size" in str(exc_info.value)
        
        # Invalid planes parameter for shq64
        with pytest.raises(UubedValidationError) as exc_info:
            encode([1, 2, 3], method="shq64", planes=63)  # not multiple of 8
        assert "multiple of 8" in str(exc_info.value)
    
    def test_decode_validation_errors(self):
        """Test that decode function properly validates inputs."""
        # Invalid input type
        with pytest.raises(UubedValidationError) as exc_info:
            decode(123)
        assert "must be a string" in str(exc_info.value)
        
        # Empty string
        with pytest.raises(UubedValidationError) as exc_info:
            decode("")
        assert "cannot be empty" in str(exc_info.value)
        
        # Unsupported method
        with pytest.raises(UubedDecodingError) as exc_info:
            decode("test", method="shq64")
        assert "not supported for shq64" in str(exc_info.value)
    
    def test_auto_method_selection(self):
        """Test automatic method selection with validation."""
        # Small embedding -> shq64
        small_embedding = np.random.randint(0, 256, 16, dtype=np.uint8)
        result = encode(small_embedding, method="auto")
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Large embedding -> eq64
        large_embedding = np.random.randint(0, 256, 1000, dtype=np.uint8)
        result = encode(large_embedding, method="auto")
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_float_embedding_conversion(self):
        """Test automatic conversion of float embeddings."""
        # Normalized float embedding [0, 1]
        float_embedding = np.random.random(64).astype(np.float32)
        result = encode(float_embedding, method="eq64")
        assert isinstance(result, str)
        
        # Float embedding in [0, 255] range
        float_embedding = np.random.uniform(0, 255, 64).astype(np.float32)
        result = encode(float_embedding, method="eq64")
        assert isinstance(result, str)


class TestGPUValidation:
    """Test GPU parameter validation."""
    
    def test_gpu_validation_no_cupy(self):
        """Test GPU validation when CuPy is not available."""
        # This will only work if CuPy is not installed
        try:
            import cupy
            pytest.skip("CuPy is available, skipping no-CuPy test")
        except ImportError:
            # CuPy not available, test should fail gracefully
            with pytest.raises(UubedResourceError) as exc_info:
                validate_gpu_parameters()
            assert "not available" in str(exc_info.value)
    
    def test_gpu_device_validation(self):
        """Test GPU device ID validation."""
        try:
            import cupy
            # Test invalid device ID type
            with pytest.raises(UubedValidationError) as exc_info:
                validate_gpu_parameters(device_id="invalid")
            assert "must be an integer" in str(exc_info.value)
            
            # Test negative device ID
            with pytest.raises(UubedValidationError) as exc_info:
                validate_gpu_parameters(device_id=-1)
            assert "must be non-negative" in str(exc_info.value)
            
        except ImportError:
            pytest.skip("CuPy not available for GPU tests")


def test_exception_inheritance():
    """Test that all exceptions inherit from UubedError."""
    assert issubclass(UubedValidationError, UubedError)
    assert issubclass(UubedEncodingError, UubedError)
    assert issubclass(UubedDecodingError, UubedError)
    assert issubclass(UubedResourceError, UubedError)
    assert issubclass(UubedConnectionError, UubedError)
    assert issubclass(UubedConfigurationError, UubedError)


def test_error_context_preservation():
    """Test that error context is preserved through the exception chain."""
    try:
        # This should trigger a validation error
        encode("invalid", method="eq64")
    except UubedValidationError as e:
        assert hasattr(e, 'context')
        assert hasattr(e, 'suggestion')
        assert e.error_code == "VALIDATION_ERROR"


if __name__ == "__main__":
    pytest.main([__file__])