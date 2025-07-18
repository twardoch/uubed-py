#!/usr/bin/env python3
"""Test validation functions and edge cases."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uubed.exceptions import UubedResourceError, UubedValidationError
from uubed.validation import (
    _validate_embedding_dimensions,
    estimate_memory_usage,
    validate_batch_parameters,
    validate_embedding_input,
    validate_encoding_method,
    validate_file_path,
    validate_gpu_parameters,
    validate_memory_usage,
    validate_method_parameters,
)


class TestValidateEncodingMethod:
    """Test encoding method validation."""

    def test_valid_methods(self):
        """Test validation of valid encoding methods."""
        valid_methods = ["eq64", "shq64", "t8q64", "zoq64", "mq64", "auto"]

        for method in valid_methods:
            assert validate_encoding_method(method) == method

    def test_case_insensitive(self):
        """Test case insensitive validation."""
        assert validate_encoding_method("EQ64") == "eq64"
        assert validate_encoding_method("SHQ64") == "shq64"
        assert validate_encoding_method("AUTO") == "auto"

    def test_whitespace_stripping(self):
        """Test whitespace stripping."""
        assert validate_encoding_method("  eq64  ") == "eq64"
        assert validate_encoding_method("\tshq64\n") == "shq64"

    def test_invalid_method(self):
        """Test invalid encoding method."""
        with pytest.raises(UubedValidationError, match="Unknown encoding method"):
            validate_encoding_method("invalid_method")

    def test_non_string_input(self):
        """Test non-string input."""
        with pytest.raises(UubedValidationError, match="must be a string"):
            validate_encoding_method(123)

        with pytest.raises(UubedValidationError, match="must be a string"):
            validate_encoding_method(None)

        with pytest.raises(UubedValidationError, match="must be a string"):
            validate_encoding_method([])


class TestValidateEmbeddingInput:
    """Test embedding input validation."""

    def test_valid_list_input(self):
        """Test valid list input."""
        input_list = [0, 50, 100, 150, 255]
        result = validate_embedding_input(input_list, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, input_list)

    def test_valid_bytes_input(self):
        """Test valid bytes input."""
        input_bytes = bytes([0, 50, 100, 150, 255])
        result = validate_embedding_input(input_bytes, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 50, 100, 150, 255])

    def test_valid_numpy_uint8(self):
        """Test valid numpy uint8 array."""
        input_array = np.array([0, 50, 100, 150, 255], dtype=np.uint8)
        result = validate_embedding_input(input_array, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, input_array)

    def test_valid_numpy_int(self):
        """Test valid numpy int array."""
        input_array = np.array([0, 50, 100, 150, 255], dtype=np.int32)
        result = validate_embedding_input(input_array, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 50, 100, 150, 255])

    def test_valid_numpy_float_normalized(self):
        """Test valid numpy float array (normalized 0-1)."""
        input_array = np.array([0.0, 0.2, 0.4, 0.6, 1.0], dtype=np.float32)
        result = validate_embedding_input(input_array, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        expected = np.array([0, 51, 102, 153, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_valid_numpy_float_0_255(self):
        """Test valid numpy float array (0-255 range)."""
        input_array = np.array([0.0, 50.0, 100.0, 150.0, 255.0], dtype=np.float32)
        result = validate_embedding_input(input_array, "eq64")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 50, 100, 150, 255])

    def test_none_input(self):
        """Test None input."""
        with pytest.raises(UubedValidationError, match="cannot be None"):
            validate_embedding_input(None, "eq64")

    def test_empty_list(self):
        """Test empty list input."""
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            validate_embedding_input([], "eq64")

    def test_empty_bytes(self):
        """Test empty bytes input."""
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            validate_embedding_input(b"", "eq64")

    def test_empty_numpy_array(self):
        """Test empty numpy array input."""
        with pytest.raises(UubedValidationError, match="cannot be empty"):
            validate_embedding_input(np.array([]), "eq64")

    def test_out_of_range_list(self):
        """Test list with out of range values."""
        with pytest.raises(UubedValidationError, match="overflow"):
            validate_embedding_input([0, 100, 256], "eq64")

        with pytest.raises(UubedValidationError, match="overflow"):
            validate_embedding_input([-1, 100, 200], "eq64")

    def test_out_of_range_int_array(self):
        """Test int array with out of range values."""
        input_array = np.array([0, 100, 256], dtype=np.int32)
        with pytest.raises(UubedValidationError, match="must be in range"):
            validate_embedding_input(input_array, "eq64")

        input_array = np.array([-1, 100, 200], dtype=np.int32)
        with pytest.raises(UubedValidationError, match="must be in range"):
            validate_embedding_input(input_array, "eq64")

    def test_out_of_range_float_array(self):
        """Test float array with out of range values."""
        input_array = np.array([0.0, 0.5, 1.5], dtype=np.float32)
        with pytest.raises(UubedValidationError, match="range"):
            validate_embedding_input(input_array, "eq64")

        input_array = np.array([-0.5, 0.5, 1.0], dtype=np.float32)
        with pytest.raises(UubedValidationError, match="range"):
            validate_embedding_input(input_array, "eq64")

    def test_unsupported_dtype(self):
        """Test unsupported numpy dtype."""
        input_array = np.array([0.0, 0.5, 1.0], dtype=np.complex64)
        with pytest.raises(UubedValidationError, match="Unsupported embedding dtype"):
            validate_embedding_input(input_array, "eq64")

    def test_unsupported_type(self):
        """Test unsupported input type."""
        with pytest.raises(UubedValidationError, match="Unsupported embedding type"):
            validate_embedding_input("string", "eq64")

        with pytest.raises(UubedValidationError, match="Unsupported embedding type"):
            validate_embedding_input({"key": "value"}, "eq64")


class TestValidateEmbeddingDimensions:
    """Test embedding dimension validation."""

    def test_eq64_dimensions(self):
        """Test eq64 dimension validation."""
        # Valid sizes
        _validate_embedding_dimensions(np.array([1], dtype=np.uint8), "eq64")
        _validate_embedding_dimensions(np.array(range(100), dtype=np.uint8), "eq64")

        # Invalid size (too large)
        with pytest.raises(UubedValidationError, match="too large"):
            _validate_embedding_dimensions(np.array(range(100001), dtype=np.uint8), "eq64")

    def test_shq64_dimensions(self):
        """Test shq64 dimension validation."""
        # Valid sizes
        _validate_embedding_dimensions(np.array(range(32), dtype=np.uint8), "shq64")
        _validate_embedding_dimensions(np.array(range(128), dtype=np.uint8), "shq64")

        # Invalid size (too small)
        with pytest.raises(UubedValidationError, match="too small"):
            _validate_embedding_dimensions(np.array(range(16), dtype=np.uint8), "shq64")

        # Invalid size (too large)
        with pytest.raises(UubedValidationError, match="too large"):
            _validate_embedding_dimensions(np.array(range(50001), dtype=np.uint8), "shq64")

    def test_t8q64_dimensions(self):
        """Test t8q64 dimension validation."""
        # Valid sizes
        _validate_embedding_dimensions(np.array(range(8), dtype=np.uint8), "t8q64")
        _validate_embedding_dimensions(np.array(range(100), dtype=np.uint8), "t8q64")

        # Invalid size (too small)
        with pytest.raises(UubedValidationError, match="too small"):
            _validate_embedding_dimensions(np.array(range(4), dtype=np.uint8), "t8q64")

    def test_zoq64_dimensions(self):
        """Test zoq64 dimension validation."""
        # Valid sizes
        _validate_embedding_dimensions(np.array(range(2), dtype=np.uint8), "zoq64")
        _validate_embedding_dimensions(np.array(range(64), dtype=np.uint8), "zoq64")

        # Invalid size (too small)
        with pytest.raises(UubedValidationError, match="too small"):
            _validate_embedding_dimensions(np.array([1], dtype=np.uint8), "zoq64")

    def test_mq64_dimensions(self):
        """Test mq64 dimension validation."""
        # Valid sizes
        _validate_embedding_dimensions(np.array([1], dtype=np.uint8), "mq64")
        _validate_embedding_dimensions(np.array(range(1000), dtype=np.uint8), "mq64")

        # Invalid size (too large)
        with pytest.raises(UubedValidationError, match="too large"):
            _validate_embedding_dimensions(np.array(range(100001), dtype=np.uint8), "mq64")

    def test_auto_method_skips_validation(self):
        """Test that auto method skips dimension validation."""
        # Should not raise error for any size
        _validate_embedding_dimensions(np.array([1], dtype=np.uint8), "auto")
        _validate_embedding_dimensions(np.array([], dtype=np.uint8), "auto")


class TestValidateMethodParameters:
    """Test method parameter validation."""

    def test_shq64_parameters(self):
        """Test shq64 parameter validation."""
        # Valid parameters
        result = validate_method_parameters("shq64", planes=64)
        assert result["planes"] == 64

        result = validate_method_parameters("shq64", planes=128)
        assert result["planes"] == 128

        # Default value
        result = validate_method_parameters("shq64")
        assert result["planes"] == 64

        # Invalid type
        with pytest.raises(UubedValidationError, match="must be an integer"):
            validate_method_parameters("shq64", planes="64")

        # Invalid value (not positive)
        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_method_parameters("shq64", planes=0)

        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_method_parameters("shq64", planes=-8)

        # Invalid value (not multiple of 8)
        with pytest.raises(UubedValidationError, match="multiple of 8"):
            validate_method_parameters("shq64", planes=63)

        # Invalid value (too large)
        with pytest.raises(UubedValidationError, match="too large"):
            validate_method_parameters("shq64", planes=2048)

    def test_t8q64_parameters(self):
        """Test t8q64 parameter validation."""
        # Valid parameters
        result = validate_method_parameters("t8q64", k=8)
        assert result["k"] == 8

        result = validate_method_parameters("t8q64", k=16)
        assert result["k"] == 16

        # Default value
        result = validate_method_parameters("t8q64")
        assert result["k"] == 8

        # Invalid type
        with pytest.raises(UubedValidationError, match="must be an integer"):
            validate_method_parameters("t8q64", k="8")

        # Invalid value (not positive)
        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_method_parameters("t8q64", k=0)

        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_method_parameters("t8q64", k=-1)

        # Invalid value (too large)
        with pytest.raises(UubedValidationError, match="too large"):
            validate_method_parameters("t8q64", k=2000)

    def test_mq64_parameters(self):
        """Test mq64 parameter validation."""
        # Valid parameters
        result = validate_method_parameters("mq64", levels=[64, 128, 256])
        assert result["levels"] == [64, 128, 256]

        # Default value (None)
        result = validate_method_parameters("mq64")
        assert result["levels"] is None

        # Invalid type
        with pytest.raises(UubedValidationError, match="must be a list"):
            validate_method_parameters("mq64", levels="[64, 128]")

        # Invalid list content
        with pytest.raises(UubedValidationError, match="must be a list"):
            validate_method_parameters("mq64", levels=[64, "128", 256])

        with pytest.raises(UubedValidationError, match="must be a list"):
            validate_method_parameters("mq64", levels=[64, -128, 256])

        # Invalid order
        with pytest.raises(UubedValidationError, match="must be sorted"):
            validate_method_parameters("mq64", levels=[128, 64, 256])

    def test_no_parameters_methods(self):
        """Test methods with no parameters."""
        # eq64 should accept no parameters
        result = validate_method_parameters("eq64")
        assert result == {}

        # zoq64 should accept no parameters
        result = validate_method_parameters("zoq64")
        assert result == {}

        # auto should accept no parameters
        result = validate_method_parameters("auto")
        assert result == {}

    def test_unknown_parameters(self):
        """Test unknown parameter rejection."""
        with pytest.raises(UubedValidationError, match="Unknown parameters"):
            validate_method_parameters("eq64", unknown_param=123)

        with pytest.raises(UubedValidationError, match="Unknown parameters"):
            validate_method_parameters("shq64", planes=64, unknown_param=123)


class TestValidateBatchParameters:
    """Test batch parameter validation."""

    def test_valid_batch_size(self):
        """Test valid batch size."""
        result = validate_batch_parameters(batch_size=100)
        assert result["batch_size"] == 100

        result = validate_batch_parameters(batch_size=1000)
        assert result["batch_size"] == 1000

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        # Non-integer
        with pytest.raises(UubedValidationError, match="must be an integer"):
            validate_batch_parameters(batch_size="100")

        # Not positive
        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_batch_parameters(batch_size=0)

        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_batch_parameters(batch_size=-10)

        # Too large
        with pytest.raises(UubedValidationError, match="too large"):
            validate_batch_parameters(batch_size=200000)

    def test_valid_max_memory(self):
        """Test valid max memory."""
        result = validate_batch_parameters(max_memory_mb=1024)
        assert result["max_memory_mb"] == 1024

        result = validate_batch_parameters(max_memory_mb=2048)
        assert result["max_memory_mb"] == 2048

    def test_invalid_max_memory(self):
        """Test invalid max memory."""
        # Non-integer
        with pytest.raises(UubedValidationError, match="must be an integer"):
            validate_batch_parameters(max_memory_mb="1024")

        # Not positive
        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_batch_parameters(max_memory_mb=0)

        with pytest.raises(UubedValidationError, match="must be positive"):
            validate_batch_parameters(max_memory_mb=-1024)

    def test_both_parameters(self):
        """Test both parameters together."""
        result = validate_batch_parameters(batch_size=500, max_memory_mb=2048)
        assert result["batch_size"] == 500
        assert result["max_memory_mb"] == 2048

    def test_none_parameters(self):
        """Test None parameters."""
        result = validate_batch_parameters()
        assert result == {}

        result = validate_batch_parameters(batch_size=None, max_memory_mb=None)
        assert result == {}


class TestValidateFilePath:
    """Test file path validation."""

    def test_valid_existing_file(self):
        """Test valid existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name

        try:
            result = validate_file_path(tmp_path, check_exists=True, check_readable=True)
            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_file()
        finally:
            os.unlink(tmp_path)

    def test_path_object_input(self):
        """Test Path object input."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = Path(tmp.name)

        try:
            result = validate_file_path(tmp_path, check_exists=True)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            tmp_path.unlink()

    def test_non_existent_file(self):
        """Test non-existent file."""
        with pytest.raises(UubedResourceError, match="does not exist"):
            validate_file_path("/nonexistent/file.txt", check_exists=True)

    def test_directory_as_file(self):
        """Test directory when expecting file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(UubedResourceError, match="not a file"):
                validate_file_path(tmp_dir, check_exists=True, check_readable=True)

    def test_writable_new_file(self):
        """Test writable path for new file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file = Path(tmp_dir) / "new_file.txt"

            result = validate_file_path(new_file, check_exists=False, check_writable=True)
            assert isinstance(result, Path)
            assert result.parent.exists()

    def test_writable_existing_file(self):
        """Test writable existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name

        try:
            result = validate_file_path(tmp_path, check_exists=True, check_writable=True)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(tmp_path)

    def test_non_writable_directory(self):
        """Test non-writable parent directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Make directory read-only
            os.chmod(tmp_dir, 0o444)

            try:
                new_file = Path(tmp_dir) / "new_file.txt"
                with pytest.raises(UubedResourceError, match="not writable"):
                    validate_file_path(new_file, check_exists=False, check_writable=True)
            finally:
                # Restore permissions for cleanup
                os.chmod(tmp_dir, 0o755)

    def test_invalid_path_type(self):
        """Test invalid path type."""
        with pytest.raises(UubedValidationError, match="must be a string or Path"):
            validate_file_path(123)

        with pytest.raises(UubedValidationError, match="must be a string or Path"):
            validate_file_path(None)


class TestMemoryValidation:
    """Test memory usage validation."""

    def test_valid_memory_usage(self):
        """Test valid memory usage."""
        # Should not raise error
        validate_memory_usage(100 * 1024 * 1024)  # 100MB
        validate_memory_usage(500 * 1024 * 1024)  # 500MB

    def test_excessive_memory_usage(self):
        """Test excessive memory usage."""
        with pytest.raises(UubedResourceError, match="too high"):
            validate_memory_usage(2 * 1024 * 1024 * 1024)  # 2GB

    def test_custom_memory_limit(self):
        """Test custom memory limit."""
        # Should pass with higher limit
        validate_memory_usage(1.5 * 1024 * 1024 * 1024, max_bytes=2 * 1024 * 1024 * 1024)

        # Should fail with lower limit
        with pytest.raises(UubedResourceError, match="too high"):
            validate_memory_usage(200 * 1024 * 1024, max_bytes=100 * 1024 * 1024)

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test different methods
        eq64_memory = estimate_memory_usage(100, 128, "eq64")
        shq64_memory = estimate_memory_usage(100, 128, "shq64")
        t8q64_memory = estimate_memory_usage(100, 128, "t8q64")

        # All should be positive
        assert eq64_memory > 0
        assert shq64_memory > 0
        assert t8q64_memory > 0

        # Different methods should have different estimates
        assert eq64_memory != shq64_memory
        assert shq64_memory != t8q64_memory

        # Base memory should be 100 * 128 = 12800
        base_memory = 100 * 128
        assert eq64_memory == int(base_memory * 2.0)
        assert shq64_memory == int(base_memory * 1.5)
        assert t8q64_memory == int(base_memory * 1.2)

    def test_estimate_memory_unknown_method(self):
        """Test memory estimation for unknown method."""
        unknown_memory = estimate_memory_usage(100, 128, "unknown_method")
        base_memory = 100 * 128
        assert unknown_memory == int(base_memory * 2.0)  # Default multiplier


class TestGPUValidation:
    """Test GPU parameter validation."""

    def test_gpu_not_available(self):
        """Test GPU validation when not available."""
        # This test depends on whether CuPy is installed
        try:
            import cupy
            # If CuPy is available, skip this test
            pytest.skip("CuPy is available, cannot test unavailable case")
        except ImportError:
            # CuPy not available, test should raise error
            with pytest.raises(UubedResourceError, match="not available"):
                validate_gpu_parameters()

    def test_device_id_validation(self):
        """Test device ID validation."""
        try:
            import cupy

            # Test invalid device ID type
            with pytest.raises(UubedValidationError, match="must be an integer"):
                validate_gpu_parameters(device_id="invalid")

            # Test negative device ID
            with pytest.raises(UubedValidationError, match="must be non-negative"):
                validate_gpu_parameters(device_id=-1)

            # Test valid device ID (assuming device 0 exists)
            try:
                result = validate_gpu_parameters(device_id=0)
                assert result["device_id"] == 0
            except UubedResourceError:
                # If no GPU is available, this is expected
                pass

            # Test out of range device ID
            with pytest.raises(UubedResourceError, match="not available"):
                validate_gpu_parameters(device_id=99)

        except ImportError:
            pytest.skip("CuPy not available for GPU tests")

    def test_no_device_id(self):
        """Test validation without device ID."""
        try:
            import cupy

            # Should check for general availability without device validation
            result = validate_gpu_parameters()
            assert result == {}

        except ImportError:
            # Should raise error about CuPy not being available
            with pytest.raises(UubedResourceError, match="not available"):
                validate_gpu_parameters()


if __name__ == "__main__":
    pytest.main([__file__])
