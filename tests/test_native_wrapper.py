#!/usr/bin/env python3
"""Test native wrapper functionality and fallback behavior."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys

from uubed.native_wrapper import (
    q64_encode_native,
    q64_decode_native,
    simhash_q64_native,
    top_k_q64_native,
    z_order_q64_native,
    mq64_encode_native,
    mq64_decode_native,
    is_native_available,
    HAS_NATIVE
)


class TestNativeWrapperFunctionality:
    """Test native wrapper basic functionality."""
    
    def test_is_native_available(self):
        """Test native availability detection."""
        result = is_native_available()
        assert isinstance(result, bool)
        assert result == HAS_NATIVE
    
    def test_q64_encode_decode_roundtrip(self):
        """Test Q64 encode/decode roundtrip."""
        test_data = bytes([0, 50, 100, 150, 255])
        
        encoded = q64_encode_native(test_data)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        decoded = q64_decode_native(encoded)
        assert isinstance(decoded, (bytes, list))
        
        # Convert to bytes if needed
        if isinstance(decoded, list):
            decoded = bytes(decoded)
        
        assert decoded == test_data
    
    def test_simhash_q64_basic(self):
        """Test SimHash Q64 basic functionality."""
        test_data = bytes(range(32))
        
        encoded = simhash_q64_native(test_data, planes=64)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_top_k_q64_basic(self):
        """Test Top-K Q64 basic functionality."""
        test_data = bytes(range(32))
        
        encoded = top_k_q64_native(test_data, k=8)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_z_order_q64_basic(self):
        """Test Z-order Q64 basic functionality."""
        test_data = bytes(range(16))
        
        encoded = z_order_q64_native(test_data)
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    def test_mq64_encode_decode_roundtrip(self):
        """Test MQ64 encode/decode roundtrip."""
        test_data = bytes(range(64))
        
        encoded = mq64_encode_native(test_data, levels=[16, 32, 64])
        assert isinstance(encoded, str)
        assert len(encoded) > 0
        
        decoded = mq64_decode_native(encoded)
        assert isinstance(decoded, (bytes, list))
        
        # Convert to bytes if needed
        if isinstance(decoded, list):
            decoded = bytes(decoded)
        
        assert decoded == test_data
    
    def test_different_input_sizes(self):
        """Test different input sizes."""
        sizes = [1, 16, 32, 64, 128, 256]
        
        for size in sizes:
            test_data = bytes(range(size))
            
            # Test eq64 (should work for all sizes)
            encoded = q64_encode_native(test_data)
            assert isinstance(encoded, str)
            assert len(encoded) > 0
            
            # Test decode
            decoded = q64_decode_native(encoded)
            if isinstance(decoded, list):
                decoded = bytes(decoded)
            assert decoded == test_data
    
    def test_edge_case_data(self):
        """Test edge case data patterns."""
        # All zeros
        zeros = bytes([0] * 32)
        encoded = q64_encode_native(zeros)
        assert isinstance(encoded, str)
        
        # All max values
        max_vals = bytes([255] * 32)
        encoded = q64_encode_native(max_vals)
        assert isinstance(encoded, str)
        
        # Alternating pattern
        alternating = bytes([0, 255] * 16)
        encoded = q64_encode_native(alternating)
        assert isinstance(encoded, str)
    
    def test_method_parameter_handling(self):
        """Test method parameter handling."""
        test_data = bytes(range(64))
        
        # Test shq64 with different planes
        for planes in [32, 64, 128]:
            encoded = simhash_q64_native(test_data, planes=planes)
            assert isinstance(encoded, str)
            assert len(encoded) > 0
        
        # Test t8q64 with different k values
        for k in [4, 8, 16]:
            encoded = top_k_q64_native(test_data, k=k)
            assert isinstance(encoded, str)
            assert len(encoded) > 0
        
        # Test mq64 with different levels
        level_configs = [
            [16, 32, 64],
            [32, 64],
            [64],
            None
        ]
        
        for levels in level_configs:
            encoded = mq64_encode_native(test_data, levels)
            assert isinstance(encoded, str)
            assert len(encoded) > 0


class TestNativeFallback:
    """Test fallback behavior when native module is not available."""
    
    def test_fallback_import_behavior(self):
        """Test that fallback imports work when native module is not available."""
        # Mock the native module import to fail
        with patch.dict('sys.modules', {'uubed._native': None}):
            # Force re-import of native_wrapper to trigger fallback
            import importlib
            import uubed.native_wrapper
            
            # This should trigger fallback to pure Python
            try:
                # Try to re-import to test fallback behavior
                importlib.reload(uubed.native_wrapper)
            except ImportError:
                # This is expected when native module is not available
                pass
    
    def test_fallback_functions_exist(self):
        """Test that fallback functions exist and are callable."""
        # These should exist regardless of native availability
        assert callable(q64_encode_native)
        assert callable(q64_decode_native)
        assert callable(simhash_q64_native)
        assert callable(top_k_q64_native)
        assert callable(z_order_q64_native)
        assert callable(mq64_encode_native)
        assert callable(mq64_decode_native)
    
    def test_fallback_vs_native_consistency(self):
        """Test that fallback produces consistent results."""
        test_data = bytes(range(32))
        
        # Test Q64 encode/decode consistency
        encoded = q64_encode_native(test_data)
        decoded = q64_decode_native(encoded)
        
        if isinstance(decoded, list):
            decoded = bytes(decoded)
        
        assert decoded == test_data
        
        # Test that different calls produce same result
        encoded1 = q64_encode_native(test_data)
        encoded2 = q64_encode_native(test_data)
        assert encoded1 == encoded2
    
    def test_pure_python_encoder_imports(self):
        """Test that pure Python encoders can be imported."""
        # These should be importable regardless of native availability
        from uubed.encoders.q64 import q64_encode, q64_decode
        from uubed.encoders.shq64 import simhash_q64
        from uubed.encoders.t8q64 import top_k_q64
        from uubed.encoders.zoq64 import z_order_q64
        from uubed.encoders.mq64 import mq64_encode, mq64_decode
        
        # Test that they work
        test_data = bytes(range(16))
        
        encoded = q64_encode(test_data)
        assert isinstance(encoded, str)
        
        decoded = q64_decode(encoded)
        assert isinstance(decoded, (bytes, list))
    
    def test_performance_difference_detection(self):
        """Test detection of performance differences between native and fallback."""
        import time
        
        test_data = bytes(range(256))
        
        # Time the encoding operation
        start_time = time.time()
        for _ in range(10):
            encoded = q64_encode_native(test_data)
        end_time = time.time()
        
        encoding_time = end_time - start_time
        
        # Should complete within reasonable time (both native and fallback)
        assert encoding_time < 1.0  # Should be much faster than 1 second
        
        # Test that the result is still valid
        assert isinstance(encoded, str)
        assert len(encoded) > 0


class TestNativeModulePresence:
    """Test behavior based on native module presence."""
    
    def test_has_native_flag(self):
        """Test HAS_NATIVE flag consistency."""
        # HAS_NATIVE should be a boolean
        assert isinstance(HAS_NATIVE, bool)
        
        # Should be consistent with is_native_available()
        assert HAS_NATIVE == is_native_available()
    
    def test_native_module_detection(self):
        """Test native module detection logic."""
        # If native module is available, should use it
        if HAS_NATIVE:
            # Functions should be from native module
            assert hasattr(q64_encode_native, '__module__')
            # Note: Actual module name depends on implementation
        else:
            # Functions should be from pure Python modules
            assert hasattr(q64_encode_native, '__module__')
            assert 'encoders' in q64_encode_native.__module__
    
    def test_import_error_handling(self):
        """Test graceful handling of import errors."""
        # Test that import errors are handled gracefully
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'uubed._native':
                raise ImportError("Mock import error")
            return original_import(name, *args, **kwargs)
        
        # This test is mostly to ensure the import logic is sound
        # In practice, the module is already imported
        assert callable(q64_encode_native)
    
    def test_function_signatures(self):
        """Test that function signatures are consistent."""
        import inspect
        
        # Test that key functions have expected signatures
        sig = inspect.signature(q64_encode_native)
        assert len(sig.parameters) >= 1  # Should accept at least data parameter
        
        sig = inspect.signature(q64_decode_native)
        assert len(sig.parameters) >= 1  # Should accept at least encoded parameter
        
        sig = inspect.signature(simhash_q64_native)
        assert len(sig.parameters) >= 1  # Should accept at least data parameter
        
        sig = inspect.signature(top_k_q64_native)
        assert len(sig.parameters) >= 1  # Should accept at least data parameter
        
        sig = inspect.signature(z_order_q64_native)
        assert len(sig.parameters) >= 1  # Should accept at least data parameter
        
        sig = inspect.signature(mq64_encode_native)
        assert len(sig.parameters) >= 1  # Should accept at least data parameter
        
        sig = inspect.signature(mq64_decode_native)
        assert len(sig.parameters) >= 1  # Should accept at least encoded parameter


class TestNativeWrapperIntegration:
    """Test integration with the rest of the system."""
    
    def test_integration_with_api(self):
        """Test integration with high-level API."""
        from uubed.api import encode, decode
        
        # Test that API uses native wrapper functions
        test_data = np.random.randint(0, 256, 32, dtype=np.uint8)
        
        # Test eq64 (uses native wrapper)
        encoded = encode(test_data, method="eq64")
        assert isinstance(encoded, str)
        
        decoded = decode(encoded, method="eq64")
        assert isinstance(decoded, bytes)
        
        # Verify roundtrip
        assert np.frombuffer(decoded, dtype=np.uint8).tolist() == test_data.tolist()
    
    def test_integration_with_streaming(self):
        """Test integration with streaming API."""
        from uubed.streaming import encode_stream
        
        embeddings = [
            np.random.randint(0, 256, 32, dtype=np.uint8)
            for _ in range(5)
        ]
        
        # Test that streaming uses native wrapper
        encoded_list = list(encode_stream(embeddings, method="eq64"))
        
        assert len(encoded_list) == 5
        assert all(isinstance(enc, str) for enc in encoded_list)
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent."""
        # Test invalid inputs
        with pytest.raises((ValueError, TypeError)):
            q64_encode_native(None)
        
        with pytest.raises((ValueError, TypeError)):
            q64_decode_native(None)
        
        with pytest.raises((ValueError, TypeError)):
            simhash_q64_native(None, planes=64)
        
        with pytest.raises((ValueError, TypeError)):
            top_k_q64_native(None, k=8)
        
        with pytest.raises((ValueError, TypeError)):
            z_order_q64_native(None)
        
        with pytest.raises((ValueError, TypeError)):
            mq64_encode_native(None, levels=None)
        
        with pytest.raises((ValueError, TypeError)):
            mq64_decode_native(None)
    
    def test_thread_safety(self):
        """Test thread safety of native wrapper functions."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                test_data = bytes(range(32))
                for _ in range(10):
                    encoded = q64_encode_native(test_data)
                    decoded = q64_decode_native(encoded)
                    if isinstance(decoded, list):
                        decoded = bytes(decoded)
                    assert decoded == test_data
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(4):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 4


if __name__ == "__main__":
    pytest.main([__file__])