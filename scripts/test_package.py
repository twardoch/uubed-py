#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["uubed"]
# ///
# this_file: scripts/test_package.py
"""Test that the uubed package works correctly after installation."""

import numpy as np
from uubed import encode, decode

def test_package():
    """Test basic functionality of the installed package."""
    print("Testing uubed package...")
    
    # Create test embedding
    embedding = np.random.randint(0, 256, 256, dtype=np.uint8)
    print(f"Created test embedding of shape: {embedding.shape}")
    
    # Test all encoding methods
    methods = ["q64", "eq64", "shq64", "t8q64", "zoq64"]
    
    for method in methods:
        print(f"\nTesting {method} encoding...")
        try:
            encoded = encode(embedding, method=method)
            print(f"  ✓ Encoded successfully, length: {len(encoded)}")
            
            # Test decode for eq64
            if method == "eq64":
                decoded = decode(encoded)
                print(f"  ✓ Decoded successfully")
                # Verify roundtrip
                original_bytes = embedding.tobytes()
                if decoded == original_bytes:
                    print(f"  ✓ Roundtrip successful")
                else:
                    print(f"  ✗ Roundtrip failed!")
                    
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    # Test auto method
    print("\nTesting auto method...")
    try:
        encoded = encode(embedding, method="auto")
        print(f"  ✓ Auto encoding successful, length: {len(encoded)}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Check if native module is available
    print("\nChecking native module...")
    try:
        from uubed import _native
        print("  ✓ Native module is available")
    except ImportError:
        print("  ⚠ Native module not available, using pure Python")
    
    print("\n✅ Package testing complete!")

if __name__ == "__main__":
    test_package()