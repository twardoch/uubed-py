#!/usr/bin/env python3
# this_file: benchmarks/bench_encoders.py
"""Benchmark script comparing native Rust vs pure Python performance."""

import time
import numpy as np
from uubed import encode
from uubed.native_wrapper import is_native_available
from uubed.encoders import q64, eq64, shq64, t8q64, zoq64

# Import native functions directly
if is_native_available():
    from uubed._native import (
        q64_encode_native,
        q64_decode_native,
        simhash_q64_native,
        top_k_q64_native,
        z_order_q64_native,
    )


def benchmark_function(func, *args, iterations=1000):
    """Time a function over multiple iterations."""
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()
    return (end - start) / iterations


def main():
    """Run benchmarks."""
    # Generate test data
    sizes = [32, 256, 1024]
    
    print("uubed Encoding Performance Benchmarks")
    print(f"Native acceleration available: {is_native_available()}")
    print("=" * 80)
    print(f"{'Size':<10} {'Method':<15} {'Implementation':<15} {'Time (μs)':<15} {'Throughput (MB/s)':<15}")
    print("-" * 80)
    
    for size in sizes:
        data = np.random.randint(0, 256, size, dtype=np.uint8).tolist()
        data_bytes = bytes(data)
        
        # Benchmark each encoding method - both pure Python and native
        methods = []
        
        # Always benchmark pure Python
        methods.extend([
            ("q64", "Pure Python", lambda: q64.q64_encode(data)),
            ("eq64", "Pure Python", lambda: eq64.eq64_encode(data)),
            ("shq64", "Pure Python", lambda: shq64.simhash_q64(data)),
            ("t8q64", "Pure Python", lambda: t8q64.top_k_q64(data)),
            ("zoq64", "Pure Python", lambda: zoq64.z_order_q64(data)),
        ])
        
        # Benchmark native if available
        if is_native_available():
            methods.extend([
                ("q64", "Native Rust", lambda: q64_encode_native(data_bytes)),
                ("shq64", "Native Rust", lambda: simhash_q64_native(data_bytes)),
                ("t8q64", "Native Rust", lambda: top_k_q64_native(data_bytes)),
                ("zoq64", "Native Rust", lambda: z_order_q64_native(data_bytes)),
            ])
        
        for method_name, impl_name, method_func in methods:
            # Use fewer iterations for larger sizes
            iters = 1000 if size <= 256 else 100
            
            time_per_op = benchmark_function(method_func, iterations=iters)
            throughput = size / (time_per_op * 1e6)  # MB/s
            
            print(f"{size:<10} {method_name:<15} {impl_name:<15} {time_per_op*1e6:<15.2f} {throughput:<15.2f}")
        
        print()
    
    # If native is available, calculate speedup
    if is_native_available():
        print("\nSpeedup Analysis")
        print("=" * 50)
        print(f"{'Method':<15} {'Size':<10} {'Speedup':<15}")
        print("-" * 50)
        
        for size in [32, 256, 1024]:
            data = np.random.randint(0, 256, size, dtype=np.uint8).tolist()
            data_bytes = bytes(data)
            
            # Compare q64 encoding
            py_time = benchmark_function(q64.q64_encode, data, iterations=100)
            native_time = benchmark_function(q64_encode_native, data_bytes, iterations=100)
            speedup = py_time / native_time
            print(f"{'q64':<15} {size:<10} {speedup:<15.2f}x")
            
            # Compare simhash
            py_time = benchmark_function(shq64.simhash_q64, data, iterations=100)
            native_time = benchmark_function(simhash_q64_native, data_bytes, iterations=100)
            speedup = py_time / native_time
            print(f"{'shq64':<15} {size:<10} {speedup:<15.2f}x")


if __name__ == "__main__":
    main()