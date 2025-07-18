#!/usr/bin/env python3
"""Performance regression testing and benchmarking for uubed."""

import sys
import time
import json
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uubed


def benchmark_encoding_methods(
    sizes: List[int] = [64, 128, 512, 1024],
    iterations: int = 100,
    methods: List[str] = ["eq64", "shq64", "t8q64", "zoq64"]
) -> Dict[str, Any]:
    """Benchmark encoding performance across different methods and sizes."""
    results = {
        "timestamp": time.time(),
        "iterations": iterations,
        "benchmarks": {}
    }
    
    for size in sizes:
        results["benchmarks"][f"size_{size}"] = {}
        
        # Generate test data
        test_data = np.random.randint(0, 256, size, dtype=np.uint8)
        
        for method in methods:
            print(f"Benchmarking {method} with size {size}...")
            
            # Warmup
            for _ in range(5):
                uubed.encode(test_data, method=method)
            
            # Memory tracking
            tracemalloc.start()
            start_time = time.perf_counter()
            
            for _ in range(iterations):
                encoded = uubed.encode(test_data, method=method)
            
            end_time = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_time = total_time / iterations
            throughput = iterations / total_time
            
            results["benchmarks"][f"size_{size}"][method] = {
                "avg_time_ms": avg_time * 1000,
                "throughput_ops_per_sec": throughput,
                "memory_current_kb": current / 1024,
                "memory_peak_kb": peak / 1024,
                "encoded_length": len(encoded) if method == "eq64" else len(encoded)
            }
    
    return results


def benchmark_streaming_performance(
    batch_sizes: List[int] = [10, 50, 100, 500],
    embedding_size: int = 512,
    total_embeddings: int = 1000
) -> Dict[str, Any]:
    """Benchmark streaming encoding performance."""
    results = {
        "timestamp": time.time(),
        "embedding_size": embedding_size,
        "total_embeddings": total_embeddings,
        "streaming_benchmarks": {}
    }
    
    # Generate test data
    test_embeddings = [
        np.random.randint(0, 256, embedding_size, dtype=np.uint8)
        for _ in range(total_embeddings)
    ]
    
    for batch_size in batch_sizes:
        print(f"Benchmarking streaming with batch size {batch_size}...")
        
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Use streaming encoder
        encoded_count = 0
        for encoded in uubed.encode_stream(
            test_embeddings,
            method="eq64",
            batch_size=batch_size,
            progress=False
        ):
            encoded_count += 1
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time = end_time - start_time
        throughput = total_embeddings / total_time
        
        results["streaming_benchmarks"][f"batch_{batch_size}"] = {
            "total_time_sec": total_time,
            "throughput_embeddings_per_sec": throughput,
            "memory_current_kb": current / 1024,
            "memory_peak_kb": peak / 1024,
            "encoded_count": encoded_count
        }
    
    return results


def memory_profile_large_datasets(
    sizes: List[int] = [1000, 5000, 10000],
    embedding_dim: int = 512
) -> Dict[str, Any]:
    """Profile memory usage for large dataset processing."""
    results = {
        "timestamp": time.time(),
        "embedding_dim": embedding_dim,
        "memory_profiles": {}
    }
    
    for size in sizes:
        print(f"Memory profiling with {size} embeddings...")
        
        # Generate test data
        test_embeddings = [
            np.random.randint(0, 256, embedding_dim, dtype=np.uint8)
            for _ in range(size)
        ]
        
        # Test batch encoding (high memory)
        tracemalloc.start()
        batch_results = uubed.batch_encode(test_embeddings, method="eq64")
        current_batch, peak_batch = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Test streaming encoding (low memory)
        tracemalloc.start()
        streaming_results = list(uubed.encode_stream(
            test_embeddings,
            method="eq64",
            batch_size=100,
            progress=False
        ))
        current_stream, peak_stream = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results["memory_profiles"][f"size_{size}"] = {
            "batch_memory_current_mb": current_batch / (1024 * 1024),
            "batch_memory_peak_mb": peak_batch / (1024 * 1024),
            "streaming_memory_current_mb": current_stream / (1024 * 1024),
            "streaming_memory_peak_mb": peak_stream / (1024 * 1024),
            "memory_reduction_factor": peak_batch / peak_stream,
            "batch_count": len(batch_results),
            "streaming_count": len(streaming_results)
        }
    
    return results


def save_benchmark_results(results: Dict[str, Any], filename: str = "benchmark_results.json") -> None:
    """Save benchmark results to JSON file."""
    output_path = Path(__file__).parent / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def compare_with_baseline(results: Dict[str, Any], baseline_file: str = "baseline_benchmark.json") -> None:
    """Compare current results with baseline performance."""
    baseline_path = Path(__file__).parent / baseline_file
    
    if not baseline_path.exists():
        print(f"No baseline found at {baseline_path}. Saving current results as baseline.")
        save_benchmark_results(results, baseline_file)
        return
    
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    print("\n=== Performance Comparison ===")
    
    # Compare encoding benchmarks
    if "benchmarks" in results and "benchmarks" in baseline:
        for size_key in results["benchmarks"]:
            if size_key in baseline["benchmarks"]:
                print(f"\n{size_key.replace('_', ' ').title()}:")
                for method in results["benchmarks"][size_key]:
                    if method in baseline["benchmarks"][size_key]:
                        current = results["benchmarks"][size_key][method]
                        base = baseline["benchmarks"][size_key][method]
                        
                        time_change = (current["avg_time_ms"] / base["avg_time_ms"] - 1) * 100
                        throughput_change = (current["throughput_ops_per_sec"] / base["throughput_ops_per_sec"] - 1) * 100
                        
                        print(f"  {method}: time {time_change:+.1f}%, throughput {throughput_change:+.1f}%")


def main():
    """Run comprehensive benchmarking suite."""
    print("Starting uubed performance benchmarking...")
    
    # Core encoding benchmarks
    print("\n1. Encoding method benchmarks...")
    encoding_results = benchmark_encoding_methods()
    
    # Streaming performance
    print("\n2. Streaming performance benchmarks...")
    streaming_results = benchmark_streaming_performance()
    
    # Memory profiling
    print("\n3. Memory usage profiling...")
    memory_results = memory_profile_large_datasets()
    
    # Combine results
    all_results = {
        **encoding_results,
        **streaming_results,
        **memory_results
    }
    
    # Save and compare
    save_benchmark_results(all_results)
    compare_with_baseline(all_results)
    
    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()