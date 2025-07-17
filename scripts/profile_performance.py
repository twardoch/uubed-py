#!/usr/bin/env python3
"""
Comprehensive performance profiling script for uubed-py.

This script provides detailed performance analysis of the uubed library,
including:
- Encoding method performance comparison
- Memory usage patterns and allocation overhead
- Function call profiling and hot path identification  
- Validation overhead analysis
- Streaming vs batch operation comparison
- Native vs Python implementation comparison
"""

import sys
import time
import cProfile
import pstats
import tracemalloc
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import json

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uubed
from uubed.native_wrapper import is_native_available
from uubed.validation import estimate_memory_usage

# Try to import line profiler if available
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

# Try to import memory profiler if available
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation: str
    method: str
    input_size: int
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float


@dataclass
class ProfilingConfig:
    """Configuration for profiling runs."""
    embedding_sizes: List[int]
    methods: List[str]
    iterations: int
    warmup_iterations: int
    enable_line_profiling: bool
    enable_memory_profiling: bool
    enable_cpu_profiling: bool
    batch_sizes: List[int]
    streaming_batch_sizes: List[int]


class PerformanceProfiler:
    """Comprehensive performance profiler for uubed operations."""
    
    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.results: List[PerformanceMetrics] = []
        self.detailed_profiles: Dict[str, Any] = {}
        
    @contextmanager
    def profile_context(self, operation: str):
        """Context manager for profiling operations."""
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial CPU usage
        process = psutil.Process()
        initial_cpu = process.cpu_percent()
        
        # Force garbage collection
        gc.collect()
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get CPU usage
            final_cpu = process.cpu_percent()
            
            # Store basic metrics
            self.last_operation_time = end_time - start_time
            self.last_memory_peak = peak / (1024 * 1024)  # MB
            self.last_memory_current = current / (1024 * 1024)  # MB
            self.last_cpu_percent = final_cpu - initial_cpu
    
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Tuple[float, List[float]]:
        """Benchmark a function with detailed timing statistics."""
        times = []
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            func(*args, **kwargs)
        
        # Actual benchmarking
        for _ in range(self.config.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return sum(times), times
    
    def profile_encoding_methods(self) -> Dict[str, List[PerformanceMetrics]]:
        """Profile all encoding methods across different input sizes."""
        print("Profiling encoding methods...")
        method_results = {}
        
        for method in self.config.methods:
            print(f"  Profiling method: {method}")
            method_results[method] = []
            
            for size in self.config.embedding_sizes:
                print(f"    Size: {size}")
                
                # Generate test data
                test_data = np.random.randint(0, 256, size, dtype=np.uint8)
                
                with self.profile_context(f"encode_{method}_{size}"):
                    total_time, times = self.benchmark_function(
                        uubed.encode, test_data, method=method
                    )
                
                metrics = PerformanceMetrics(
                    operation="encode",
                    method=method,
                    input_size=size,
                    iterations=self.config.iterations,
                    total_time=total_time,
                    avg_time=np.mean(times),
                    min_time=np.min(times),
                    max_time=np.max(times),
                    std_time=np.std(times),
                    throughput_ops_per_sec=self.config.iterations / total_time,
                    memory_peak_mb=self.last_memory_peak,
                    memory_current_mb=self.last_memory_current,
                    cpu_percent=self.last_cpu_percent
                )
                
                method_results[method].append(metrics)
                self.results.append(metrics)
        
        return method_results
    
    def profile_validation_overhead(self) -> Dict[str, PerformanceMetrics]:
        """Profile the overhead of input validation."""
        print("Profiling validation overhead...")
        
        size = 512  # Standard size
        test_data = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # Profile validation time
        from uubed.validation import validate_embedding_input, validate_encoding_method
        
        with self.profile_context("validation"):
            total_time, times = self.benchmark_function(
                validate_embedding_input, test_data, "eq64"
            )
        
        validation_metrics = PerformanceMetrics(
            operation="validation",
            method="input_validation",
            input_size=size,
            iterations=self.config.iterations,
            total_time=total_time,
            avg_time=np.mean(times),
            min_time=np.min(times),
            max_time=np.max(times),
            std_time=np.std(times),
            throughput_ops_per_sec=self.config.iterations / total_time,
            memory_peak_mb=self.last_memory_peak,
            memory_current_mb=self.last_memory_current,
            cpu_percent=self.last_cpu_percent
        )
        
        # Profile method validation
        with self.profile_context("method_validation"):
            total_time, times = self.benchmark_function(
                validate_encoding_method, "eq64"
            )
        
        method_validation_metrics = PerformanceMetrics(
            operation="validation",
            method="method_validation",
            input_size=0,
            iterations=self.config.iterations,
            total_time=total_time,
            avg_time=np.mean(times),
            min_time=np.min(times),
            max_time=np.max(times),
            std_time=np.std(times),
            throughput_ops_per_sec=self.config.iterations / total_time,
            memory_peak_mb=self.last_memory_peak,
            memory_current_mb=self.last_memory_current,
            cpu_percent=self.last_cpu_percent
        )
        
        self.results.extend([validation_metrics, method_validation_metrics])
        
        return {
            "input_validation": validation_metrics,
            "method_validation": method_validation_metrics
        }
    
    def profile_streaming_vs_batch(self) -> Dict[str, List[PerformanceMetrics]]:
        """Profile streaming vs batch operations."""
        print("Profiling streaming vs batch operations...")
        
        streaming_results = []
        batch_results = []
        
        embedding_count = 1000
        embedding_size = 512
        
        # Generate test embeddings
        test_embeddings = [
            np.random.randint(0, 256, embedding_size, dtype=np.uint8)
            for _ in range(embedding_count)
        ]
        
        # Profile batch encoding
        with self.profile_context("batch_encode"):
            total_time, times = self.benchmark_function(
                uubed.batch_encode, test_embeddings, method="eq64"
            )
        
        batch_metrics = PerformanceMetrics(
            operation="batch_encode",
            method="eq64",
            input_size=embedding_count * embedding_size,
            iterations=len(times),
            total_time=total_time,
            avg_time=np.mean(times),
            min_time=np.min(times),
            max_time=np.max(times),
            std_time=np.std(times),
            throughput_ops_per_sec=embedding_count * len(times) / total_time,
            memory_peak_mb=self.last_memory_peak,
            memory_current_mb=self.last_memory_current,
            cpu_percent=self.last_cpu_percent
        )
        batch_results.append(batch_metrics)
        
        # Profile streaming encoding with different batch sizes
        for batch_size in self.config.streaming_batch_sizes:
            with self.profile_context(f"stream_encode_batch_{batch_size}"):
                start_time = time.perf_counter()
                
                for _ in range(self.config.warmup_iterations):
                    list(uubed.encode_stream(test_embeddings, method="eq64", batch_size=batch_size))
                
                times = []
                for _ in range(self.config.iterations):
                    iter_start = time.perf_counter()
                    list(uubed.encode_stream(test_embeddings, method="eq64", batch_size=batch_size))
                    iter_end = time.perf_counter()
                    times.append(iter_end - iter_start)
                
                total_time = sum(times)
            
            stream_metrics = PerformanceMetrics(
                operation="stream_encode",
                method=f"eq64_batch_{batch_size}",
                input_size=embedding_count * embedding_size,
                iterations=len(times),
                total_time=total_time,
                avg_time=np.mean(times),
                min_time=np.min(times),
                max_time=np.max(times),
                std_time=np.std(times),
                throughput_ops_per_sec=embedding_count * len(times) / total_time,
                memory_peak_mb=self.last_memory_peak,
                memory_current_mb=self.last_memory_current,
                cpu_percent=self.last_cpu_percent
            )
            streaming_results.append(stream_metrics)
        
        self.results.extend(batch_results + streaming_results)
        
        return {
            "batch": batch_results,
            "streaming": streaming_results
        }
    
    def profile_native_vs_python(self) -> Optional[Dict[str, Any]]:
        """Profile native vs Python implementation performance."""
        if not is_native_available():
            print("Native implementation not available, skipping comparison")
            return None
        
        print("Profiling native vs Python implementation...")
        
        size = 512
        test_data = np.random.randint(0, 256, size, dtype=np.uint8).tobytes()
        
        results = {}
        
        # Profile native implementations
        from uubed.native_wrapper import (
            q64_encode_native, simhash_q64_native, 
            top_k_q64_native, z_order_q64_native
        )
        
        native_functions = {
            "q64": lambda: q64_encode_native(test_data),
            "simhash": lambda: simhash_q64_native(test_data),
            "topk": lambda: top_k_q64_native(test_data),
            "zorder": lambda: z_order_q64_native(test_data)
        }
        
        # Profile Python implementations
        from uubed.encoders import q64, shq64, t8q64, zoq64
        
        python_functions = {
            "q64": lambda: q64.q64_encode(test_data),
            "simhash": lambda: shq64.simhash_q64(test_data),
            "topk": lambda: t8q64.top_k_q64(test_data),
            "zorder": lambda: zoq64.z_order_q64(test_data)
        }
        
        for method_name in native_functions.keys():
            # Profile native
            with self.profile_context(f"native_{method_name}"):
                total_time_native, times_native = self.benchmark_function(
                    native_functions[method_name]
                )
            
            native_metrics = PerformanceMetrics(
                operation="encode_native",
                method=method_name,
                input_size=size,
                iterations=self.config.iterations,
                total_time=total_time_native,
                avg_time=np.mean(times_native),
                min_time=np.min(times_native),
                max_time=np.max(times_native),
                std_time=np.std(times_native),
                throughput_ops_per_sec=self.config.iterations / total_time_native,
                memory_peak_mb=self.last_memory_peak,
                memory_current_mb=self.last_memory_current,
                cpu_percent=self.last_cpu_percent
            )
            
            # Profile Python
            with self.profile_context(f"python_{method_name}"):
                total_time_python, times_python = self.benchmark_function(
                    python_functions[method_name]
                )
            
            python_metrics = PerformanceMetrics(
                operation="encode_python",
                method=method_name,
                input_size=size,
                iterations=self.config.iterations,
                total_time=total_time_python,
                avg_time=np.mean(times_python),
                min_time=np.min(times_python),
                max_time=np.max(times_python),
                std_time=np.std(times_python),
                throughput_ops_per_sec=self.config.iterations / total_time_python,
                memory_peak_mb=self.last_memory_peak,
                memory_current_mb=self.last_memory_current,
                cpu_percent=self.last_cpu_percent
            )
            
            speedup = total_time_python / total_time_native
            
            results[method_name] = {
                "native": native_metrics,
                "python": python_metrics,
                "speedup": speedup
            }
            
            self.results.extend([native_metrics, python_metrics])
        
        return results
    
    def profile_function_calls(self) -> Dict[str, Any]:
        """Profile function calls using cProfile."""
        if not self.config.enable_cpu_profiling:
            return {}
        
        print("Profiling function calls...")
        
        size = 512
        test_data = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # Profile encoding function
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(100):  # Fewer iterations for detailed profiling
            uubed.encode(test_data, method="eq64")
        
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        # Store detailed profile data
        profile_data = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            profile_data.append({
                'function': f"{func[0]}:{func[1]}({func[2]})",
                'call_count': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call_time': tt/cc if cc > 0 else 0,
                'per_call_cumulative': ct/cc if cc > 0 else 0
            })
        
        # Sort by cumulative time
        profile_data.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        self.detailed_profiles['function_calls'] = profile_data[:50]  # Top 50
        
        return self.detailed_profiles['function_calls']
    
    def profile_memory_patterns(self) -> Dict[str, Any]:
        """Profile memory allocation patterns."""
        print("Profiling memory allocation patterns...")
        
        results = {}
        
        for method in ['eq64', 'shq64', 't8q64']:
            for size in [128, 512, 1024]:
                test_data = np.random.randint(0, 256, size, dtype=np.uint8)
                
                # Track memory allocations
                tracemalloc.start()
                
                # Encode multiple times to see allocation patterns
                for _ in range(10):
                    encoded = uubed.encode(test_data, method=method)
                
                current, peak = tracemalloc.get_traced_memory()
                
                # Get top memory allocations
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                tracemalloc.stop()
                
                results[f"{method}_{size}"] = {
                    'peak_memory_mb': peak / (1024 * 1024),
                    'current_memory_mb': current / (1024 * 1024),
                    'top_allocations': [
                        {
                            'file': str(stat.traceback.format()[0]) if stat.traceback.format() else 'unknown',
                            'size_mb': stat.size / (1024 * 1024),
                            'count': stat.count
                        }
                        for stat in top_stats[:10]
                    ]
                }
        
        self.detailed_profiles['memory_patterns'] = results
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("Generating performance report...")
        
        # Organize results by operation and method
        organized_results = {}
        for metric in self.results:
            operation = metric.operation
            method = metric.method
            
            if operation not in organized_results:
                organized_results[operation] = {}
            if method not in organized_results[operation]:
                organized_results[operation][method] = []
            
            organized_results[operation][method].append(metric)
        
        # Calculate summary statistics
        summary = {}
        for operation, methods in organized_results.items():
            summary[operation] = {}
            for method, metrics in methods.items():
                if metrics:
                    avg_times = [m.avg_time for m in metrics]
                    throughputs = [m.throughput_ops_per_sec for m in metrics]
                    memory_peaks = [m.memory_peak_mb for m in metrics]
                    
                    summary[operation][method] = {
                        'avg_time_ms': np.mean(avg_times) * 1000,
                        'avg_throughput': np.mean(throughputs),
                        'avg_memory_peak_mb': np.mean(memory_peaks),
                        'data_points': len(metrics)
                    }
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        report = {
            'timestamp': time.time(),
            'config': {
                'embedding_sizes': self.config.embedding_sizes,
                'methods': self.config.methods,
                'iterations': self.config.iterations,
                'native_available': is_native_available()
            },
            'summary': summary,
            'detailed_results': [
                {
                    'operation': m.operation,
                    'method': m.method,
                    'input_size': m.input_size,
                    'iterations': m.iterations,
                    'avg_time_ms': m.avg_time * 1000,
                    'throughput_ops_per_sec': m.throughput_ops_per_sec,
                    'memory_peak_mb': m.memory_peak_mb,
                    'cpu_percent': m.cpu_percent
                }
                for m in self.results
            ],
            'bottlenecks': bottlenecks,
            'detailed_profiles': self.detailed_profiles
        }
        
        return report
    
    def _identify_bottlenecks(self) -> Dict[str, Any]:
        """Identify performance bottlenecks from the collected data."""
        bottlenecks = {
            'slowest_operations': [],
            'memory_intensive_operations': [],
            'cpu_intensive_operations': [],
            'recommendations': []
        }
        
        # Find slowest operations
        sorted_by_time = sorted(self.results, key=lambda x: x.avg_time, reverse=True)
        bottlenecks['slowest_operations'] = [
            {
                'operation': m.operation,
                'method': m.method,
                'input_size': m.input_size,
                'avg_time_ms': m.avg_time * 1000
            }
            for m in sorted_by_time[:10]
        ]
        
        # Find memory intensive operations
        sorted_by_memory = sorted(self.results, key=lambda x: x.memory_peak_mb, reverse=True)
        bottlenecks['memory_intensive_operations'] = [
            {
                'operation': m.operation,
                'method': m.method,
                'input_size': m.input_size,
                'memory_peak_mb': m.memory_peak_mb
            }
            for m in sorted_by_memory[:10]
        ]
        
        # Find CPU intensive operations
        sorted_by_cpu = sorted(self.results, key=lambda x: x.cpu_percent, reverse=True)
        bottlenecks['cpu_intensive_operations'] = [
            {
                'operation': m.operation,
                'method': m.method,
                'input_size': m.input_size,
                'cpu_percent': m.cpu_percent
            }
            for m in sorted_by_cpu[:10]
        ]
        
        # Generate recommendations
        recommendations = []
        
        # Check for validation overhead
        validation_results = [r for r in self.results if r.operation == 'validation']
        if validation_results:
            avg_validation_time = np.mean([r.avg_time for r in validation_results])
            if avg_validation_time > 0.001:  # 1ms
                recommendations.append(
                    "Validation overhead is significant. Consider caching validation results or reducing validation complexity."
                )
        
        # Check for memory usage patterns
        high_memory_ops = [r for r in self.results if r.memory_peak_mb > 100]
        if high_memory_ops:
            recommendations.append(
                "High memory usage detected. Consider using streaming operations for large datasets."
            )
        
        # Check for native vs Python performance
        if is_native_available():
            recommendations.append(
                "Native implementation is available. Ensure it's being used for performance-critical operations."
            )
        else:
            recommendations.append(
                "Native implementation not available. Install the native extension for better performance."
            )
        
        bottlenecks['recommendations'] = recommendations
        
        return bottlenecks


def create_default_config() -> ProfilingConfig:
    """Create default profiling configuration."""
    return ProfilingConfig(
        embedding_sizes=[32, 128, 256, 512, 768, 1024],
        methods=['eq64', 'shq64', 't8q64', 'zoq64'],
        iterations=100,
        warmup_iterations=5,
        enable_line_profiling=LINE_PROFILER_AVAILABLE,
        enable_memory_profiling=MEMORY_PROFILER_AVAILABLE,
        enable_cpu_profiling=True,
        batch_sizes=[100, 500, 1000],
        streaming_batch_sizes=[100, 500, 1000]
    )


def save_report(report: Dict[str, Any], output_path: str = "performance_profile_report.json"):
    """Save the performance report to a JSON file."""
    output_file = Path(__file__).parent / output_path
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Performance report saved to: {output_file}")


def print_summary(report: Dict[str, Any]):
    """Print a summary of the performance report."""
    print("\n" + "="*80)
    print("UUBED PERFORMANCE PROFILING SUMMARY")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Native available: {report['config']['native_available']}")
    print(f"  Embedding sizes: {report['config']['embedding_sizes']}")
    print(f"  Methods tested: {report['config']['methods']}")
    print(f"  Iterations per test: {report['config']['iterations']}")
    
    print(f"\nEncoding Performance Summary:")
    print(f"{'Method':<12} {'Avg Time (ms)':<15} {'Throughput (ops/s)':<20} {'Memory (MB)':<12}")
    print("-" * 65)
    
    if 'encode' in report['summary']:
        for method, stats in report['summary']['encode'].items():
            print(f"{method:<12} {stats['avg_time_ms']:<15.3f} {stats['avg_throughput']:<20.1f} {stats['avg_memory_peak_mb']:<12.2f}")
    
    print(f"\nTop Performance Bottlenecks:")
    bottlenecks = report.get('bottlenecks', {})
    
    if 'slowest_operations' in bottlenecks:
        print("\nSlowest Operations:")
        for i, op in enumerate(bottlenecks['slowest_operations'][:5]):
            print(f"  {i+1}. {op['operation']} ({op['method']}) - {op['avg_time_ms']:.3f}ms")
    
    if 'memory_intensive_operations' in bottlenecks:
        print("\nMemory Intensive Operations:")
        for i, op in enumerate(bottlenecks['memory_intensive_operations'][:5]):
            print(f"  {i+1}. {op['operation']} ({op['method']}) - {op['memory_peak_mb']:.2f}MB")
    
    if 'recommendations' in bottlenecks:
        print("\nOptimization Recommendations:")
        for i, rec in enumerate(bottlenecks['recommendations']):
            print(f"  {i+1}. {rec}")
    
    print("\n" + "="*80)


def main():
    """Main profiling entry point."""
    print("Starting comprehensive uubed performance profiling...")
    
    # Create configuration
    config = create_default_config()
    
    # Initialize profiler
    profiler = PerformanceProfiler(config)
    
    # Run all profiling operations
    profiler.profile_encoding_methods()
    profiler.profile_validation_overhead()
    profiler.profile_streaming_vs_batch()
    profiler.profile_native_vs_python()
    profiler.profile_function_calls()
    profiler.profile_memory_patterns()
    
    # Generate and save report
    report = profiler.generate_report()
    save_report(report)
    
    # Print summary
    print_summary(report)
    
    print("\nProfiling complete! See performance_profile_report.json for detailed results.")


if __name__ == "__main__":
    main()