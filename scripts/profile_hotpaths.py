#!/usr/bin/env python3
"""
Hot path profiling script for uubed-py.

This script focuses on profiling the most performance-critical code paths
using line-by-line profiling to identify specific bottlenecks within functions.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Callable
import json

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uubed
from uubed.native_wrapper import is_native_available

# Try to import line profiler
try:
    from line_profiler import LineProfiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    print("line_profiler not available. Install with: pip install line_profiler")

try:
    import cProfile
    import pstats
    from pstats import SortKey
    CPROFILE_AVAILABLE = True
except ImportError:
    CPROFILE_AVAILABLE = False


class HotPathProfiler:
    """Specialized profiler for identifying hot code paths."""
    
    def __init__(self):
        self.results = {}
        
    def profile_api_encode_path(self):
        """Profile the main encode API path line by line."""
        if not LINE_PROFILER_AVAILABLE:
            print("Skipping line profiling (line_profiler not available)")
            return
        
        print("Profiling API encode path...")
        
        # Import the functions we want to profile
        from uubed.api import encode, _auto_select_method
        from uubed.validation import validate_embedding_input, validate_encoding_method, validate_method_parameters
        
        profiler = LineProfiler()
        
        # Add functions to profile
        profiler.add_function(encode)
        profiler.add_function(validate_embedding_input)
        profiler.add_function(validate_encoding_method)
        profiler.add_function(validate_method_parameters)
        profiler.add_function(_auto_select_method)
        
        # Generate test data
        test_data = np.random.randint(0, 256, 512, dtype=np.uint8)
        
        # Enable profiling and run test
        profiler.enable_by_count()
        
        for _ in range(100):
            uubed.encode(test_data, method="eq64")
        
        profiler.disable_by_count()
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            profiler.print_stats()
            profile_output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.results['api_encode_path'] = profile_output
        
        # Also save to file
        with open(Path(__file__).parent / "line_profile_api_encode.txt", "w") as f:
            f.write(profile_output)
        
        print("API encode path profiling complete. Results saved to line_profile_api_encode.txt")
    
    def profile_streaming_encode_path(self):
        """Profile the streaming encode path."""
        if not LINE_PROFILER_AVAILABLE:
            return
        
        print("Profiling streaming encode path...")
        
        from uubed.streaming import encode_stream, StreamingEncoder
        
        profiler = LineProfiler()
        profiler.add_function(encode_stream)
        
        # Generate test data
        test_embeddings = [
            np.random.randint(0, 256, 256, dtype=np.uint8)
            for _ in range(100)
        ]
        
        profiler.enable_by_count()
        
        # Test streaming
        for _ in range(10):
            list(uubed.encode_stream(test_embeddings, method="eq64", batch_size=10))
        
        profiler.disable_by_count()
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            profiler.print_stats()
            profile_output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.results['streaming_encode_path'] = profile_output
        
        with open(Path(__file__).parent / "line_profile_streaming.txt", "w") as f:
            f.write(profile_output)
        
        print("Streaming encode path profiling complete. Results saved to line_profile_streaming.txt")
    
    def profile_validation_functions(self):
        """Profile validation functions in detail."""
        if not LINE_PROFILER_AVAILABLE:
            return
        
        print("Profiling validation functions...")
        
        from uubed.validation import (
            validate_embedding_input, 
            validate_encoding_method,
            validate_method_parameters,
            _validate_embedding_dimensions,
            estimate_memory_usage
        )
        
        profiler = LineProfiler()
        profiler.add_function(validate_embedding_input)
        profiler.add_function(validate_encoding_method)
        profiler.add_function(validate_method_parameters)
        profiler.add_function(_validate_embedding_dimensions)
        profiler.add_function(estimate_memory_usage)
        
        # Test different input types
        test_data_list = [1, 2, 3, 4, 5] * 100
        test_data_bytes = bytes(test_data_list)
        test_data_numpy = np.array(test_data_list, dtype=np.uint8)
        test_data_float = np.random.random(500).astype(np.float32)
        
        profiler.enable_by_count()
        
        for _ in range(100):
            # Test different validation scenarios
            validate_embedding_input(test_data_list, "eq64")
            validate_embedding_input(test_data_bytes, "shq64")
            validate_embedding_input(test_data_numpy, "t8q64")
            validate_embedding_input(test_data_float, "zoq64")
            
            validate_encoding_method("eq64")
            validate_encoding_method("shq64")
            
            validate_method_parameters("shq64", planes=64)
            validate_method_parameters("t8q64", k=8)
            
            estimate_memory_usage(100, 512, "eq64")
        
        profiler.disable_by_count()
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            profiler.print_stats()
            profile_output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.results['validation_functions'] = profile_output
        
        with open(Path(__file__).parent / "line_profile_validation.txt", "w") as f:
            f.write(profile_output)
        
        print("Validation functions profiling complete. Results saved to line_profile_validation.txt")
    
    def profile_encoder_implementations(self):
        """Profile individual encoder implementations."""
        if not LINE_PROFILER_AVAILABLE:
            return
        
        print("Profiling encoder implementations...")
        
        from uubed.encoders import eq64, shq64, t8q64, zoq64
        
        profiler = LineProfiler()
        profiler.add_function(eq64.eq64_encode)
        profiler.add_function(shq64.simhash_q64)
        profiler.add_function(t8q64.top_k_q64)
        profiler.add_function(zoq64.z_order_q64)
        
        # Test data
        test_data = list(range(256))
        
        profiler.enable_by_count()
        
        for _ in range(50):
            eq64.eq64_encode(test_data)
            shq64.simhash_q64(test_data)
            t8q64.top_k_q64(test_data)
            zoq64.z_order_q64(test_data)
        
        profiler.disable_by_count()
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            profiler.print_stats()
            profile_output = mystdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        self.results['encoder_implementations'] = profile_output
        
        with open(Path(__file__).parent / "line_profile_encoders.txt", "w") as f:
            f.write(profile_output)
        
        print("Encoder implementations profiling complete. Results saved to line_profile_encoders.txt")
    
    def profile_function_call_overhead(self):
        """Profile function call overhead using cProfile."""
        if not CPROFILE_AVAILABLE:
            return
        
        print("Profiling function call overhead...")
        
        test_data = np.random.randint(0, 256, 512, dtype=np.uint8)
        
        # Profile with cProfile
        profiler = cProfile.Profile(builtins=False)
        
        profiler.enable()
        
        for _ in range(1000):
            uubed.encode(test_data, method="eq64")
        
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        
        # Get top function calls
        function_stats = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if cc > 0:  # Only include functions that were called
                function_stats.append({
                    'function': f"{func[0]}:{func[1]}({func[2]})",
                    'call_count': cc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'per_call_time': tt/cc,
                    'per_call_cumulative': ct/cc,
                    'filename': func[0],
                    'line_number': func[1],
                    'function_name': func[2]
                })
        
        # Sort by cumulative time
        function_stats.sort(key=lambda x: x['cumulative_time'], reverse=True)
        
        self.results['function_call_overhead'] = function_stats[:100]  # Top 100
        
        # Save detailed call overhead analysis
        with open(Path(__file__).parent / "function_call_overhead.json", "w") as f:
            json.dump(function_stats[:100], f, indent=2)
        
        print("Function call overhead profiling complete. Results saved to function_call_overhead.json")
    
    def identify_critical_paths(self):
        """Analyze results to identify critical performance paths."""
        print("Analyzing critical performance paths...")
        
        critical_paths = {
            'high_frequency_calls': [],
            'time_consuming_functions': [],
            'validation_bottlenecks': [],
            'encoding_bottlenecks': [],
            'recommendations': []
        }
        
        # Analyze function call overhead if available
        if 'function_call_overhead' in self.results:
            overhead_data = self.results['function_call_overhead']
            
            # Find high frequency calls
            high_freq = [f for f in overhead_data if f['call_count'] > 1000]
            high_freq.sort(key=lambda x: x['call_count'], reverse=True)
            critical_paths['high_frequency_calls'] = high_freq[:10]
            
            # Find time consuming functions
            time_consuming = [f for f in overhead_data if f['cumulative_time'] > 0.01]
            time_consuming.sort(key=lambda x: x['cumulative_time'], reverse=True)
            critical_paths['time_consuming_functions'] = time_consuming[:10]
            
            # Identify validation bottlenecks
            validation_funcs = [f for f in overhead_data if 'validation' in f['function'].lower()]
            validation_funcs.sort(key=lambda x: x['cumulative_time'], reverse=True)
            critical_paths['validation_bottlenecks'] = validation_funcs[:5]
            
            # Identify encoding bottlenecks
            encoding_funcs = [f for f in overhead_data if any(enc in f['function'].lower() 
                                                            for enc in ['encode', 'q64', 'simhash', 'topk', 'zorder'])]
            encoding_funcs.sort(key=lambda x: x['cumulative_time'], reverse=True)
            critical_paths['encoding_bottlenecks'] = encoding_funcs[:10]
        
        # Generate recommendations
        recommendations = []
        
        if critical_paths['validation_bottlenecks']:
            total_validation_time = sum(f['cumulative_time'] for f in critical_paths['validation_bottlenecks'])
            if total_validation_time > 0.1:  # 100ms
                recommendations.append(
                    f"Validation functions consume {total_validation_time:.3f}s total. "
                    "Consider caching validation results or optimizing validation logic."
                )
        
        if critical_paths['high_frequency_calls']:
            top_call = critical_paths['high_frequency_calls'][0]
            if top_call['call_count'] > 10000:
                recommendations.append(
                    f"Function {top_call['function_name']} called {top_call['call_count']} times. "
                    "Consider reducing call frequency or optimizing this function."
                )
        
        if critical_paths['time_consuming_functions']:
            top_time = critical_paths['time_consuming_functions'][0]
            if top_time['cumulative_time'] > 0.5:  # 500ms
                recommendations.append(
                    f"Function {top_time['function_name']} consumes {top_time['cumulative_time']:.3f}s. "
                    "This is a major performance bottleneck that should be optimized."
                )
        
        critical_paths['recommendations'] = recommendations
        
        # Save analysis
        with open(Path(__file__).parent / "critical_paths_analysis.json", "w") as f:
            json.dump(critical_paths, f, indent=2)
        
        print("Critical paths analysis complete. Results saved to critical_paths_analysis.json")
        
        return critical_paths
    
    def generate_hotpath_report(self):
        """Generate a comprehensive hot path report."""
        print("Generating hot path report...")
        
        critical_paths = self.identify_critical_paths()
        
        report = {
            'timestamp': time.time(),
            'profiler_available': {
                'line_profiler': LINE_PROFILER_AVAILABLE,
                'cprofile': CPROFILE_AVAILABLE
            },
            'critical_paths': critical_paths,
            'line_profiles': {
                key: value for key, value in self.results.items() 
                if key not in ['function_call_overhead']
            }
        }
        
        # Save report
        with open(Path(__file__).parent / "hotpath_analysis_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Hot path report saved to hotpath_analysis_report.json")
        
        return report


def print_hotpath_summary(report: Dict[str, Any]):
    """Print a summary of hot path analysis."""
    print("\n" + "="*80)
    print("UUBED HOT PATH ANALYSIS SUMMARY")
    print("="*80)
    
    critical_paths = report.get('critical_paths', {})
    
    print("\nHigh Frequency Function Calls:")
    high_freq = critical_paths.get('high_frequency_calls', [])
    for i, func in enumerate(high_freq[:5]):
        print(f"  {i+1}. {func['function_name']} - {func['call_count']} calls, {func['per_call_time']*1000:.3f}ms per call")
    
    print("\nMost Time-Consuming Functions:")
    time_consuming = critical_paths.get('time_consuming_functions', [])
    for i, func in enumerate(time_consuming[:5]):
        print(f"  {i+1}. {func['function_name']} - {func['cumulative_time']:.3f}s total, {func['per_call_time']*1000:.3f}ms per call")
    
    print("\nValidation Bottlenecks:")
    validation_bottlenecks = critical_paths.get('validation_bottlenecks', [])
    for i, func in enumerate(validation_bottlenecks[:3]):
        print(f"  {i+1}. {func['function_name']} - {func['cumulative_time']:.3f}s total")
    
    print("\nEncoding Bottlenecks:")
    encoding_bottlenecks = critical_paths.get('encoding_bottlenecks', [])
    for i, func in enumerate(encoding_bottlenecks[:3]):
        print(f"  {i+1}. {func['function_name']} - {func['cumulative_time']:.3f}s total")
    
    print("\nOptimization Recommendations:")
    recommendations = critical_paths.get('recommendations', [])
    for i, rec in enumerate(recommendations):
        print(f"  {i+1}. {rec}")
    
    print("\n" + "="*80)


def main():
    """Main hot path profiling entry point."""
    print("Starting hot path analysis for uubed...")
    
    profiler = HotPathProfiler()
    
    # Run all hot path profiling
    profiler.profile_api_encode_path()
    profiler.profile_streaming_encode_path()
    profiler.profile_validation_functions()
    profiler.profile_encoder_implementations()
    profiler.profile_function_call_overhead()
    
    # Generate report
    report = profiler.generate_hotpath_report()
    
    # Print summary
    print_hotpath_summary(report)
    
    print("\nHot path analysis complete! Check the generated files for detailed line-by-line profiles.")


if __name__ == "__main__":
    main()