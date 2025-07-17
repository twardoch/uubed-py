#!/usr/bin/env python3
"""
Memory profiling script for uubed-py.

This script provides detailed analysis of memory usage patterns,
allocation overhead, and memory leaks in the uubed library.
"""

import sys
import time
import gc
import tracemalloc
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import json

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uubed
from uubed.native_wrapper import is_native_available

try:
    from memory_profiler import profile as memory_profile, memory_usage
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    print("memory_profiler not available. Install with: pip install memory-profiler")


class MemoryProfiler:
    """Specialized memory profiler for uubed operations."""
    
    def __init__(self):
        self.results = {}
        self.baseline_memory = None
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def establish_baseline(self):
        """Establish baseline memory usage."""
        gc.collect()  # Force garbage collection
        time.sleep(0.1)  # Allow system to settle
        self.baseline_memory = self.get_memory_info()
        print(f"Baseline memory: {self.baseline_memory['rss_mb']:.2f} MB RSS")
    
    def profile_encoding_memory_scaling(self):
        """Profile how memory usage scales with input size."""
        print("Profiling memory scaling with input size...")
        
        sizes = [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 4096]
        methods = ['eq64', 'shq64', 't8q64', 'zoq64']
        
        scaling_results = {}
        
        for method in methods:
            scaling_results[method] = []
            
            for size in sizes:
                print(f"  Testing {method} with size {size}")
                
                # Force garbage collection before test
                gc.collect()
                
                # Generate test data
                test_data = np.random.randint(0, 256, size, dtype=np.uint8)
                
                # Start memory tracking
                tracemalloc.start()
                start_memory = self.get_memory_info()
                
                # Perform encoding
                start_time = time.perf_counter()
                encoded = uubed.encode(test_data, method=method)
                end_time = time.perf_counter()
                
                # Get memory usage
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                end_memory = self.get_memory_info()
                
                # Calculate memory growth
                memory_growth = end_memory['rss_mb'] - start_memory['rss_mb']
                
                result = {
                    'input_size': size,
                    'encoding_time': end_time - start_time,
                    'output_length': len(encoded),
                    'memory_growth_mb': memory_growth,
                    'tracemalloc_current_mb': current / (1024 * 1024),
                    'tracemalloc_peak_mb': peak / (1024 * 1024),
                    'compression_ratio': len(encoded) / size,
                    'memory_efficiency': size / peak if peak > 0 else 0  # bytes processed per byte allocated
                }
                
                scaling_results[method].append(result)
                
                # Clean up
                del encoded, test_data
                gc.collect()
        
        self.results['memory_scaling'] = scaling_results
        
        # Save detailed scaling analysis
        with open(Path(__file__).parent / "memory_scaling_analysis.json", "w") as f:
            json.dump(scaling_results, f, indent=2)
        
        print("Memory scaling analysis complete.")
        return scaling_results
    
    def profile_batch_vs_streaming_memory(self):
        """Compare memory usage between batch and streaming operations."""
        print("Profiling batch vs streaming memory usage...")
        
        embedding_counts = [100, 500, 1000, 2000, 5000]
        embedding_size = 512
        
        comparison_results = {}
        
        for count in embedding_counts:
            print(f"  Testing with {count} embeddings")
            
            # Generate test data
            test_embeddings = [
                np.random.randint(0, 256, embedding_size, dtype=np.uint8)
                for _ in range(count)
            ]
            
            comparison_results[count] = {}
            
            # Test batch encoding
            gc.collect()
            tracemalloc.start()
            start_memory = self.get_memory_info()
            
            start_time = time.perf_counter()
            batch_results = uubed.batch_encode(test_embeddings, method="eq64")
            end_time = time.perf_counter()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_memory = self.get_memory_info()
            
            comparison_results[count]['batch'] = {
                'time': end_time - start_time,
                'memory_growth_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
                'tracemalloc_peak_mb': peak / (1024 * 1024),
                'results_count': len(batch_results)
            }
            
            # Clean up
            del batch_results
            gc.collect()
            
            # Test streaming encoding
            tracemalloc.start()
            start_memory = self.get_memory_info()
            
            start_time = time.perf_counter()
            streaming_results = list(uubed.encode_stream(test_embeddings, method="eq64", batch_size=100))
            end_time = time.perf_counter()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end_memory = self.get_memory_info()
            
            comparison_results[count]['streaming'] = {
                'time': end_time - start_time,
                'memory_growth_mb': end_memory['rss_mb'] - start_memory['rss_mb'],
                'tracemalloc_peak_mb': peak / (1024 * 1024),
                'results_count': len(streaming_results)
            }
            
            # Calculate memory efficiency
            batch_peak = comparison_results[count]['batch']['tracemalloc_peak_mb']
            streaming_peak = comparison_results[count]['streaming']['tracemalloc_peak_mb']
            
            if streaming_peak > 0:
                comparison_results[count]['memory_reduction_factor'] = batch_peak / streaming_peak
            else:
                comparison_results[count]['memory_reduction_factor'] = 0
            
            # Clean up
            del streaming_results, test_embeddings
            gc.collect()
        
        self.results['batch_vs_streaming'] = comparison_results
        
        with open(Path(__file__).parent / "batch_vs_streaming_memory.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print("Batch vs streaming memory analysis complete.")
        return comparison_results
    
    def profile_memory_leaks(self):
        """Test for memory leaks during repeated operations."""
        print("Profiling for memory leaks...")
        
        test_data = np.random.randint(0, 256, 512, dtype=np.uint8)
        
        leak_test_results = {}
        
        for method in ['eq64', 'shq64', 't8q64']:
            print(f"  Testing {method} for memory leaks")
            
            memory_snapshots = []
            iterations = 1000
            
            # Force garbage collection before starting
            gc.collect()
            baseline = self.get_memory_info()
            
            for i in range(iterations):
                # Encode data
                encoded = uubed.encode(test_data, method=method)
                
                # Take memory snapshots every 100 iterations
                if i % 100 == 0:
                    if i > 0:  # Skip first iteration to allow for initialization
                        gc.collect()  # Force garbage collection
                        memory_info = self.get_memory_info()
                        memory_snapshots.append({
                            'iteration': i,
                            'rss_mb': memory_info['rss_mb'],
                            'rss_growth_mb': memory_info['rss_mb'] - baseline['rss_mb']
                        })
                
                # Clean up explicitly
                del encoded
            
            # Final garbage collection and measurement
            gc.collect()
            final_memory = self.get_memory_info()
            
            # Analyze memory growth trend
            if len(memory_snapshots) > 1:
                initial_growth = memory_snapshots[0]['rss_growth_mb']
                final_growth = memory_snapshots[-1]['rss_growth_mb']
                growth_trend = final_growth - initial_growth
            else:
                growth_trend = 0
            
            leak_test_results[method] = {
                'iterations': iterations,
                'baseline_mb': baseline['rss_mb'],
                'final_mb': final_memory['rss_mb'],
                'total_growth_mb': final_memory['rss_mb'] - baseline['rss_mb'],
                'growth_trend_mb': growth_trend,
                'snapshots': memory_snapshots,
                'potential_leak': growth_trend > 1.0  # Flag if growth > 1MB
            }
        
        self.results['memory_leaks'] = leak_test_results
        
        with open(Path(__file__).parent / "memory_leak_analysis.json", "w") as f:
            json.dump(leak_test_results, f, indent=2)
        
        print("Memory leak analysis complete.")
        return leak_test_results
    
    def profile_allocation_patterns(self):
        """Profile memory allocation patterns using tracemalloc."""
        print("Profiling memory allocation patterns...")
        
        allocation_results = {}
        
        for method in ['eq64', 'shq64', 't8q64']:
            print(f"  Analyzing allocation patterns for {method}")
            
            test_data = np.random.randint(0, 256, 512, dtype=np.uint8)
            
            # Start detailed tracking
            tracemalloc.start()
            
            # Perform multiple operations to see allocation patterns
            for _ in range(10):
                encoded = uubed.encode(test_data, method=method)
                del encoded
            
            # Take snapshot
            snapshot = tracemalloc.take_snapshot()
            
            # Get top allocations
            top_stats = snapshot.statistics('lineno')
            
            tracemalloc.stop()
            
            # Process allocation data
            allocations = []
            for stat in top_stats[:20]:  # Top 20 allocations
                allocations.append({
                    'file': stat.traceback.format()[-1] if stat.traceback.format() else 'unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count,
                    'average_size_bytes': stat.size / stat.count if stat.count > 0 else 0
                })
            
            allocation_results[method] = {
                'total_allocations': len(top_stats),
                'top_allocations': allocations,
                'total_allocated_mb': sum(stat.size for stat in top_stats) / (1024 * 1024)
            }
        
        self.results['allocation_patterns'] = allocation_results
        
        with open(Path(__file__).parent / "allocation_patterns.json", "w") as f:
            json.dump(allocation_results, f, indent=2)
        
        print("Allocation patterns analysis complete.")
        return allocation_results
    
    def profile_validation_memory_overhead(self):
        """Profile memory overhead of validation functions."""
        print("Profiling validation memory overhead...")
        
        validation_results = {}
        
        # Test different input types and sizes
        test_cases = {
            'small_list': list(range(64)),
            'medium_list': list(range(512)),
            'large_list': list(range(2048)),
            'small_bytes': bytes(range(64)),
            'medium_bytes': bytes(range(256)) * 2,  # 512 bytes
            'large_bytes': bytes(range(256)) * 8,   # 2048 bytes
            'small_numpy': np.random.randint(0, 256, 64, dtype=np.uint8),
            'medium_numpy': np.random.randint(0, 256, 512, dtype=np.uint8),
            'large_numpy': np.random.randint(0, 256, 2048, dtype=np.uint8),
            'float_numpy': np.random.random(512).astype(np.float32)
        }
        
        from uubed.validation import validate_embedding_input
        
        for case_name, test_data in test_cases.items():
            print(f"  Testing validation overhead for {case_name}")
            
            # Measure validation overhead
            gc.collect()
            tracemalloc.start()
            
            start_time = time.perf_counter()
            for _ in range(100):
                validated = validate_embedding_input(test_data, "eq64")
            end_time = time.perf_counter()
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            input_size = len(test_data) if hasattr(test_data, '__len__') else test_data.size
            
            validation_results[case_name] = {
                'input_type': type(test_data).__name__,
                'input_size': input_size,
                'validation_time': end_time - start_time,
                'memory_peak_mb': peak / (1024 * 1024),
                'memory_per_validation_kb': (peak / 100) / 1024,  # Per validation call
                'time_per_validation_us': ((end_time - start_time) / 100) * 1000000  # microseconds
            }
        
        self.results['validation_overhead'] = validation_results
        
        with open(Path(__file__).parent / "validation_memory_overhead.json", "w") as f:
            json.dump(validation_results, f, indent=2)
        
        print("Validation memory overhead analysis complete.")
        return validation_results
    
    def analyze_memory_efficiency(self):
        """Analyze overall memory efficiency patterns."""
        print("Analyzing memory efficiency patterns...")
        
        efficiency_analysis = {}
        
        # Analyze scaling efficiency
        if 'memory_scaling' in self.results:
            scaling_data = self.results['memory_scaling']
            
            for method, results in scaling_data.items():
                if results:
                    # Calculate efficiency metrics
                    sizes = [r['input_size'] for r in results]
                    memory_usage = [r['tracemalloc_peak_mb'] for r in results]
                    
                    # Memory efficiency (input bytes per MB of peak memory)
                    efficiency_ratios = []
                    for i, result in enumerate(results):
                        if result['tracemalloc_peak_mb'] > 0:
                            ratio = result['input_size'] / result['tracemalloc_peak_mb']
                            efficiency_ratios.append(ratio)
                    
                    if efficiency_ratios:
                        efficiency_analysis[f"{method}_memory_efficiency"] = {
                            'avg_efficiency': np.mean(efficiency_ratios),
                            'min_efficiency': np.min(efficiency_ratios),
                            'max_efficiency': np.max(efficiency_ratios),
                            'efficiency_trend': 'improving' if len(efficiency_ratios) > 1 and efficiency_ratios[-1] > efficiency_ratios[0] else 'declining'
                        }
        
        # Analyze batch vs streaming efficiency
        if 'batch_vs_streaming' in self.results:
            batch_streaming_data = self.results['batch_vs_streaming']
            
            memory_reductions = []
            for count, data in batch_streaming_data.items():
                if 'memory_reduction_factor' in data:
                    memory_reductions.append(data['memory_reduction_factor'])
            
            if memory_reductions:
                efficiency_analysis['streaming_vs_batch'] = {
                    'avg_memory_reduction': np.mean(memory_reductions),
                    'max_memory_reduction': np.max(memory_reductions),
                    'consistent_improvement': all(r > 1.0 for r in memory_reductions)
                }
        
        # Analyze potential memory leaks
        if 'memory_leaks' in self.results:
            leak_data = self.results['memory_leaks']
            
            methods_with_leaks = []
            for method, data in leak_data.items():
                if data.get('potential_leak', False):
                    methods_with_leaks.append(method)
            
            efficiency_analysis['memory_leak_assessment'] = {
                'methods_with_potential_leaks': methods_with_leaks,
                'leak_free_methods': [m for m in leak_data.keys() if m not in methods_with_leaks]
            }
        
        self.results['efficiency_analysis'] = efficiency_analysis
        
        with open(Path(__file__).parent / "memory_efficiency_analysis.json", "w") as f:
            json.dump(efficiency_analysis, f, indent=2)
        
        print("Memory efficiency analysis complete.")
        return efficiency_analysis
    
    def generate_memory_report(self):
        """Generate comprehensive memory profiling report."""
        print("Generating memory profiling report...")
        
        # Perform efficiency analysis
        efficiency = self.analyze_memory_efficiency()
        
        # Create comprehensive report
        report = {
            'timestamp': time.time(),
            'baseline_memory': self.baseline_memory,
            'final_memory': self.get_memory_info(),
            'native_available': is_native_available(),
            'memory_profiler_available': MEMORY_PROFILER_AVAILABLE,
            'detailed_results': self.results,
            'efficiency_analysis': efficiency,
            'recommendations': self._generate_memory_recommendations()
        }
        
        # Save report
        with open(Path(__file__).parent / "memory_profiling_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Memory profiling report saved to memory_profiling_report.json")
        
        return report
    
    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Check for memory leaks
        if 'memory_leaks' in self.results:
            leak_data = self.results['memory_leaks']
            for method, data in leak_data.items():
                if data.get('potential_leak', False):
                    recommendations.append(
                        f"Potential memory leak detected in {method} method. "
                        f"Memory grew by {data['growth_trend_mb']:.2f}MB over {data['iterations']} iterations."
                    )
        
        # Check batch vs streaming efficiency
        if 'batch_vs_streaming' in self.results:
            batch_streaming_data = self.results['batch_vs_streaming']
            for count, data in batch_streaming_data.items():
                reduction_factor = data.get('memory_reduction_factor', 1)
                if reduction_factor > 2:  # Streaming uses less than half the memory
                    recommendations.append(
                        f"For {count} embeddings, streaming reduces memory usage by {reduction_factor:.1f}x. "
                        "Consider using streaming for large datasets."
                    )
        
        # Check validation overhead
        if 'validation_overhead' in self.results:
            validation_data = self.results['validation_overhead']
            high_overhead_cases = []
            for case, data in validation_data.items():
                if data['memory_per_validation_kb'] > 10:  # More than 10KB per validation
                    high_overhead_cases.append(case)
            
            if high_overhead_cases:
                recommendations.append(
                    f"High validation memory overhead detected for: {', '.join(high_overhead_cases)}. "
                    "Consider optimizing validation for these input types."
                )
        
        # Check memory scaling efficiency
        if 'memory_scaling' in self.results:
            scaling_data = self.results['memory_scaling']
            for method, results in scaling_data.items():
                if results:
                    # Find cases where memory usage grows faster than input size
                    inefficient_sizes = []
                    for result in results:
                        if result['memory_efficiency'] < 10:  # Less than 10 bytes per byte allocated
                            inefficient_sizes.append(result['input_size'])
                    
                    if inefficient_sizes:
                        recommendations.append(
                            f"Method {method} shows poor memory efficiency for sizes: {inefficient_sizes}. "
                            "Consider alternative approaches for these input sizes."
                        )
        
        if not recommendations:
            recommendations.append("No significant memory issues detected. Memory usage appears optimal.")
        
        return recommendations


def print_memory_summary(report: Dict[str, Any]):
    """Print a summary of memory profiling results."""
    print("\n" + "="*80)
    print("UUBED MEMORY PROFILING SUMMARY")
    print("="*80)
    
    baseline = report.get('baseline_memory', {})
    final = report.get('final_memory', {})
    
    print(f"\nMemory Usage:")
    print(f"  Baseline: {baseline.get('rss_mb', 0):.2f} MB RSS")
    print(f"  Final: {final.get('rss_mb', 0):.2f} MB RSS")
    print(f"  Growth: {final.get('rss_mb', 0) - baseline.get('rss_mb', 0):.2f} MB")
    
    # Print efficiency analysis
    efficiency = report.get('efficiency_analysis', {})
    
    print(f"\nMemory Efficiency Analysis:")
    for method, data in efficiency.items():
        if isinstance(data, dict) and 'avg_efficiency' in data:
            print(f"  {method}: {data['avg_efficiency']:.1f} bytes/MB (avg)")
    
    # Print leak assessment
    leak_assessment = efficiency.get('memory_leak_assessment', {})
    if leak_assessment:
        leak_methods = leak_assessment.get('methods_with_potential_leaks', [])
        if leak_methods:
            print(f"\nPotential Memory Leaks Detected:")
            for method in leak_methods:
                print(f"  - {method}")
        else:
            print(f"\nNo memory leaks detected.")
    
    # Print streaming vs batch analysis
    streaming_analysis = efficiency.get('streaming_vs_batch', {})
    if streaming_analysis:
        reduction = streaming_analysis.get('avg_memory_reduction', 1)
        print(f"\nStreaming vs Batch:")
        print(f"  Average memory reduction: {reduction:.1f}x")
        print(f"  Consistent improvement: {streaming_analysis.get('consistent_improvement', False)}")
    
    # Print recommendations
    recommendations = report.get('recommendations', [])
    print(f"\nOptimization Recommendations:")
    for i, rec in enumerate(recommendations[:5]):  # Show top 5
        print(f"  {i+1}. {rec}")
    
    print("\n" + "="*80)


def main():
    """Main memory profiling entry point."""
    print("Starting comprehensive memory profiling for uubed...")
    
    profiler = MemoryProfiler()
    
    # Establish baseline
    profiler.establish_baseline()
    
    # Run all memory profiling operations
    profiler.profile_encoding_memory_scaling()
    profiler.profile_batch_vs_streaming_memory()
    profiler.profile_memory_leaks()
    profiler.profile_allocation_patterns()
    profiler.profile_validation_memory_overhead()
    
    # Generate report
    report = profiler.generate_memory_report()
    
    # Print summary
    print_memory_summary(report)
    
    print("\nMemory profiling complete! Check the generated JSON files for detailed analysis.")


if __name__ == "__main__":
    main()