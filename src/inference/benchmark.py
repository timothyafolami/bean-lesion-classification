"""
Performance benchmarking utilities for PyTorch vs ONNX models.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import onnxruntime as ort
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
import statistics
from dataclasses import dataclass

from src.training.models import load_model
from src.utils.logging_config import inference_logger
from src.utils.helpers import Timer, get_device, format_bytes, save_json


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_type: str
    model_path: str
    avg_inference_time: float
    min_inference_time: float
    max_inference_time: float
    std_inference_time: float
    throughput: float  # samples per second
    memory_usage: Optional[float] = None
    model_size: Optional[int] = None


class ModelBenchmark:
    """
    Benchmark PyTorch and ONNX models for performance comparison.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize benchmark utility.
        
        Args:
            device: Device to use for PyTorch benchmarking
        """
        self.device = device or get_device()
        inference_logger.info(f"Benchmark initialized with device: {self.device}")
    
    def benchmark_pytorch_model(
        self,
        model_path: str,
        architecture: str,
        test_data: np.ndarray,
        num_classes: int = 3,
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark PyTorch model performance.
        
        Args:
            model_path: Path to PyTorch model checkpoint
            architecture: Model architecture name
            test_data: Test data for benchmarking
            num_classes: Number of output classes
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        inference_logger.info(f"Benchmarking PyTorch model: {model_path}")
        
        # Load model
        model, _ = load_model(
            filepath=model_path,
            architecture=architecture,
            num_classes=num_classes,
            device=self.device
        )
        
        model.eval()
        
        # Convert test data to tensor
        test_tensor = torch.from_numpy(test_data).to(self.device)
        
        # Warmup runs
        inference_logger.info(f"Running {warmup_runs} warmup iterations...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(test_tensor)
        
        # Synchronize GPU if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark runs
        inference_logger.info(f"Running {benchmark_runs} benchmark iterations...")
        inference_times = []
        
        with torch.no_grad():
            for _ in range(benchmark_runs):
                start_time = time.perf_counter()
                _ = model(test_tensor)
                
                # Synchronize GPU if using CUDA
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        throughput = test_data.shape[0] / avg_time  # samples per second
        
        # Get model size
        model_size = Path(model_path).stat().st_size
        
        result = BenchmarkResult(
            model_type="PyTorch",
            model_path=model_path,
            avg_inference_time=avg_time,
            min_inference_time=min_time,
            max_inference_time=max_time,
            std_inference_time=std_time,
            throughput=throughput,
            model_size=model_size
        )
        
        inference_logger.info(f"PyTorch benchmark completed:")
        inference_logger.info(f"  Avg inference time: {avg_time*1000:.2f} ms")
        inference_logger.info(f"  Throughput: {throughput:.1f} samples/sec")
        
        return result
    
    def benchmark_onnx_model(
        self,
        model_path: str,
        test_data: np.ndarray,
        providers: Optional[List[str]] = None,
        warmup_runs: int = 10,
        benchmark_runs: int = 100
    ) -> BenchmarkResult:
        """
        Benchmark ONNX model performance.
        
        Args:
            model_path: Path to ONNX model
            test_data: Test data for benchmarking
            providers: ONNX Runtime providers to use
            warmup_runs: Number of warmup runs
            benchmark_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        inference_logger.info(f"Benchmarking ONNX model: {model_path}")
        
        # Set default providers
        if providers is None:
            if torch.cuda.is_available():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        inference_logger.info(f"Using providers: {providers}")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Warmup runs
        inference_logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            _ = session.run(None, {input_name: test_data})
        
        # Benchmark runs
        inference_logger.info(f"Running {benchmark_runs} benchmark iterations...")
        inference_times = []
        
        for _ in range(benchmark_runs):
            start_time = time.perf_counter()
            _ = session.run(None, {input_name: test_data})
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = statistics.mean(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0.0
        throughput = test_data.shape[0] / avg_time  # samples per second
        
        # Get model size
        model_size = Path(model_path).stat().st_size
        
        result = BenchmarkResult(
            model_type="ONNX",
            model_path=model_path,
            avg_inference_time=avg_time,
            min_inference_time=min_time,
            max_inference_time=max_time,
            std_inference_time=std_time,
            throughput=throughput,
            model_size=model_size
        )
        
        inference_logger.info(f"ONNX benchmark completed:")
        inference_logger.info(f"  Avg inference time: {avg_time*1000:.2f} ms")
        inference_logger.info(f"  Throughput: {throughput:.1f} samples/sec")
        
        return result
    
    def compare_models(
        self,
        pytorch_path: str,
        onnx_path: str,
        architecture: str,
        test_data: Optional[np.ndarray] = None,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_classes: int = 3
    ) -> Dict[str, Any]:
        """
        Compare PyTorch and ONNX model performance across different batch sizes.
        
        Args:
            pytorch_path: Path to PyTorch model
            onnx_path: Path to ONNX model
            architecture: Model architecture name
            test_data: Test data (if None, random data will be generated)
            batch_sizes: List of batch sizes to test
            num_classes: Number of output classes
            
        Returns:
            Comprehensive comparison results
        """
        inference_logger.info("Starting comprehensive model comparison...")
        
        results = {
            'pytorch_model': pytorch_path,
            'onnx_model': onnx_path,
            'architecture': architecture,
            'batch_comparisons': {},
            'summary': {}
        }
        
        for batch_size in batch_sizes:
            inference_logger.info(f"\n{'='*50}")
            inference_logger.info(f"Benchmarking batch size: {batch_size}")
            inference_logger.info(f"{'='*50}")
            
            # Generate test data if not provided
            if test_data is None or test_data.shape[0] != batch_size:
                current_test_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            else:
                current_test_data = test_data[:batch_size]
            
            # Benchmark PyTorch model
            pytorch_result = self.benchmark_pytorch_model(
                model_path=pytorch_path,
                architecture=architecture,
                test_data=current_test_data,
                num_classes=num_classes,
                warmup_runs=5,
                benchmark_runs=50
            )
            
            # Benchmark ONNX model
            onnx_result = self.benchmark_onnx_model(
                model_path=onnx_path,
                test_data=current_test_data,
                warmup_runs=5,
                benchmark_runs=50
            )
            
            # Calculate speedup
            speedup = pytorch_result.avg_inference_time / onnx_result.avg_inference_time
            throughput_improvement = (onnx_result.throughput - pytorch_result.throughput) / pytorch_result.throughput * 100
            
            batch_comparison = {
                'batch_size': batch_size,
                'pytorch': {
                    'avg_time_ms': pytorch_result.avg_inference_time * 1000,
                    'throughput': pytorch_result.throughput,
                    'model_size_mb': pytorch_result.model_size / (1024 * 1024)
                },
                'onnx': {
                    'avg_time_ms': onnx_result.avg_inference_time * 1000,
                    'throughput': onnx_result.throughput,
                    'model_size_mb': onnx_result.model_size / (1024 * 1024)
                },
                'speedup': speedup,
                'throughput_improvement_percent': throughput_improvement
            }
            
            results['batch_comparisons'][batch_size] = batch_comparison
            
            inference_logger.info(f"Batch {batch_size} Results:")
            inference_logger.info(f"  PyTorch: {pytorch_result.avg_inference_time*1000:.2f} ms")
            inference_logger.info(f"  ONNX: {onnx_result.avg_inference_time*1000:.2f} ms")
            inference_logger.info(f"  Speedup: {speedup:.2f}x")
            inference_logger.info(f"  Throughput improvement: {throughput_improvement:.1f}%")
        
        # Calculate summary statistics
        speedups = [comp['speedup'] for comp in results['batch_comparisons'].values()]
        throughput_improvements = [comp['throughput_improvement_percent'] for comp in results['batch_comparisons'].values()]
        
        results['summary'] = {
            'avg_speedup': statistics.mean(speedups),
            'min_speedup': min(speedups),
            'max_speedup': max(speedups),
            'avg_throughput_improvement': statistics.mean(throughput_improvements),
            'pytorch_model_size_mb': pytorch_result.model_size / (1024 * 1024),
            'onnx_model_size_mb': onnx_result.model_size / (1024 * 1024),
            'model_size_reduction_percent': (pytorch_result.model_size - onnx_result.model_size) / pytorch_result.model_size * 100
        }
        
        # Log summary
        inference_logger.info(f"\n{'='*60}")
        inference_logger.info("BENCHMARK SUMMARY")
        inference_logger.info(f"{'='*60}")
        inference_logger.info(f"Average speedup: {results['summary']['avg_speedup']:.2f}x")
        inference_logger.info(f"Best speedup: {results['summary']['max_speedup']:.2f}x")
        inference_logger.info(f"Average throughput improvement: {results['summary']['avg_throughput_improvement']:.1f}%")
        inference_logger.info(f"Model size reduction: {results['summary']['model_size_reduction_percent']:.1f}%")
        
        return results
    
    def save_benchmark_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save benchmark results to JSON file.
        
        Args:
            results: Benchmark results dictionary
            output_path: Path to save results
        """
        save_json(results, output_path)
        inference_logger.info(f"Benchmark results saved to: {output_path}")
    
    def create_benchmark_visualization(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualization of benchmark results.
        
        Args:
            results: Benchmark results dictionary
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Extract data for plotting
            batch_sizes = list(results['batch_comparisons'].keys())
            pytorch_times = [results['batch_comparisons'][bs]['pytorch']['avg_time_ms'] for bs in batch_sizes]
            onnx_times = [results['batch_comparisons'][bs]['onnx']['avg_time_ms'] for bs in batch_sizes]
            speedups = [results['batch_comparisons'][bs]['speedup'] for bs in batch_sizes]
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Inference times
            x = np.arange(len(batch_sizes))
            width = 0.35
            
            ax1.bar(x - width/2, pytorch_times, width, label='PyTorch', alpha=0.8, color='blue')
            ax1.bar(x + width/2, onnx_times, width, label='ONNX', alpha=0.8, color='orange')
            
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Inference Time (ms)')
            ax1.set_title('Inference Time Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(batch_sizes)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Speedup
            ax2.plot(batch_sizes, speedups, marker='o', linewidth=2, markersize=8, color='green')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Speedup (x)')
            ax2.set_title('ONNX Speedup vs PyTorch')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                inference_logger.info(f"Benchmark visualization saved to: {save_path}")
            
            plt.show()
            
        except ImportError:
            inference_logger.warning("Matplotlib not available, skipping visualization")
        except Exception as e:
            inference_logger.error(f"Error creating benchmark visualization: {e}")


def benchmark_model_conversion(
    pytorch_path: str,
    onnx_path: str,
    architecture: str,
    output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """
    Convenience function to benchmark PyTorch vs ONNX model performance.
    
    Args:
        pytorch_path: Path to PyTorch model
        onnx_path: Path to ONNX model
        architecture: Model architecture name
        output_dir: Directory to save results
        
    Returns:
        Benchmark results
    """
    benchmark = ModelBenchmark()
    
    # Run comprehensive comparison
    results = benchmark.compare_models(
        pytorch_path=pytorch_path,
        onnx_path=onnx_path,
        architecture=architecture,
        batch_sizes=[1, 4, 8, 16, 32]
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    benchmark.save_benchmark_results(
        results,
        str(output_path / "benchmark_results.json")
    )
    
    benchmark.create_benchmark_visualization(
        results,
        str(output_path / "benchmark_comparison.png")
    )
    
    return results