"""
Script to convert all trained models to ONNX format and compare their performance.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import subprocess
from datetime import datetime

from src.utils.logging_config import setup_logging, inference_logger
from src.utils.helpers import save_json


def find_trained_models(models_dir: str = "models"):
    """Find all trained models with their configurations."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        inference_logger.error(f"Models directory not found: {models_path}")
        return []
    
    trained_models = []
    
    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            # Look for model files
            model_file = model_dir / "best_model.pth"
            config_file = model_dir / "training_config.json"
            
            if model_file.exists() and config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    trained_models.append({
                        'name': model_dir.name,
                        'model_path': str(model_file),
                        'architecture': config['architecture'],
                        'config': config
                    })
                except Exception as e:
                    inference_logger.warning(f"Could not load config for {model_dir}: {e}")
    
    return trained_models


def convert_model(model_info, output_dir="onnx_models", benchmark=True):
    """Convert a single model to ONNX."""
    inference_logger.info(f"Converting {model_info['name']} ({model_info['architecture']})...")
    
    # Build conversion command
    cmd = [
        sys.executable, 'convert_to_onnx.py',
        '--model-path', model_info['model_path'],
        '--architecture', model_info['architecture'],
        '--output-dir', output_dir,
        '--optimize'
    ]
    
    if benchmark:
        cmd.append('--benchmark')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        inference_logger.info(f"✅ {model_info['name']} conversion completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        inference_logger.error(f"❌ {model_info['name']} conversion failed")
        inference_logger.error(f"Error: {e.stderr}")
        return False, e.stderr


def compare_onnx_models(onnx_dir="onnx_models"):
    """Compare all converted ONNX models."""
    onnx_path = Path(onnx_dir)
    
    if not onnx_path.exists():
        inference_logger.warning("No ONNX models directory found")
        return {}
    
    # Find all benchmark results
    benchmark_results = {}
    
    for benchmark_dir in onnx_path.glob("**/benchmarks"):
        results_file = benchmark_dir / "benchmark_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Extract model name from path
                model_name = results['architecture']
                benchmark_results[model_name] = results['summary']
                
            except Exception as e:
                inference_logger.warning(f"Could not load benchmark results from {results_file}: {e}")
    
    return benchmark_results


def create_comparison_report(benchmark_results, output_path="onnx_models/model_comparison.json"):
    """Create a comprehensive comparison report."""
    if not benchmark_results:
        inference_logger.warning("No benchmark results to compare")
        return
    
    # Sort models by average speedup
    sorted_models = sorted(
        benchmark_results.items(),
        key=lambda x: x[1]['avg_speedup'],
        reverse=True
    )
    
    comparison_report = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(benchmark_results),
        'model_rankings': [],
        'summary_statistics': {}
    }
    
    # Create rankings
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        comparison_report['model_rankings'].append({
            'rank': rank,
            'model': model_name,
            'avg_speedup': results['avg_speedup'],
            'max_speedup': results['max_speedup'],
            'avg_throughput_improvement': results['avg_throughput_improvement'],
            'model_size_reduction_percent': results['model_size_reduction_percent'],
            'pytorch_model_size_mb': results['pytorch_model_size_mb'],
            'onnx_model_size_mb': results['onnx_model_size_mb']
        })
    
    # Calculate summary statistics
    speedups = [results['avg_speedup'] for results in benchmark_results.values()]
    throughput_improvements = [results['avg_throughput_improvement'] for results in benchmark_results.values()]
    size_reductions = [results['model_size_reduction_percent'] for results in benchmark_results.values()]
    
    comparison_report['summary_statistics'] = {
        'avg_speedup_across_models': sum(speedups) / len(speedups),
        'best_speedup': max(speedups),
        'avg_throughput_improvement': sum(throughput_improvements) / len(throughput_improvements),
        'avg_size_reduction': sum(size_reductions) / len(size_reductions)
    }
    
    # Save report
    save_json(comparison_report, output_path)
    inference_logger.info(f"Comparison report saved to: {output_path}")
    
    return comparison_report


def print_comparison_summary(comparison_report):
    """Print a summary of the model comparison."""
    if not comparison_report:
        return
    
    inference_logger.info("\n" + "="*80)
    inference_logger.info("🏆 ONNX MODEL COMPARISON SUMMARY")
    inference_logger.info("="*80)
    
    # Print rankings
    inference_logger.info(f"{'Rank':<4} {'Model':<20} {'Speedup':<8} {'Throughput':<12} {'Size Reduction':<15}")
    inference_logger.info("-" * 80)
    
    for model_info in comparison_report['model_rankings']:
        rank = model_info['rank']
        model = model_info['model']
        speedup = model_info['avg_speedup']
        throughput = model_info['avg_throughput_improvement']
        size_reduction = model_info['model_size_reduction_percent']
        
        inference_logger.info(f"{rank:<4} {model:<20} {speedup:.2f}x     {throughput:+.1f}%       {size_reduction:+.1f}%")
    
    # Print summary statistics
    stats = comparison_report['summary_statistics']
    inference_logger.info(f"\n📊 OVERALL STATISTICS:")
    inference_logger.info(f"  Average speedup across all models: {stats['avg_speedup_across_models']:.2f}x")
    inference_logger.info(f"  Best speedup achieved: {stats['best_speedup']:.2f}x")
    inference_logger.info(f"  Average throughput improvement: {stats['avg_throughput_improvement']:.1f}%")
    inference_logger.info(f"  Average model size reduction: {stats['avg_size_reduction']:.1f}%")
    
    # Recommendations
    best_model = comparison_report['model_rankings'][0]
    inference_logger.info(f"\n💡 RECOMMENDATION:")
    inference_logger.info(f"  Best performing model: {best_model['model']}")
    inference_logger.info(f"  Speedup: {best_model['avg_speedup']:.2f}x")
    inference_logger.info(f"  Use this model for production deployment!")


def main():
    """Main function to convert all models and compare performance."""
    setup_logging(log_level="INFO")
    
    inference_logger.info("="*60)
    inference_logger.info("🔄 BATCH ONNX CONVERSION")
    inference_logger.info("="*60)
    
    # Find all trained models
    trained_models = find_trained_models()
    
    if not trained_models:
        inference_logger.error("No trained models found!")
        inference_logger.info("Train some models first using 'python train_model.py'")
        return False
    
    inference_logger.info(f"Found {len(trained_models)} trained models:")
    for model in trained_models:
        inference_logger.info(f"  - {model['name']} ({model['architecture']})")
    
    # Convert all models
    conversion_results = {}
    successful_conversions = 0
    
    for model_info in trained_models:
        inference_logger.info(f"\n{'='*50}")
        inference_logger.info(f"Converting: {model_info['name']}")
        inference_logger.info(f"{'='*50}")
        
        success, output = convert_model(model_info, benchmark=True)
        conversion_results[model_info['name']] = {
            'success': success,
            'architecture': model_info['architecture'],
            'output': output
        }
        
        if success:
            successful_conversions += 1
    
    # Summary of conversions
    inference_logger.info(f"\n{'='*60}")
    inference_logger.info("CONVERSION SUMMARY")
    inference_logger.info(f"{'='*60}")
    inference_logger.info(f"Successful conversions: {successful_conversions}/{len(trained_models)}")
    
    for model_name, result in conversion_results.items():
        status = "✅" if result['success'] else "❌"
        inference_logger.info(f"{status} {model_name} ({result['architecture']})")
    
    if successful_conversions == 0:
        inference_logger.error("No models were successfully converted!")
        return False
    
    # Compare benchmark results
    inference_logger.info("\n🔍 Analyzing benchmark results...")
    benchmark_results = compare_onnx_models()
    
    if benchmark_results:
        comparison_report = create_comparison_report(benchmark_results)
        print_comparison_summary(comparison_report)
    else:
        inference_logger.warning("No benchmark results found for comparison")
    
    # Final message
    inference_logger.info(f"\n🎉 Batch conversion completed!")
    inference_logger.info(f"✅ {successful_conversions} models successfully converted to ONNX")
    inference_logger.info("📁 Check the 'onnx_models/' directory for converted models")
    inference_logger.info("🚀 Models are ready for FastAPI integration!")
    
    return successful_conversions > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)