"""
Test script for ONNX conversion pipeline.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.inference.onnx_converter import ONNXConverter
from src.inference.benchmark import ModelBenchmark
from src.training.models import ModelFactory
from src.utils.logging_config import setup_logging, inference_logger


def test_onnx_converter():
    """Test basic ONNX conversion functionality."""
    inference_logger.info("Testing ONNX converter...")
    
    try:
        # Create a simple test model
        model = ModelFactory.create_model(
            architecture='resnet18',
            num_classes=3,
            pretrained=False
        )
        
        # Initialize converter
        converter = ONNXConverter()
        
        # Test conversion
        test_output_path = "test_model.onnx"
        onnx_path = converter.convert_pytorch_to_onnx(
            model=model,
            input_shape=(1, 3, 224, 224),
            output_path=test_output_path
        )
        
        # Test validation
        validation_results = converter.validate_conversion(model, onnx_path)
        
        # Clean up
        if Path(test_output_path).exists():
            Path(test_output_path).unlink()
        
        if validation_results['validation_passed']:
            inference_logger.info("✅ ONNX converter test passed")
            return True
        else:
            inference_logger.error("❌ ONNX validation failed")
            return False
            
    except Exception as e:
        inference_logger.error(f"ONNX converter test failed: {e}")
        return False


def test_model_benchmark():
    """Test model benchmarking functionality."""
    inference_logger.info("Testing model benchmark...")
    
    try:
        # Create test data
        test_data = np.random.randn(4, 3, 224, 224).astype(np.float32)
        
        # Create a simple ONNX model for testing
        model = ModelFactory.create_model(
            architecture='resnet18',
            num_classes=3,
            pretrained=False
        )
        
        converter = ONNXConverter()
        test_onnx_path = "test_benchmark_model.onnx"
        
        converter.convert_pytorch_to_onnx(
            model=model,
            input_shape=(4, 3, 224, 224),
            output_path=test_onnx_path
        )
        
        # Test ONNX benchmarking
        benchmark = ModelBenchmark()
        
        onnx_result = benchmark.benchmark_onnx_model(
            model_path=test_onnx_path,
            test_data=test_data,
            warmup_runs=2,
            benchmark_runs=5
        )
        
        # Clean up
        if Path(test_onnx_path).exists():
            Path(test_onnx_path).unlink()
        
        if onnx_result.avg_inference_time > 0:
            inference_logger.info("✅ Model benchmark test passed")
            inference_logger.info(f"  Avg inference time: {onnx_result.avg_inference_time*1000:.2f} ms")
            return True
        else:
            inference_logger.error("❌ Benchmark test failed")
            return False
            
    except Exception as e:
        inference_logger.error(f"Model benchmark test failed: {e}")
        return False


def test_optimization():
    """Test ONNX model optimization."""
    inference_logger.info("Testing ONNX optimization...")
    
    try:
        # Create a test model
        model = ModelFactory.create_model(
            architecture='resnet18',
            num_classes=3,
            pretrained=False
        )
        
        converter = ONNXConverter()
        
        # Convert to ONNX
        original_path = "test_original.onnx"
        converter.convert_pytorch_to_onnx(
            model=model,
            input_shape=(1, 3, 224, 224),
            output_path=original_path
        )
        
        # Optimize the model
        optimized_path = converter.optimize_onnx_model(
            onnx_path=original_path,
            optimization_level="basic"
        )
        
        # Check that both files exist
        original_exists = Path(original_path).exists()
        optimized_exists = Path(optimized_path).exists()
        
        # Clean up
        for path in [original_path, optimized_path]:
            if Path(path).exists():
                Path(path).unlink()
        
        if original_exists and optimized_exists:
            inference_logger.info("✅ ONNX optimization test passed")
            return True
        else:
            inference_logger.error("❌ ONNX optimization test failed")
            return False
            
    except Exception as e:
        inference_logger.error(f"ONNX optimization test failed: {e}")
        return False


def test_with_trained_model():
    """Test conversion with an actual trained model if available."""
    inference_logger.info("Testing with trained model...")
    
    # Look for trained models
    models_dir = Path("models")
    if not models_dir.exists():
        inference_logger.info("No models directory found, skipping trained model test")
        return True
    
    # Find the first available trained model
    trained_model = None
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_file = model_dir / "best_model.pth"
            config_file = model_dir / "training_config.json"
            
            if model_file.exists() and config_file.exists():
                import json
                try:
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    trained_model = {
                        'path': str(model_file),
                        'architecture': config['architecture']
                    }
                    break
                except Exception:
                    continue
    
    if not trained_model:
        inference_logger.info("No trained models found, skipping trained model test")
        return True
    
    try:
        inference_logger.info(f"Testing with {trained_model['architecture']} model...")
        
        converter = ONNXConverter()
        
        # Convert the trained model
        test_output_path = "test_trained_model.onnx"
        results = converter.convert_from_checkpoint(
            checkpoint_path=trained_model['path'],
            architecture=trained_model['architecture'],
            output_path=test_output_path,
            optimize=True
        )
        
        # Clean up
        if Path(test_output_path).exists():
            Path(test_output_path).unlink()
        
        if results['success'] and results['validation_results']['validation_passed']:
            inference_logger.info("✅ Trained model conversion test passed")
            return True
        else:
            inference_logger.error("❌ Trained model conversion test failed")
            return False
            
    except Exception as e:
        inference_logger.error(f"Trained model conversion test failed: {e}")
        return False


def main():
    """Run all ONNX conversion tests."""
    setup_logging(log_level="INFO")
    
    inference_logger.info("="*60)
    inference_logger.info("🧪 ONNX CONVERSION PIPELINE TESTS")
    inference_logger.info("="*60)
    
    # Run tests
    tests = [
        ("ONNX Converter", test_onnx_converter),
        ("Model Benchmark", test_model_benchmark),
        ("ONNX Optimization", test_optimization),
        ("Trained Model Conversion", test_with_trained_model)
    ]
    
    results = {}
    for test_name, test_func in tests:
        inference_logger.info(f"\n{'='*50}")
        inference_logger.info(f"Running test: {test_name}")
        inference_logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            inference_logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    inference_logger.info(f"\n{'='*60}")
    inference_logger.info("TEST SUMMARY")
    inference_logger.info(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        inference_logger.info(f"{test_name}: {status}")
    
    inference_logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        inference_logger.info("🎉 All ONNX conversion tests passed!")
        inference_logger.info("The ONNX conversion pipeline is working correctly.")
        return True
    else:
        inference_logger.error(f"❌ {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ONNX conversion pipeline is ready!")
    else:
        print("\n❌ ONNX conversion pipeline has issues!")
    
    sys.exit(0 if success else 1)