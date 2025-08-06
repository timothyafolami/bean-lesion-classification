"""
Test script for the ONNX inference engine.
"""

import sys
import os
from pathlib import Path
import asyncio
import time
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference.onnx_engine import (
    ONNXInferenceEngine,
    create_optimized_engine,
    benchmark_model
)


async def test_engine_initialization():
    """Test engine initialization and basic functionality."""
    print("🧪 Testing ONNX Engine Initialization")
    print("=" * 50)
    
    # Find available ONNX model
    onnx_models_dir = Path("onnx_models")
    if not onnx_models_dir.exists():
        print("❌ No onnx_models directory found")
        return False
    
    onnx_files = list(onnx_models_dir.glob("*.onnx"))
    if not onnx_files:
        print("❌ No ONNX models found in onnx_models directory")
        return False
    
    model_path = onnx_files[0]  # Use first available model
    print(f"Using model: {model_path}")
    
    try:
        # Test basic initialization
        engine = ONNXInferenceEngine(
            model_path=str(model_path),
            class_names=["healthy", "angular_leaf_spot", "bean_rust"],
            max_sessions=2,
            enable_caching=True,
            cache_size=100
        )
        
        print("✅ Engine initialized successfully")
        
        # Test model info
        model_info = engine.get_model_info()
        print(f"Model info:")
        print(f"  Classes: {model_info['num_classes']}")
        print(f"  Providers: {model_info['providers']}")
        print(f"  Input shape: {model_info['metadata']['input_shape']}")
        print(f"  Model size: {model_info['model_size_mb']:.1f}MB")
        
        # Cleanup
        await engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False


async def test_single_inference():
    """Test single image inference."""
    print("\n🧪 Testing Single Inference")
    print("=" * 50)
    
    # Find available ONNX model
    onnx_models_dir = Path("onnx_models")
    onnx_files = list(onnx_models_dir.glob("*.onnx"))
    model_path = onnx_files[0]
    
    try:
        engine = create_optimized_engine(str(model_path))
        
        # Create dummy input data (1, 3, 224, 224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Test single prediction
        start_time = time.time()
        result = await engine.predict_single(
            dummy_input, 
            return_probabilities=True, 
            use_cache=True
        )
        total_time = time.time() - start_time
        
        print("✅ Single inference completed")
        print(f"  Predicted class: {result.class_name}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Inference time: {result.inference_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Session ID: {result.session_id}")
        print(f"  Provider: {result.provider}")
        
        if result.probabilities:
            print("  Probabilities:")
            for class_name, prob in result.probabilities.items():
                print(f"    {class_name}: {prob:.3f}")
        
        # Test cache hit
        start_time = time.time()
        cached_result = await engine.predict_single(
            dummy_input, 
            return_probabilities=True, 
            use_cache=True
        )
        cache_time = time.time() - start_time
        
        print(f"\n✅ Cache test completed")
        print(f"  Cache hit time: {cache_time:.3f}s")
        print(f"  Speed improvement: {result.inference_time / cache_time:.1f}x")
        
        await engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Single inference failed: {e}")
        return False


async def test_batch_inference():
    """Test batch inference."""
    print("\n🧪 Testing Batch Inference")
    print("=" * 50)
    
    # Find available ONNX model
    onnx_models_dir = Path("onnx_models")
    onnx_files = list(onnx_models_dir.glob("*.onnx"))
    model_path = onnx_files[0]
    
    try:
        engine = create_optimized_engine(str(model_path))
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Create dummy batch data
            dummy_batch = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
            
            start_time = time.time()
            batch_result = await engine.predict_batch(
                dummy_batch, 
                return_probabilities=True
            )
            total_time = time.time() - start_time
            
            print(f"  ✅ Batch inference completed")
            print(f"  Batch size: {batch_result.batch_size}")
            print(f"  Total time: {batch_result.total_inference_time:.3f}s")
            print(f"  Average per image: {batch_result.average_inference_time:.3f}s")
            print(f"  Throughput: {batch_size / batch_result.total_inference_time:.1f} images/sec")
            print(f"  Session ID: {batch_result.session_id}")
            print(f"  Provider: {batch_result.provider}")
            
            # Show first few results
            for i, result in enumerate(batch_result.results[:3]):
                print(f"    Image {i}: {result.class_name} ({result.confidence:.3f})")
        
        await engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Batch inference failed: {e}")
        return False


async def test_performance_features():
    """Test performance features like warmup and benchmarking."""
    print("\n🧪 Testing Performance Features")
    print("=" * 50)
    
    # Find available ONNX model
    onnx_models_dir = Path("onnx_models")
    onnx_files = list(onnx_models_dir.glob("*.onnx"))
    model_path = onnx_files[0]
    
    try:
        engine = create_optimized_engine(str(model_path))
        
        # Test warmup
        print("Running warmup...")
        warmup_stats = await engine.warmup(num_warmup_runs=3)
        print(f"✅ Warmup completed:")
        print(f"  Runs: {warmup_stats['warmup_runs']}")
        print(f"  Average time: {warmup_stats.get('average_time', 0):.3f}s")
        print(f"  Min time: {warmup_stats.get('min_time', 0):.3f}s")
        print(f"  Max time: {warmup_stats.get('max_time', 0):.3f}s")
        
        # Test health check
        print("\nRunning health check...")
        health_result = await engine.health_check()
        print(f"✅ Health check: {health_result['status']}")
        print(f"  Health check time: {health_result.get('health_check_time', 0):.3f}s")
        
        if 'test_prediction' in health_result:
            pred = health_result['test_prediction']
            print(f"  Test prediction: {pred['class_name']} ({pred['confidence']:.3f})")
        
        # Test performance stats
        print("\nPerformance statistics:")
        stats = engine.get_performance_stats()
        print(f"  Total inferences: {stats['inference_count']}")
        print(f"  Average time: {stats['average_inference_time']:.3f}s")
        print(f"  Throughput: {stats['throughput_per_second']:.1f} images/sec")
        
        # Session pool stats
        if 'session_pool' in stats:
            pool_stats = stats['session_pool']
            print(f"  Session pool: {pool_stats['available_sessions']}/{pool_stats['total_sessions']} available")
        
        # Cache stats
        if 'cache' in stats:
            cache_stats = stats['cache']
            print(f"  Cache: {cache_stats['size']}/{cache_stats['max_size']} ({cache_stats['utilization']:.1%} full)")
        
        await engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return False


async def test_benchmark():
    """Test the benchmarking functionality."""
    print("\n🧪 Testing Benchmark Functionality")
    print("=" * 50)
    
    # Find available ONNX model
    onnx_models_dir = Path("onnx_models")
    onnx_files = list(onnx_models_dir.glob("*.onnx"))
    model_path = onnx_files[0]
    
    try:
        print("Running quick benchmark...")
        
        # Use the utility function for quick benchmarking
        benchmark_results = await benchmark_model(
            str(model_path),
            class_names=["healthy", "angular_leaf_spot", "bean_rust"],
            num_runs=10  # Quick benchmark
        )
        
        print("✅ Benchmark completed")
        
        # Show results
        if 'benchmark_results' in benchmark_results:
            for batch_name, batch_stats in benchmark_results['benchmark_results'].items():
                batch_size = batch_stats['batch_size']
                avg_time = batch_stats['average_time']
                throughput = batch_stats['throughput_images_per_second']
                
                print(f"  Batch {batch_size}:")
                print(f"    Average time: {avg_time:.3f}s")
                print(f"    Throughput: {throughput:.1f} images/sec")
                print(f"    Time per image: {batch_stats['average_time_per_image']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False


async def main():
    """Run all ONNX engine tests."""
    print("🚀 ONNX Inference Engine Tests")
    print("=" * 60)
    
    # Check if ONNX models exist
    onnx_models_dir = Path("onnx_models")
    if not onnx_models_dir.exists() or not list(onnx_models_dir.glob("*.onnx")):
        print("❌ No ONNX models found!")
        print("Please ensure you have ONNX models in the 'onnx_models' directory.")
        print("You can convert PyTorch models using: python convert_to_onnx.py")
        return False
    
    tests = [
        ("Engine Initialization", test_engine_initialization),
        ("Single Inference", test_single_inference),
        ("Batch Inference", test_batch_inference),
        ("Performance Features", test_performance_features),
        ("Benchmark", test_benchmark)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ONNX engine tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)