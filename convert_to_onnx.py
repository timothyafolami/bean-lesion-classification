"""
Main script to convert trained PyTorch models to optimized ONNX format.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime

from src.inference.onnx_converter import ONNXConverter
from src.inference.benchmark import benchmark_model_conversion
from src.utils.logging_config import setup_logging, inference_logger
from src.utils.helpers import save_json


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint (.pth file)",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b1",
            "vgg16",
            "densenet121",
        ],
        help="Model architecture name",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for ONNX model (default: auto-generated)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes (default: 3)",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size [height width] (default: 224 224)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for conversion (default: 1)",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Apply ONNX optimizations (default: True)",
    )

    parser.add_argument(
        "--optimization-level",
        type=str,
        choices=["basic", "extended", "all"],
        default="basic",
        help="ONNX optimization level (default: basic)",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after conversion",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="onnx_models",
        help="Output directory for ONNX models (default: onnx_models)",
    )

    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(log_level="INFO")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output path if not provided
    if args.output_path is None:
        model_name = Path(args.model_path).parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_path = str(
            output_dir / f"{model_name}_{args.architecture}_optimized.onnx"
        )

    inference_logger.info("=" * 60)
    inference_logger.info("🔄 PYTORCH TO ONNX CONVERSION")
    inference_logger.info("=" * 60)
    inference_logger.info(f"Input model: {args.model_path}")
    inference_logger.info(f"Architecture: {args.architecture}")
    inference_logger.info(f"Output path: {args.output_path}")
    inference_logger.info(f"Input size: {args.input_size}")
    inference_logger.info(f"Batch size: {args.batch_size}")
    inference_logger.info(f"Optimization: {args.optimize} ({args.optimization_level})")
    inference_logger.info(f"Benchmark: {args.benchmark}")
    inference_logger.info("=" * 60)

    try:
        # Check if input model exists
        if not Path(args.model_path).exists():
            inference_logger.error(f"Model file not found: {args.model_path}")
            return False

        # Initialize converter
        converter = ONNXConverter()

        # Convert model
        inference_logger.info("🚀 Starting conversion...")

        input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])

        results = converter.convert_from_checkpoint(
            checkpoint_path=args.model_path,
            architecture=args.architecture,
            output_path=args.output_path,
            num_classes=args.num_classes,
            input_shape=input_shape,
            optimize=args.optimize,
            optimization_level=args.optimization_level,
        )

        if not results["success"]:
            inference_logger.error(
                f"Conversion failed: {results.get('error', 'Unknown error')}"
            )
            return False

        # Save conversion results
        results_path = (
            output_dir / f"{Path(args.output_path).stem}_conversion_results.json"
        )
        save_json(results, str(results_path))
        inference_logger.info(f"Conversion results saved to: {results_path}")

        # Print conversion summary
        inference_logger.info("\n" + "=" * 60)
        inference_logger.info("✅ CONVERSION COMPLETED SUCCESSFULLY")
        inference_logger.info("=" * 60)

        validation = results["validation_results"]
        inference_logger.info(
            f"✅ Validation passed: {validation['validation_passed']}"
        )
        inference_logger.info(f"✅ Max difference: {validation['max_difference']:.2e}")
        inference_logger.info(
            f"✅ Mean difference: {validation['mean_difference']:.2e}"
        )

        if args.optimize and "optimized_validation" in validation:
            opt_validation = validation["optimized_validation"]
            inference_logger.info(
                f"✅ Optimized validation passed: {opt_validation['validation_passed']}"
            )

        # Get file sizes
        pytorch_size = Path(args.model_path).stat().st_size / (1024 * 1024)  # MB
        onnx_size = Path(args.output_path).stat().st_size / (1024 * 1024)  # MB

        inference_logger.info(f"📊 PyTorch model size: {pytorch_size:.1f} MB")
        inference_logger.info(f"📊 ONNX model size: {onnx_size:.1f} MB")
        inference_logger.info(
            f"📊 Size change: {((onnx_size - pytorch_size) / pytorch_size * 100):+.1f}%"
        )

        # Run benchmark if requested
        if args.benchmark:
            inference_logger.info("\n🏃‍♂️ Running performance benchmark...")

            benchmark_dir = output_dir / "benchmarks"
            benchmark_results = benchmark_model_conversion(
                pytorch_path=args.model_path,
                onnx_path=args.output_path,
                architecture=args.architecture,
                output_dir=str(benchmark_dir),
            )

            # Print benchmark summary
            summary = benchmark_results["summary"]
            inference_logger.info("\n📈 BENCHMARK RESULTS:")
            inference_logger.info(f"  Average speedup: {summary['avg_speedup']:.2f}x")
            inference_logger.info(f"  Best speedup: {summary['max_speedup']:.2f}x")
            inference_logger.info(
                f"  Throughput improvement: {summary['avg_throughput_improvement']:.1f}%"
            )

        # Final recommendations
        inference_logger.info("\n💡 NEXT STEPS:")
        inference_logger.info("1. Test the ONNX model with your inference pipeline")
        inference_logger.info("2. Integrate the ONNX model into your FastAPI backend")
        inference_logger.info("3. Deploy the optimized model for production use")

        inference_logger.info(
            f"\n🎯 ONNX model ready for deployment: {args.output_path}"
        )

        return True

    except Exception as e:
        inference_logger.error(f"❌ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ONNX conversion completed successfully!")
        print("Your model is ready for fast inference deployment.")
    else:
        print("\n❌ ONNX conversion failed!")

    sys.exit(0 if success else 1)
