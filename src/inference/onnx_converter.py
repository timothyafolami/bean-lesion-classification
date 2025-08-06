"""
ONNX conversion utilities for PyTorch models.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import time

from src.training.models import ModelFactory, load_model
from src.utils.logging_config import inference_logger
from src.utils.helpers import Timer, get_device, format_bytes


class ONNXConverter:
    """
    Convert PyTorch models to optimized ONNX format.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize ONNX converter.
        
        Args:
            device: Device to use for conversion
        """
        self.device = device or get_device()
        inference_logger.info(f"ONNX Converter initialized with device: {self.device}")
    
    def convert_pytorch_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        output_path: str = "model.onnx",
        opset_version: int = 11,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None
    ) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (batch_size, channels, height, width)
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable input sizes
            input_names: Names for input tensors
            output_names: Names for output tensors
            
        Returns:
            Path to saved ONNX model
        """
        inference_logger.info(f"Converting PyTorch model to ONNX...")
        inference_logger.info(f"Input shape: {input_shape}")
        inference_logger.info(f"Output path: {output_path}")
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        model.eval()
        model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)
        
        # Default dynamic axes for batch size flexibility
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Default input/output names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']
        
        try:
            with Timer("ONNX conversion"):
                torch.onnx.export(
                    model,
                    dummy_input,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=False
                )
            
            # Verify the exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            # Get model size
            model_size = Path(output_path).stat().st_size
            inference_logger.info(f"✅ ONNX model saved: {output_path}")
            inference_logger.info(f"Model size: {format_bytes(model_size)}")
            
            return output_path
            
        except Exception as e:
            inference_logger.error(f"ONNX conversion failed: {e}")
            raise
    
    def validate_conversion(
        self,
        pytorch_model: torch.nn.Module,
        onnx_path: str,
        test_inputs: Optional[torch.Tensor] = None,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Validate that ONNX model produces same outputs as PyTorch model.
        
        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            test_inputs: Test inputs for validation
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Validation results dictionary
        """
        inference_logger.info("Validating ONNX conversion...")
        
        # Create test inputs if not provided
        if test_inputs is None:
            test_inputs = torch.randn(5, 3, 224, 224, device=self.device)
        
        # Get PyTorch predictions
        pytorch_model.eval()
        pytorch_model.to(self.device)
        
        with torch.no_grad():
            pytorch_outputs = pytorch_model(test_inputs)
            if isinstance(pytorch_outputs, tuple):
                pytorch_outputs = pytorch_outputs[0]
            pytorch_outputs = pytorch_outputs.cpu().numpy()
        
        # Get ONNX predictions
        ort_session = ort.InferenceSession(onnx_path)
        onnx_inputs = {ort_session.get_inputs()[0].name: test_inputs.cpu().numpy()}
        onnx_outputs = ort_session.run(None, onnx_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_outputs - onnx_outputs))
        mean_diff = np.mean(np.abs(pytorch_outputs - onnx_outputs))
        
        validation_passed = max_diff < tolerance
        
        results = {
            'validation_passed': validation_passed,
            'max_difference': float(max_diff),
            'mean_difference': float(mean_diff),
            'tolerance': tolerance,
            'test_samples': test_inputs.shape[0],
            'pytorch_output_shape': pytorch_outputs.shape,
            'onnx_output_shape': onnx_outputs.shape
        }
        
        if validation_passed:
            inference_logger.info("✅ ONNX validation passed")
            inference_logger.info(f"Max difference: {max_diff:.2e}")
            inference_logger.info(f"Mean difference: {mean_diff:.2e}")
        else:
            inference_logger.error("❌ ONNX validation failed")
            inference_logger.error(f"Max difference: {max_diff:.2e} (tolerance: {tolerance:.2e})")
        
        return results
    
    def optimize_onnx_model(
        self,
        onnx_path: str,
        optimized_path: Optional[str] = None,
        optimization_level: str = "basic"
    ) -> str:
        """
        Optimize ONNX model for better performance.
        
        Args:
            onnx_path: Path to input ONNX model
            optimized_path: Path to save optimized model
            optimization_level: Optimization level ('basic', 'extended', 'all')
            
        Returns:
            Path to optimized ONNX model
        """
        if optimized_path is None:
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
        
        inference_logger.info(f"Optimizing ONNX model: {optimization_level} level")
        
        try:
            # Load the model
            onnx_model = onnx.load(onnx_path)
            
            # Apply optimizations based on level
            try:
                import onnxoptimizer
                
                if optimization_level == "basic":
                    # Basic optimizations - use common safe passes
                    passes = [
                        'eliminate_identity',
                        'eliminate_nop_transpose',
                        'eliminate_nop_pad',
                        'eliminate_unused_initializer'
                    ]
                    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
                    
                elif optimization_level == "extended":
                    # Extended optimizations - use fuse and elimination passes
                    optimized_model = onnxoptimizer.optimize(onnx_model, onnxoptimizer.get_fuse_and_elimination_passes())
                    
                elif optimization_level == "all":
                    # All available optimizations
                    optimized_model = onnxoptimizer.optimize(onnx_model, onnxoptimizer.get_available_passes())
                    
                else:
                    raise ValueError(f"Unknown optimization level: {optimization_level}")
                    
            except ImportError:
                inference_logger.warning("onnxoptimizer not available, skipping optimization")
                optimized_model = onnx_model
            except Exception as e:
                inference_logger.warning(f"ONNX optimization failed: {e}, using original model")
                optimized_model = onnx_model
            
            # Save optimized model
            onnx.save(optimized_model, optimized_path)
            
            # Compare sizes
            original_size = Path(onnx_path).stat().st_size
            optimized_size = Path(optimized_path).stat().st_size
            size_reduction = (original_size - optimized_size) / original_size * 100
            
            inference_logger.info(f"✅ Optimized model saved: {optimized_path}")
            inference_logger.info(f"Original size: {format_bytes(original_size)}")
            inference_logger.info(f"Optimized size: {format_bytes(optimized_size)}")
            inference_logger.info(f"Size reduction: {size_reduction:.1f}%")
            
            return optimized_path
            
        except Exception as e:
            inference_logger.error(f"ONNX optimization failed: {e}")
            # Return original path if optimization fails
            return onnx_path
    
    def convert_from_checkpoint(
        self,
        checkpoint_path: str,
        architecture: str,
        output_path: str,
        num_classes: int = 3,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        optimize: bool = True,
        optimization_level: str = "basic"
    ) -> Dict[str, Any]:
        """
        Convert PyTorch checkpoint to optimized ONNX model.
        
        Args:
            checkpoint_path: Path to PyTorch checkpoint
            architecture: Model architecture name
            output_path: Path to save ONNX model
            num_classes: Number of output classes
            input_shape: Input tensor shape
            optimize: Whether to optimize the ONNX model
            optimization_level: Level of optimization to apply
            
        Returns:
            Conversion results dictionary
        """
        inference_logger.info(f"Converting checkpoint to ONNX: {checkpoint_path}")
        
        try:
            # Load PyTorch model
            model, checkpoint_info = load_model(
                filepath=checkpoint_path,
                architecture=architecture,
                num_classes=num_classes,
                device=self.device
            )
            
            inference_logger.info(f"Loaded model: {architecture}")
            inference_logger.info(f"Checkpoint epoch: {checkpoint_info.get('epoch', 'unknown')}")
            
            # Convert to ONNX
            onnx_path = self.convert_pytorch_to_onnx(
                model=model,
                input_shape=input_shape,
                output_path=output_path
            )
            
            # Validate conversion
            validation_results = self.validate_conversion(model, onnx_path)
            
            # Optimize if requested
            optimized_path = onnx_path
            if optimize:
                optimized_path = self.optimize_onnx_model(
                    onnx_path=onnx_path,
                    optimization_level=optimization_level
                )
                
                # Validate optimized model
                optimized_validation = self.validate_conversion(model, optimized_path)
                validation_results['optimized_validation'] = optimized_validation
            
            results = {
                'success': True,
                'pytorch_checkpoint': checkpoint_path,
                'onnx_model': optimized_path,
                'architecture': architecture,
                'input_shape': input_shape,
                'validation_results': validation_results,
                'checkpoint_info': checkpoint_info,
                'optimized': optimize
            }
            
            inference_logger.info("✅ Checkpoint to ONNX conversion completed successfully")
            return results
            
        except Exception as e:
            inference_logger.error(f"Checkpoint conversion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pytorch_checkpoint': checkpoint_path
            }


def convert_model_to_onnx(
    checkpoint_path: str,
    architecture: str,
    output_path: str,
    num_classes: int = 3,
    optimize: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to convert a trained model to ONNX.
    
    Args:
        checkpoint_path: Path to PyTorch checkpoint
        architecture: Model architecture name
        output_path: Path to save ONNX model
        num_classes: Number of output classes
        optimize: Whether to optimize the ONNX model
        
    Returns:
        Conversion results
    """
    converter = ONNXConverter()
    
    return converter.convert_from_checkpoint(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        output_path=output_path,
        num_classes=num_classes,
        optimize=optimize
    )