"""
Model management for both ONNX and PyTorch formats.
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
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from src.training.models import load_model, ModelFactory
from src.utils.logging_config import api_logger
from src.utils.helpers import get_device
from src.api.image_processor import ImagePreprocessor, ImageValidationError
from src.inference.onnx_engine import ONNXInferenceEngine, create_optimized_engine


class ModelManager:
    """
    Manages both ONNX and PyTorch models for inference.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_format: str = "auto",  # auto, onnx, pytorch
        architecture: str = "efficientnet_b0",
        device: str = "auto",
        num_classes: int = 3
    ):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to model file
            model_format: Model format (auto, onnx, pytorch)
            architecture: Model architecture name
            device: Device to use (auto, cpu, cuda, mps)
            num_classes: Number of output classes
        """
        self.model_path = model_path
        self.model_format = model_format
        self.architecture = architecture
        self.num_classes = num_classes
        self.class_names = ["healthy", "angular_leaf_spot", "bean_rust"]
        
        # Device setup
        if device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(device)
        
        # Model instances
        self.pytorch_model = None
        self.onnx_session = None
        self.onnx_engine = None  # Advanced ONNX engine
        self.model_loaded = False
        self.use_advanced_onnx = True  # Use advanced ONNX engine by default
        
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize image preprocessor
        self.image_preprocessor = ImagePreprocessor(
            model_input_size=(224, 224),
            enable_quality_check=True,
            enable_corruption_check=True
        )
        
        api_logger.info(f"ModelManager initialized:")
        api_logger.info(f"  Model path: {model_path}")
        api_logger.info(f"  Format: {model_format}")
        api_logger.info(f"  Architecture: {architecture}")
        api_logger.info(f"  Device: {self.device}")
    
    async def load_model(self) -> None:
        """
        Load the model based on the specified format.
        """
        if not self.model_path:
            raise ValueError("Model path not specified")
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Determine model format
        if self.model_format == "auto":
            if model_path.suffix.lower() == ".onnx":
                self.model_format = "onnx"
            elif model_path.suffix.lower() in [".pth", ".pt"]:
                self.model_format = "pytorch"
            else:
                raise ValueError(f"Cannot determine model format from file extension: {model_path.suffix}")
        
        api_logger.info(f"Loading {self.model_format} model from: {model_path}")
        
        try:
            if self.model_format == "onnx":
                if self.use_advanced_onnx:
                    await self._load_advanced_onnx_model(str(model_path))
                else:
                    await self._load_onnx_model(str(model_path))
            elif self.model_format == "pytorch":
                await self._load_pytorch_model(str(model_path))
            else:
                raise ValueError(f"Unsupported model format: {self.model_format}")
            
            self.model_loaded = True
            api_logger.info(f"✅ {self.model_format.upper()} model loaded successfully")
            
        except Exception as e:
            api_logger.error(f"❌ Failed to load {self.model_format} model: {e}")
            raise
    
    async def _load_onnx_model(self, model_path: str) -> None:
        """Load ONNX model."""
        def _load():
            # Set up providers
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create ONNX Runtime session
            session = ort.InferenceSession(model_path, providers=providers)
            
            api_logger.info(f"ONNX providers: {session.get_providers()}")
            api_logger.info(f"Input shape: {session.get_inputs()[0].shape}")
            api_logger.info(f"Output shape: {session.get_outputs()[0].shape}")
            
            return session
        
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.onnx_session = await loop.run_in_executor(self.executor, _load)
    
    async def _load_advanced_onnx_model(self, model_path: str) -> None:
        """Load advanced ONNX model using the high-performance engine."""
        def _load():
            # Create optimized ONNX engine
            engine = create_optimized_engine(
                model_path=model_path,
                class_names=self.class_names,
                device="auto"
            )
            
            api_logger.info(f"Advanced ONNX engine created")
            api_logger.info(f"Session pool size: {engine.session_pool.max_sessions}")
            api_logger.info(f"Caching enabled: {engine.enable_caching}")
            api_logger.info(f"Providers: {engine.providers}")
            
            return engine
        
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.onnx_engine = await loop.run_in_executor(self.executor, _load)
        
        # Warm up the engine
        try:
            warmup_stats = await self.onnx_engine.warmup(num_warmup_runs=3)
            api_logger.info(f"Engine warmup completed: {warmup_stats.get('average_time', 0):.3f}s avg")
        except Exception as e:
            api_logger.warning(f"Engine warmup failed: {e}")
    
    async def _load_pytorch_model(self, model_path: str) -> None:
        """Load PyTorch model."""
        def _load():
            # Load model from checkpoint
            model, checkpoint_info = load_model(
                filepath=model_path,
                architecture=self.architecture,
                num_classes=self.num_classes,
                device=self.device
            )
            
            model.eval()
            
            api_logger.info(f"PyTorch model loaded from epoch: {checkpoint_info.get('epoch', 'unknown')}")
            api_logger.info(f"Model device: {next(model.parameters()).device}")
            
            return model
        
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self.pytorch_model = await loop.run_in_executor(self.executor, _load)
    
    async def predict(
        self,
        image: Union[Image.Image, np.ndarray, bytes],
        return_probabilities: bool = True,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        return_validation_info: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image: PIL Image, numpy array, or raw bytes
            return_probabilities: Whether to return class probabilities
            filename: Original filename (for bytes input)
            content_type: MIME content type (for bytes input)
            return_validation_info: Whether to include validation information
            
        Returns:
            Prediction results dictionary
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Preprocess image using the advanced preprocessor
            if isinstance(image, bytes):
                # Use advanced preprocessing for raw bytes
                if return_validation_info:
                    input_tensor, validation_info = await self.image_preprocessor.preprocess_single_image(
                        image, filename, content_type, return_validation_info=True
                    )
                else:
                    input_tensor = await self.image_preprocessor.preprocess_single_image(
                        image, filename, content_type, return_validation_info=False
                    )
            else:
                # Use legacy preprocessing for PIL/numpy inputs
                input_tensor = self._preprocess_image(image)
                validation_info = None
            
            # Run inference based on model format
            if self.model_format == "onnx":
                if self.use_advanced_onnx and self.onnx_engine:
                    # Use advanced ONNX engine
                    engine_result = await self.onnx_engine.predict_single(
                        input_tensor, 
                        return_probabilities=return_probabilities,
                        use_cache=True
                    )
                    # Convert engine result to our format
                    result = {
                        'class_id': engine_result.class_id,
                        'class_name': engine_result.class_name,
                        'confidence': engine_result.confidence,
                        'processing_time': time.time() - start_time,
                        'model_format': self.model_format,
                        'inference_time': engine_result.inference_time,
                        'session_id': engine_result.session_id,
                        'provider': engine_result.provider
                    }
                    if return_probabilities and engine_result.probabilities:
                        result['probabilities'] = engine_result.probabilities
                    
                    # Add validation info if requested and available
                    if return_validation_info and validation_info:
                        result['validation_info'] = validation_info
                    
                    return result
                else:
                    # Use legacy ONNX session
                    predictions = await self._predict_onnx(input_tensor)
            elif self.model_format == "pytorch":
                predictions = await self._predict_pytorch(input_tensor)
            else:
                raise ValueError(f"Unsupported model format: {self.model_format}")
            
            # Post-process results (for legacy paths)
            result = self._postprocess_predictions(predictions, return_probabilities)
            result['processing_time'] = time.time() - start_time
            result['model_format'] = self.model_format
            
            # Add validation info if requested and available
            if return_validation_info and validation_info:
                result['validation_info'] = validation_info
            
            return result
            
            # Post-process results
            result = self._postprocess_predictions(predictions, return_probabilities)
            result['processing_time'] = time.time() - start_time
            result['model_format'] = self.model_format
            
            # Add validation info if requested and available
            if return_validation_info and validation_info:
                result['validation_info'] = validation_info
            
            return result
            
        except ImageValidationError as e:
            api_logger.error(f"Image validation failed: {e}")
            raise RuntimeError(f"Image validation failed: {str(e)}")
        except Exception as e:
            api_logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    async def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, Tuple[bytes, Optional[str], Optional[str]]]],
        return_probabilities: bool = True,
        return_validation_info: bool = False,
        fail_on_error: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of PIL Images, numpy arrays, or (bytes, filename, content_type) tuples
            return_probabilities: Whether to return class probabilities
            return_validation_info: Whether to include validation information
            fail_on_error: Whether to fail on first error or skip invalid images
            
        Returns:
            List of prediction results
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Check if we have bytes input (advanced preprocessing)
            if images and isinstance(images[0], tuple):
                # Use advanced batch preprocessing
                if return_validation_info:
                    input_tensors, validation_infos = await self.image_preprocessor.preprocess_batch_images(
                        images, return_validation_info=True, fail_on_error=fail_on_error
                    )
                else:
                    input_tensors = await self.image_preprocessor.preprocess_batch_images(
                        images, return_validation_info=False, fail_on_error=fail_on_error
                    )
                    validation_infos = None
                
                # Stack tensors for batch inference
                if input_tensors:
                    input_batch = np.vstack(input_tensors)
                else:
                    return []  # No valid images to process
                
            else:
                # Use legacy preprocessing for PIL/numpy inputs
                input_batch = np.stack([self._preprocess_image(img) for img in images])
                validation_infos = None
            
            # Run batch inference
            if self.model_format == "onnx":
                if self.use_advanced_onnx and self.onnx_engine:
                    # Use advanced ONNX engine for batch processing
                    batch_result = await self.onnx_engine.predict_batch(
                        input_batch,
                        return_probabilities=return_probabilities
                    )
                    
                    # Convert engine results to our format
                    results = []
                    for i, engine_result in enumerate(batch_result.results):
                        result = {
                            'class_id': engine_result.class_id,
                            'class_name': engine_result.class_name,
                            'confidence': engine_result.confidence,
                            'processing_time': engine_result.inference_time,
                            'model_format': self.model_format,
                            'inference_time': engine_result.inference_time,
                            'session_id': engine_result.session_id,
                            'provider': engine_result.provider,
                            'batch_index': engine_result.batch_index
                        }
                        if return_probabilities and engine_result.probabilities:
                            result['probabilities'] = engine_result.probabilities
                        
                        # Add validation info if available
                        if return_validation_info and validation_infos and i < len(validation_infos):
                            result['validation_info'] = validation_infos[i]
                        
                        results.append(result)
                    
                    return results
                else:
                    # Use legacy ONNX session
                    predictions = await self._predict_onnx_batch(input_batch)
            elif self.model_format == "pytorch":
                predictions = await self._predict_pytorch_batch(input_batch)
            else:
                raise ValueError(f"Unsupported model format: {self.model_format}")
            
            # Post-process results (for legacy paths)
            results = []
            for i, pred in enumerate(predictions):
                result = self._postprocess_predictions(pred[np.newaxis, :], return_probabilities)
                result['processing_time'] = (time.time() - start_time) / len(predictions)
                result['model_format'] = self.model_format
                
                # Add validation info if available
                if return_validation_info and validation_infos and i < len(validation_infos):
                    result['validation_info'] = validation_infos[i]
                
                results.append(result)
            
            return results
            
        except ImageValidationError as e:
            api_logger.error(f"Batch image validation failed: {e}")
            raise RuntimeError(f"Batch validation failed: {str(e)}")
        except Exception as e:
            api_logger.error(f"Batch prediction failed: {e}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    async def _predict_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX inference."""
        def _infer():
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_tensor})
            return outputs[0]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _infer)
    
    async def _predict_onnx_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Run ONNX batch inference."""
        def _infer():
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_batch})
            return outputs[0]
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _infer)
    
    async def _predict_pytorch(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run PyTorch inference."""
        def _infer():
            # Convert to torch tensor with explicit float32 for MPS compatibility
            tensor = torch.from_numpy(input_tensor).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.pytorch_model(tensor)
                return outputs.cpu().numpy()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _infer)
    
    async def _predict_pytorch_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Run PyTorch batch inference."""
        def _infer():
            # Convert to torch tensor with explicit float32 for MPS compatibility
            tensor = torch.from_numpy(input_batch).float().to(self.device)
            
            with torch.no_grad():
                outputs = self.pytorch_model(tensor)
                return outputs.cpu().numpy()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _infer)
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for model input using EXACT same transforms as training validation.
        
        This matches the validation transforms from src/training/dataset.py:
        - Resize to slightly larger size (224 * 1.14 = 255)
        - Center crop to 224x224
        - ToTensor (converts to [0,1] and CHW format)
        - ImageNet normalization
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed numpy array
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # EXACT SAME PREPROCESSING AS TRAINING VALIDATION:
        # 1. Resize to slightly larger size (matches training validation transforms)
        resize_size = int(224 * 1.14)  # 255 pixels
        image = image.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
        
        # 2. Center crop to target size (matches training validation transforms)
        width, height = image.size
        target_size = 224
        
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        image = image.crop((left, top, right, bottom))
        
        # 3. Convert to tensor format (matches ToTensor transform)
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1] (ToTensor does this)
        img_array = img_array / 255.0
        
        # 4. Apply ImageNet normalization (exact same values as training)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # 5. Transpose from HWC to CHW format (ToTensor does this)
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        
        # 6. Add batch dimension
        img_array = img_array[np.newaxis, :]  # Add batch dimension
        
        return img_array
    
    def _postprocess_predictions(
        self,
        predictions: np.ndarray,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Post-process model predictions.
        
        Args:
            predictions: Raw model outputs
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Formatted prediction results
        """
        # Apply softmax to get probabilities
        probabilities = self._softmax(predictions[0])
        
        # Get predicted class
        predicted_class_id = int(np.argmax(probabilities))
        predicted_class_name = self.class_names[predicted_class_id]
        confidence = float(probabilities[predicted_class_id])
        
        result = {
            'class_id': predicted_class_id,
            'class_name': predicted_class_name,
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }
        
        return result
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_format": self.model_format,
            "model_path": str(self.model_path),
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "device": str(self.device)
        }
        
        if self.model_format == "onnx":
            if self.use_advanced_onnx and self.onnx_engine:
                # Get info from advanced engine
                engine_info = self.onnx_engine.get_model_info()
                info.update({
                    "providers": engine_info["providers"],
                    "input_shape": engine_info["metadata"]["input_shape"],
                    "output_shape": engine_info["metadata"]["output_shape"],
                    "session_pool_size": engine_info["session_pool_size"],
                    "caching_enabled": engine_info["caching_enabled"],
                    "model_size_mb": engine_info["model_size_mb"]
                })
                
                # Add performance stats
                perf_stats = self.onnx_engine.get_performance_stats()
                info.update({
                    "inference_count": perf_stats["inference_count"],
                    "average_inference_time": perf_stats["average_inference_time"],
                    "throughput_per_second": perf_stats["throughput_per_second"]
                })
            elif self.onnx_session:
                # Legacy ONNX session info
                info.update({
                    "providers": self.onnx_session.get_providers(),
                    "input_shape": self.onnx_session.get_inputs()[0].shape,
                    "output_shape": self.onnx_session.get_outputs()[0].shape
                })
        elif self.model_format == "pytorch" and self.pytorch_model:
            # Count parameters
            total_params = sum(p.numel() for p in self.pytorch_model.parameters())
            trainable_params = sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)
            
            info.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_device": str(next(self.pytorch_model.parameters()).device)
            })
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the model.
        
        Returns:
            Health check results
        """
        if not self.model_loaded:
            return {
                "status": "unhealthy",
                "message": "Model not loaded"
            }
        
        try:
            # Create a dummy image for testing
            dummy_image = Image.new('RGB', (224, 224), color='green')
            
            # Run a test prediction
            start_time = time.time()
            result = await self.predict(dummy_image, return_probabilities=False)
            inference_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model_format": self.model_format,
                "inference_time": inference_time,
                "test_prediction": result
            }
            
        except Exception as e:
            api_logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": str(e)
            }
    
    async def cleanup(self) -> None:
        """
        Clean up resources.
        """
        api_logger.info("Cleaning up model manager...")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Clear model references
        self.pytorch_model = None
        self.onnx_session = None
        self.model_loaded = False
        
        # Cleanup advanced ONNX engine
        if self.onnx_engine:
            await self.onnx_engine.cleanup()
            self.onnx_engine = None
        
        # Cleanup image preprocessor
        if self.image_preprocessor:
            await self.image_preprocessor.cleanup()
        
        api_logger.info("Model manager cleanup completed")
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """
        Get information about image preprocessing capabilities.
        
        Returns:
            Preprocessing information dictionary
        """
        return self.image_preprocessor.get_preprocessing_stats()