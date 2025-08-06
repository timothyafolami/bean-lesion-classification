"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class ModelFormat(str, Enum):
    """Supported model formats."""
    ONNX = "onnx"
    PYTORCH = "pytorch"


class ValidationInfo(BaseModel):
    """Image validation information."""
    file_size: Optional[int] = Field(None, description="File size in bytes")
    detected_format: Optional[str] = Field(None, description="Detected image format")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")
    quality_score: Optional[float] = Field(None, description="Image quality score (0-1)")
    quality_issues: Optional[List[str]] = Field(None, description="List of quality issues")
    is_corrupted: Optional[bool] = Field(None, description="Whether image appears corrupted")
    processing_time: Optional[float] = Field(None, description="Validation processing time")


class ClassificationResult(BaseModel):
    """Single classification result."""
    class_id: int = Field(..., description="Predicted class ID")
    class_name: str = Field(..., description="Predicted class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_format: str = Field(..., description="Model format used for prediction")
    filename: Optional[str] = Field(None, description="Original filename (for batch predictions)")
    validation_info: Optional[ValidationInfo] = Field(None, description="Image validation information")


class PredictionResponse(BaseModel):
    """Response for single image prediction."""
    success: bool = Field(..., description="Whether the prediction was successful")
    result: ClassificationResult = Field(..., description="Classification result")
    total_processing_time: float = Field(..., description="Total processing time in seconds")


class BatchPredictionResponse(BaseModel):
    """Response for batch image prediction."""
    success: bool = Field(..., description="Whether the batch prediction was successful")
    results: List[ClassificationResult] = Field(..., description="List of classification results")
    batch_size: int = Field(..., description="Number of images processed")
    total_processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status (healthy/unhealthy)")
    message: str = Field(..., description="Health status message")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Basic model information")


class ModelInfoResponse(BaseModel):
    """Model information response."""
    status: str = Field(..., description="Model status")
    model_format: Optional[str] = Field(None, description="Model format (onnx/pytorch)")
    architecture: Optional[str] = Field(None, description="Model architecture")
    num_classes: Optional[int] = Field(None, description="Number of output classes")
    class_names: List[str] = Field(default=[], description="List of class names")
    device: Optional[str] = Field(None, description="Device used for inference")
    additional_info: Dict[str, Any] = Field(default={}, description="Additional model information")


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(False, description="Always false for error responses")
    error: Dict[str, Any] = Field(..., description="Error details")


class ClassInfo(BaseModel):
    """Information about a classification class."""
    id: int = Field(..., description="Class ID")
    name: str = Field(..., description="Class name")
    description: str = Field(..., description="Class description")


class ClassesResponse(BaseModel):
    """Response for available classes."""
    classes: List[ClassInfo] = Field(..., description="List of available classes")


class PreprocessingInfo(BaseModel):
    """Image preprocessing information."""
    configuration: Dict[str, Any] = Field(..., description="Preprocessing configuration")
    normalization: Dict[str, Any] = Field(..., description="Normalization parameters")


class PreprocessingInfoResponse(BaseModel):
    """Response for preprocessing information."""
    success: bool = Field(..., description="Whether the request was successful")
    preprocessing_info: PreprocessingInfo = Field(..., description="Preprocessing information")


# Request models (if needed for more complex requests)

class PredictionRequest(BaseModel):
    """Request model for predictions (if using JSON instead of form data)."""
    return_probabilities: bool = Field(True, description="Whether to return class probabilities")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions (if using JSON instead of form data)."""
    return_probabilities: bool = Field(True, description="Whether to return class probabilities")
    max_batch_size: int = Field(10, description="Maximum number of images to process")


# Configuration models

class ModelConfig(BaseModel):
    """Model configuration."""
    model_path: str = Field(..., description="Path to model file")
    model_format: str = Field("auto", description="Model format (auto/onnx/pytorch)")
    architecture: str = Field("efficientnet_b0", description="Model architecture")
    device: str = Field("auto", description="Device to use for inference")
    num_classes: int = Field(3, description="Number of output classes")


class APIConfig(BaseModel):
    """API configuration."""
    host: str = Field("0.0.0.0", description="Host to bind to")
    port: int = Field(8000, description="Port to bind to")
    workers: int = Field(1, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload for development")
    max_upload_size: int = Field(10 * 1024 * 1024, description="Maximum upload size in bytes")
    max_batch_size: int = Field(10, description="Maximum batch size for predictions")


# Example responses for documentation

class ExampleResponses:
    """Example responses for API documentation."""
    
    SINGLE_PREDICTION_SUCCESS = {
        "success": True,
        "result": {
            "class_id": 0,
            "class_name": "healthy",
            "confidence": 0.95,
            "probabilities": {
                "healthy": 0.95,
                "angular_leaf_spot": 0.03,
                "bean_rust": 0.02
            },
            "processing_time": 0.15,
            "model_format": "onnx"
        },
        "total_processing_time": 0.18
    }
    
    BATCH_PREDICTION_SUCCESS = {
        "success": True,
        "results": [
            {
                "class_id": 0,
                "class_name": "healthy",
                "confidence": 0.95,
                "probabilities": {
                    "healthy": 0.95,
                    "angular_leaf_spot": 0.03,
                    "bean_rust": 0.02
                },
                "processing_time": 0.15,
                "model_format": "onnx",
                "filename": "leaf1.jpg"
            },
            {
                "class_id": 1,
                "class_name": "angular_leaf_spot",
                "confidence": 0.87,
                "probabilities": {
                    "healthy": 0.08,
                    "angular_leaf_spot": 0.87,
                    "bean_rust": 0.05
                },
                "processing_time": 0.14,
                "model_format": "onnx",
                "filename": "leaf2.jpg"
            }
        ],
        "batch_size": 2,
        "total_processing_time": 0.32
    }
    
    HEALTH_CHECK_SUCCESS = {
        "status": "healthy",
        "message": "Service is healthy",
        "model_info": {
            "format": "onnx",
            "inference_time": 0.05
        }
    }
    
    MODEL_INFO_SUCCESS = {
        "status": "loaded",
        "model_format": "onnx",
        "architecture": "efficientnet_b0",
        "num_classes": 3,
        "class_names": ["healthy", "angular_leaf_spot", "bean_rust"],
        "device": "cpu",
        "additional_info": {
            "providers": ["CPUExecutionProvider"],
            "input_shape": [1, 3, 224, 224],
            "output_shape": [1, 3]
        }
    }
    
    ERROR_RESPONSE = {
        "success": False,
        "error": {
            "code": 400,
            "message": "Invalid image file",
            "type": "validation_error"
        }
    }