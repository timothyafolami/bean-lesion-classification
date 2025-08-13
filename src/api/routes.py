"""
API routes for bean lesion classification.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import time

from src.api.schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
    ErrorResponse,
)
from src.api.models import ModelManager
from src.utils.config import load_api_config
from src.utils.logging_config import api_logger


# Dependency function to get model manager
def get_model_manager():
    """
    Dependency to get the model manager instance.
    This will be overridden in main.py with the actual instance.
    """
    from src.api.main import model_manager

    if model_manager is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please check server logs."
        )
    return model_manager


# Health router
health_router = APIRouter()

# Prediction router
prediction_router = APIRouter()

# Load configurable limits (fallback to env/defaults)
try:
    _api_config = load_api_config()
    DEFAULT_MAX_BATCH_SIZE = int(_api_config.get("upload", {}).get("max_batch_size", 10))
except Exception:
    import os
    DEFAULT_MAX_BATCH_SIZE = int(os.getenv("BATCH_MAX_SIZE", "10"))


@health_router.get("/", response_model=HealthResponse)
async def health_check(model_manager: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint.
    """
    try:
        health_result = await model_manager.health_check()

        return HealthResponse(
            status=health_result["status"],
            message=health_result.get("message", "Service is healthy"),
            model_info={
                "format": health_result.get("model_format"),
                "inference_time": health_result.get("inference_time"),
            },
        )

    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@health_router.get("/model", response_model=ModelInfoResponse)
async def get_model_info(model_manager: ModelManager = Depends(get_model_manager)):
    """
    Get detailed model information.
    """
    try:
        model_info = await model_manager.get_model_info()

        return ModelInfoResponse(
            status=model_info["status"],
            model_format=model_info.get("model_format"),
            architecture=model_info.get("architecture"),
            num_classes=model_info.get("num_classes"),
            class_names=model_info.get("class_names", []),
            device=model_info.get("device"),
            additional_info=model_info,
        )

    except Exception as e:
        api_logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve model information"
        )


@prediction_router.post("/single", response_model=PredictionResponse)
async def predict_single(
    file: UploadFile = File(..., description="Image file to classify"),
    return_probabilities: bool = Form(
        True, description="Whether to return class probabilities"
    ),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Classify a single image.

    Args:
        file: Image file (JPEG, PNG, WebP)
        return_probabilities: Whether to return class probabilities
        model_manager: Model manager dependency

    Returns:
        Classification result
    """
    start_time = time.time()

    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="File must be an image (JPEG, PNG, WebP)"
            )

        # Read image data
        try:
            image_data = await file.read()
        except Exception as e:
            api_logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=400, detail="Failed to read uploaded file")

        # Make prediction with enhanced preprocessing
        try:
            result = await model_manager.predict(
                image=image_data,
                filename=file.filename,
                content_type=file.content_type,
                return_probabilities=return_probabilities,
                return_validation_info=True,
            )

            api_logger.info(
                f"Single prediction completed: {result['class_name']} ({result['confidence']:.3f})"
            )

            return PredictionResponse(
                success=True,
                result=result,
                total_processing_time=time.time() - start_time,
            )

        except Exception as e:
            api_logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Unexpected error in single prediction: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )


@prediction_router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    files: List[UploadFile] = File(..., description="List of image files to classify"),
    return_probabilities: bool = Form(
        True, description="Whether to return class probabilities"
    ),
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Classify multiple images in batch.

    Args:
        files: List of image files (JPEG, PNG, WebP)
        return_probabilities: Whether to return class probabilities
        model_manager: Model manager dependency

    Returns:
        Batch classification results
    """
    start_time = time.time()

    try:
        # Validate batch size (configurable)
        max_batch_size = DEFAULT_MAX_BATCH_SIZE
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum {max_batch_size} files allowed.",
            )

        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        # Process all images with enhanced preprocessing
        image_data_list = []

        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} must be an image (JPEG, PNG, WebP)",
                )

            try:
                image_data = await file.read()
                image_data_list.append(
                    (
                        image_data,
                        file.filename or f"image_{len(image_data_list)}",
                        file.content_type,
                    )
                )
            except Exception as e:
                api_logger.error(f"Failed to read file {file.filename}: {e}")
                raise HTTPException(
                    status_code=400, detail=f"Failed to read file: {file.filename}"
                )

        # Make batch prediction with enhanced preprocessing
        try:
            results = await model_manager.predict_batch(
                images=image_data_list,
                return_probabilities=return_probabilities,
                return_validation_info=True,
                fail_on_error=False,  # Skip invalid images instead of failing
            )

            api_logger.info(
                f"Batch prediction completed: {len(results)} images processed"
            )

            return BatchPredictionResponse(
                success=True,
                results=results,
                batch_size=len(results),
                total_processing_time=time.time() - start_time,
            )

        except Exception as e:
            api_logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Unexpected error in batch prediction: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during batch prediction"
        )


@prediction_router.get("/classes")
async def get_classes():
    """
    Get available classification classes.
    """
    return {
        "classes": [
            {
                "id": 0,
                "name": "healthy",
                "description": "Healthy bean leaves without disease",
            },
            {
                "id": 1,
                "name": "angular_leaf_spot",
                "description": "Angular leaf spot disease caused by Pseudocercospora griseola",
            },
            {
                "id": 2,
                "name": "bean_rust",
                "description": "Bean rust disease caused by Uromyces appendiculatus",
            },
        ]
    }


@prediction_router.get("/preprocessing-info")
async def get_preprocessing_info(
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Get information about image preprocessing capabilities and requirements.
    """
    try:
        preprocessing_info = model_manager.get_preprocessing_info()
        return {"success": True, "preprocessing_info": preprocessing_info}
    except Exception as e:
        api_logger.error(f"Failed to get preprocessing info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve preprocessing information"
        )


@prediction_router.get("/performance-stats")
async def get_performance_stats(
    model_manager: ModelManager = Depends(get_model_manager),
):
    """
    Get detailed performance statistics from the ONNX inference engine.
    """
    try:
        if (
            model_manager.model_format == "onnx"
            and model_manager.use_advanced_onnx
            and model_manager.onnx_engine
        ):

            performance_stats = model_manager.onnx_engine.get_performance_stats()
            return {
                "success": True,
                "performance_stats": performance_stats,
                "engine_type": "advanced_onnx",
            }
        else:
            return {
                "success": True,
                "message": "Performance stats only available for advanced ONNX engine",
                "engine_type": model_manager.model_format,
            }
    except Exception as e:
        api_logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve performance statistics"
        )


@prediction_router.post("/benchmark")
async def run_benchmark(model_manager: ModelManager = Depends(get_model_manager)):
    """
    Run a performance benchmark on the inference engine.
    """
    try:
        if (
            model_manager.model_format == "onnx"
            and model_manager.use_advanced_onnx
            and model_manager.onnx_engine
        ):

            api_logger.info("Starting inference benchmark...")

            benchmark_results = await model_manager.onnx_engine.benchmark(
                num_runs=20, batch_sizes=[1, 4, 8], return_detailed_stats=False
            )

            return {"success": True, "benchmark_results": benchmark_results}
        else:
            raise HTTPException(
                status_code=400,
                detail="Benchmarking only available for advanced ONNX engine",
            )
    except Exception as e:
        api_logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")
