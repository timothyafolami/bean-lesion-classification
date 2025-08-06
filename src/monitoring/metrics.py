"""
Prometheus metrics for Bean Classification API
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Enum,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Request, Response
import logging

# Create custom registry for better control
REGISTRY = CollectorRegistry()

# API Request Metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code'],
    registry=REGISTRY
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

api_request_size_bytes = Histogram(
    'api_request_size_bytes',
    'API request size in bytes',
    ['method', 'endpoint'],
    buckets=[1024, 10240, 102400, 1048576, 10485760, 52428800],  # 1KB to 50MB
    registry=REGISTRY
)

api_response_size_bytes = Histogram(
    'api_response_size_bytes',
    'API response size in bytes',
    ['method', 'endpoint'],
    buckets=[1024, 10240, 102400, 1048576, 10485760],
    registry=REGISTRY
)

# Model Inference Metrics
inference_requests_total = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['model_format', 'prediction_type'],  # single/batch
    registry=REGISTRY
)

inference_duration_seconds = Histogram(
    'inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_format', 'prediction_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

inference_preprocessing_duration_seconds = Histogram(
    'inference_preprocessing_duration_seconds',
    'Image preprocessing duration in seconds',
    ['prediction_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY
)

inference_confidence_score = Histogram(
    'inference_confidence_score',
    'Model confidence scores',
    ['predicted_class'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    registry=REGISTRY
)

# Image Processing Metrics
images_processed_total = Counter(
    'images_processed_total',
    'Total number of images processed',
    ['status'],  # success/failed
    registry=REGISTRY
)

image_size_bytes = Histogram(
    'image_size_bytes',
    'Size of uploaded images in bytes',
    buckets=[10240, 102400, 1048576, 5242880, 10485760, 52428800],  # 10KB to 50MB
    registry=REGISTRY
)

batch_size_images = Histogram(
    'batch_size_images',
    'Number of images in batch requests',
    buckets=[1, 2, 5, 10, 20, 50, 100],
    registry=REGISTRY
)

# System Resource Metrics
active_sessions_gauge = Gauge(
    'active_inference_sessions',
    'Number of active ONNX inference sessions',
    registry=REGISTRY
)

memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component'],  # api/model/preprocessing
    registry=REGISTRY
)

model_load_time_seconds = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model in seconds',
    registry=REGISTRY
)

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'component'],
    registry=REGISTRY
)

# Model Information
model_info = Info(
    'model_info',
    'Information about the loaded model',
    registry=REGISTRY
)

# Application Status
app_status = Enum(
    'app_status',
    'Current application status',
    states=['starting', 'ready', 'degraded', 'error'],
    registry=REGISTRY
)

# Cache Metrics (if caching is enabled)
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'result'],  # get/set/delete, hit/miss/error
    registry=REGISTRY
)

cache_size_items = Gauge(
    'cache_size_items',
    'Number of items in cache',
    registry=REGISTRY
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Centralized metrics collection and management"""
    
    def __init__(self):
        self.start_time = time.time()
        self._setup_initial_metrics()
    
    def _setup_initial_metrics(self):
        """Initialize metrics with default values"""
        app_status.state('starting')
        active_sessions_gauge.set(0)
        cache_size_items.set(0)
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, 
                          duration: float, request_size: int = 0, response_size: int = 0):
        """Record API request metrics"""
        api_requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        api_request_duration_seconds.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            api_request_size_bytes.labels(
                method=method, 
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            api_response_size_bytes.labels(
                method=method, 
                endpoint=endpoint
            ).observe(response_size)
    
    def record_inference(self, model_format: str, prediction_type: str, 
                        duration: float, confidence: float = None, 
                        predicted_class: str = None):
        """Record inference metrics"""
        inference_requests_total.labels(
            model_format=model_format,
            prediction_type=prediction_type
        ).inc()
        
        inference_duration_seconds.labels(
            model_format=model_format,
            prediction_type=prediction_type
        ).observe(duration)
        
        if confidence is not None and predicted_class:
            inference_confidence_score.labels(
                predicted_class=predicted_class
            ).observe(confidence)
    
    def record_preprocessing(self, prediction_type: str, duration: float):
        """Record preprocessing metrics"""
        inference_preprocessing_duration_seconds.labels(
            prediction_type=prediction_type
        ).observe(duration)
    
    def record_image_processing(self, status: str, image_size: int = None, 
                               batch_size: int = None):
        """Record image processing metrics"""
        images_processed_total.labels(status=status).inc()
        
        if image_size:
            image_size_bytes.observe(image_size)
        
        if batch_size:
            batch_size_images.observe(batch_size)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics"""
        errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def update_model_info(self, model_data: Dict[str, str]):
        """Update model information"""
        model_info.info(model_data)
    
    def update_app_status(self, status: str):
        """Update application status"""
        if status in ['starting', 'ready', 'degraded', 'error']:
            app_status.state(status)
    
    def update_active_sessions(self, count: int):
        """Update active sessions count"""
        active_sessions_gauge.set(count)
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage"""
        memory_usage_bytes.labels(component=component).set(bytes_used)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation"""
        cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def update_cache_size(self, size: int):
        """Update cache size"""
        cache_size_items.set(size)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(REGISTRY)
    
    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics"""
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_api_metrics(func: Callable) -> Callable:
    """Decorator to automatically track API metrics"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        request = None
        
        # Find request object in args/kwargs
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        if not request:
            request = kwargs.get('request')
        
        method = request.method if request else 'UNKNOWN'
        endpoint = request.url.path if request else 'UNKNOWN'
        status_code = 200
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status_code = getattr(e, 'status_code', 500)
            metrics_collector.record_error(
                error_type=type(e).__name__,
                component='api'
            )
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_api_request(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                duration=duration
            )
    
    return wrapper


def track_inference_metrics(model_format: str, prediction_type: str):
    """Decorator to track inference metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract confidence and class from result if available
                confidence = None
                predicted_class = None
                
                if isinstance(result, dict):
                    confidence = result.get('confidence')
                    predicted_class = result.get('class_name')
                elif isinstance(result, list) and result:
                    # For batch results, use first result
                    first_result = result[0]
                    if isinstance(first_result, dict):
                        confidence = first_result.get('confidence')
                        predicted_class = first_result.get('class_name')
                
                duration = time.time() - start_time
                metrics_collector.record_inference(
                    model_format=model_format,
                    prediction_type=prediction_type,
                    duration=duration,
                    confidence=confidence,
                    predicted_class=predicted_class
                )
                
                return result
                
            except Exception as e:
                metrics_collector.record_error(
                    error_type=type(e).__name__,
                    component='inference'
                )
                raise
        
        return wrapper
    return decorator


def track_preprocessing_metrics(prediction_type: str):
    """Decorator to track preprocessing metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics_collector.record_preprocessing(prediction_type, duration)
                return result
            except Exception as e:
                metrics_collector.record_error(
                    error_type=type(e).__name__,
                    component='preprocessing'
                )
                raise
        
        return wrapper
    return decorator