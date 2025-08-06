"""
FastAPI middleware for monitoring and metrics collection
"""

import time
import logging
import json
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .metrics import metrics_collector

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API metrics automatically"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics collection for metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)
        
        start_time = time.time()
        method = request.method
        endpoint = request.url.path
        
        # Get request size
        request_size = 0
        if hasattr(request, 'headers'):
            content_length = request.headers.get('content-length')
            if content_length:
                try:
                    request_size = int(content_length)
                except ValueError:
                    pass
        
        status_code = 200
        response_size = 0
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Get response size
            if hasattr(response, 'headers'):
                content_length = response.headers.get('content-length')
                if content_length:
                    try:
                        response_size = int(content_length)
                    except ValueError:
                        pass
            
            return response
            
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
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging"""
    
    def __init__(self, app: ASGIApp, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip logging for health checks and metrics
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        request_log = {
            "event": "request_started",
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "timestamp": start_time
        }
        
        # Remove sensitive headers
        sensitive_headers = ['authorization', 'cookie', 'x-api-key']
        for header in sensitive_headers:
            if header in request_log["headers"]:
                request_log["headers"][header] = "[REDACTED]"
        
        logger.log(self.log_level, "API Request", extra=request_log)
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log response
            response_log = {
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": duration,
                "response_headers": dict(response.headers),
                "timestamp": time.time()
            }
            
            logger.log(self.log_level, "API Response", extra=response_log)
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            error_log = {
                "event": "request_failed",
                "method": request.method,
                "path": request.url.path,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration_seconds": duration,
                "timestamp": time.time()
            }
            
            logger.error("API Error", extra=error_log)
            raise


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error tracking and alerting"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_counts = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            
            # Track 4xx and 5xx responses as errors
            if response.status_code >= 400:
                error_type = f"http_{response.status_code}"
                self._track_error(error_type, request.url.path)
            
            return response
            
        except Exception as e:
            error_type = type(e).__name__
            self._track_error(error_type, request.url.path)
            
            # Create structured error response
            error_response = {
                "error": {
                    "type": error_type,
                    "message": str(e),
                    "path": request.url.path,
                    "method": request.method,
                    "timestamp": time.time()
                }
            }
            
            # Determine status code
            status_code = getattr(e, 'status_code', 500)
            
            return JSONResponse(
                content=error_response,
                status_code=status_code
            )
    
    def _track_error(self, error_type: str, path: str):
        """Track error occurrence"""
        key = f"{error_type}:{path}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Record in metrics
        metrics_collector.record_error(error_type, "api")
        
        # Log high-frequency errors
        if self.error_counts[key] % 10 == 0:  # Every 10th occurrence
            logger.warning(
                f"High frequency error detected: {error_type} on {path} "
                f"(count: {self.error_counts[key]})"
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and alerting"""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Alert on slow requests
            if duration > self.slow_request_threshold:
                logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} "
                    f"took {duration:.2f}s (threshold: {self.slow_request_threshold}s)"
                )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed request: {request.method} {request.url.path} "
                f"failed after {duration:.2f}s with {type(e).__name__}: {str(e)}"
            )
            raise