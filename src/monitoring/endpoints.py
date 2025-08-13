"""
Monitoring and metrics endpoints
"""

import time
import psutil
import platform
from typing import Dict, Any, List
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import PlainTextResponse

from .metrics import metrics_collector, REGISTRY
from .logging_config import get_logger

logger = get_logger(__name__)

# Use a single, consistent tag to avoid duplicate sections in API docs
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format
    """
    try:
        metrics_data = metrics_collector.get_metrics()
        return PlainTextResponse(
            content=metrics_data,
            media_type=metrics_collector.get_metrics_content_type(),
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating metrics")


@router.get("/health/detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """
    Detailed health check with system metrics
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=1)

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        health_data = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - metrics_collector.start_time,
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_usage_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent,
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2),
                },
            },
            "process": {
                "pid": process.pid,
                "memory_mb": round(process_memory.rss / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections()),
            },
        }

        # Check for potential issues
        warnings = []
        if memory.percent > 90:
            warnings.append("High memory usage")
        if cpu_percent > 90:
            warnings.append("High CPU usage")
        if disk.free < (1024**3):  # Less than 1GB free
            warnings.append("Low disk space")

        if warnings:
            health_data["warnings"] = warnings
            health_data["status"] = "degraded"

        return health_data

    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {"status": "error", "timestamp": time.time(), "error": str(e)}


@router.get("/stats")
async def get_application_stats() -> Dict[str, Any]:
    """
    Get application statistics and performance metrics
    """
    try:
        # This would typically come from your metrics collector
        # For now, we'll return some basic stats
        stats = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - metrics_collector.start_time,
            "requests": {
                "total": "See /monitoring/metrics for detailed request metrics",
                "active": "Check active_requests_gauge metric",
            },
            "inference": {
                "total": "See /monitoring/metrics for inference metrics",
                "average_duration": "Check inference_duration_seconds metric",
            },
            "errors": {"total": "See /monitoring/metrics for error metrics"},
            "memory": {
                "process_mb": round(psutil.Process().memory_info().rss / (1024**2), 2)
            },
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting application stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting application stats")


@router.get("/logs/recent")
async def get_recent_logs(lines: int = 100) -> Dict[str, Any]:
    """
    Get recent log entries (if file logging is enabled)
    """
    try:
        import os
        from pathlib import Path

        log_file = os.getenv("LOG_FILE", "./logs/app.log")
        log_path = Path(log_file)

        if not log_path.exists():
            return {"message": "Log file not found", "log_file": str(log_path)}

        # Read last N lines
        with open(log_path, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "log_file": str(log_path),
            "total_lines": len(all_lines),
            "returned_lines": len(recent_lines),
            "logs": [line.strip() for line in recent_lines],
        }

    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reading logs")


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get performance-related metrics
    """
    try:
        process = psutil.Process()

        # CPU and memory over time (simple snapshot)
        cpu_times = process.cpu_times()
        memory_info = process.memory_info()

        performance_data = {
            "timestamp": time.time(),
            "cpu": {
                "percent": process.cpu_percent(),
                "user_time": cpu_times.user,
                "system_time": cpu_times.system,
            },
            "memory": {
                "rss_mb": round(memory_info.rss / (1024**2), 2),
                "vms_mb": round(memory_info.vms / (1024**2), 2),
                "percent": process.memory_percent(),
            },
            "io": {
                "read_count": process.io_counters().read_count,
                "write_count": process.io_counters().write_count,
                "read_bytes": process.io_counters().read_bytes,
                "write_bytes": process.io_counters().write_bytes,
            },
            "threads": process.num_threads(),
            "file_descriptors": (
                process.num_fds() if hasattr(process, "num_fds") else None
            ),
        }

        return performance_data

    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error getting performance metrics")


@router.post("/alerts/test")
async def test_alert_system() -> Dict[str, Any]:
    """
    Test the alerting system by generating test metrics
    """
    try:
        # Generate some test metrics
        metrics_collector.record_error("TestError", "monitoring")
        metrics_collector.record_api_request("POST", "/test", 500, 1.0)

        logger.warning(
            "Test alert generated",
            extra={
                "event": "test_alert",
                "component": "monitoring",
                "severity": "warning",
            },
        )

        return {
            "message": "Test alert generated successfully",
            "timestamp": time.time(),
            "metrics_recorded": ["test_error_metric", "test_api_request_metric"],
        }

    except Exception as e:
        logger.error(f"Error generating test alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating test alert")


@router.get("/config")
async def get_monitoring_config() -> Dict[str, Any]:
    """
    Get current monitoring configuration
    """
    import os

    config = {
        "metrics_enabled": os.getenv("METRICS_ENABLED", "true").lower() == "true",
        "prometheus_enabled": os.getenv("PROMETHEUS_ENABLED", "false").lower()
        == "true",
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "log_format": os.getenv("LOG_FORMAT", "text"),
        "log_file": os.getenv("LOG_FILE"),
        "environment": os.getenv("ENVIRONMENT", "development"),
    }

    return config
