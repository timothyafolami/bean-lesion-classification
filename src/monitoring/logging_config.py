"""
Structured logging configuration for Bean Classification API
"""

import os
import sys
import json
import logging
import logging.config
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from loguru import logger as loguru_logger


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Enhanced text formatter with colors and structure"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and structure"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Add color if terminal supports it
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET'] if color else ''
        
        # Base format
        formatted = (
            f"{timestamp} | "
            f"{color}{record.levelname:8}{reset} | "
            f"{record.name:20} | "
            f"{record.getMessage()}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_file: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> None:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
        log_file: Path to log file (optional)
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
    """
    
    # Create logs directory if it doesn't exist
    if log_file and enable_file_logging:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure formatters
    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # Setup handlers
    handlers = {}
    
    if enable_console_logging:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'default',
            'stream': sys.stdout
        }
    
    if enable_file_logging and log_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'default',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }
        
        # Separate error log file
        error_log_file = str(Path(log_file).with_suffix('.error.log'))
        handlers['error_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'default',
            'filename': error_log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 3,
            'encoding': 'utf8'
        }
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                '()': JSONFormatter if log_format.lower() == "json" else TextFormatter,
            }
        },
        'handlers': handlers,
        'loggers': {
            # Application loggers
            'src': {
                'level': log_level,
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'bean_classification': {
                'level': log_level,
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            # FastAPI and Uvicorn loggers
            'fastapi': {
                'level': log_level,
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'uvicorn': {
                'level': log_level,
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'WARNING',  # Reduce noise from access logs
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            # Third-party loggers
            'onnxruntime': {
                'level': 'WARNING',
                'handlers': list(handlers.keys()),
                'propagate': False
            },
            'PIL': {
                'level': 'WARNING',
                'handlers': list(handlers.keys()),
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': list(handlers.keys())
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Setup loguru for additional features
    setup_loguru(log_level, log_format, log_file)
    
    # Log configuration info
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={log_format}, file={log_file}")


def setup_loguru(log_level: str, log_format: str, log_file: Optional[str] = None):
    """Setup loguru for enhanced logging features"""
    
    # Remove default handler
    loguru_logger.remove()
    
    # Console handler
    if log_format.lower() == "json":
        console_format = (
            '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", "logger": "{name}", "message": "{message}", '
            '"module": "{module}", "function": "{function}", "line": {line}}'
        )
    else:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    loguru_logger.add(
        sys.stdout,
        format=console_format,
        level=log_level,
        colorize=log_format.lower() != "json"
    )
    
    # File handler
    if log_file:
        loguru_logger.add(
            log_file,
            format=console_format,
            level=log_level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            serialize=log_format.lower() == "json"
        )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance"""
    return logging.getLogger(name)


def log_system_info():
    """Log system information at startup"""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "disk_usage_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
    }
    
    logger.info("System Information", extra=system_info)


def log_application_startup(app_name: str, version: str, config: Dict[str, Any]):
    """Log application startup information"""
    logger = get_logger(__name__)
    
    startup_info = {
        "event": "application_startup",
        "app_name": app_name,
        "version": version,
        "config": config,
        "pid": os.getpid()
    }
    
    logger.info("Application Starting", extra=startup_info)


# Context manager for request logging
class RequestLogger:
    """Context manager for request-specific logging"""
    
    def __init__(self, request_id: str, endpoint: str, method: str):
        self.request_id = request_id
        self.endpoint = endpoint
        self.method = method
        self.logger = get_logger(f"request.{request_id}")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            f"Request started: {self.method} {self.endpoint}",
            extra={
                "request_id": self.request_id,
                "event": "request_start",
                "method": self.method,
                "endpoint": self.endpoint
            }
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type:
            self.logger.error(
                f"Request failed: {self.method} {self.endpoint} ({duration:.3f}s)",
                extra={
                    "request_id": self.request_id,
                    "event": "request_error",
                    "method": self.method,
                    "endpoint": self.endpoint,
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val)
                },
                exc_info=True
            )
        else:
            self.logger.info(
                f"Request completed: {self.method} {self.endpoint} ({duration:.3f}s)",
                extra={
                    "request_id": self.request_id,
                    "event": "request_complete",
                    "method": self.method,
                    "endpoint": self.endpoint,
                    "duration_seconds": duration
                }
            )