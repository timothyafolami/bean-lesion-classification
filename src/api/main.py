"""
FastAPI backend for bean lesion classification.
Supports both ONNX and PyTorch model formats.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import Dict, Any, Optional

from src.api.models import ModelManager
from src.api.routes import prediction_router, health_router
from src.utils.logging_config import setup_logging, api_logger
from src.utils.config import load_api_config, get_settings


# Global model manager instance
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    global model_manager
    
    # Startup
    api_logger.info("🚀 Starting Bean Lesion Classification API...")
    
    try:
        # Load configuration
        config = load_api_config()
        settings = get_settings()
        
        # Initialize model manager
        model_manager = ModelManager(
            model_path=settings.model_path,
            model_format=getattr(settings, 'model_format', 'auto'),  # auto, onnx, pytorch
            architecture=getattr(settings, 'architecture', 'efficientnet_b0'),
            device=getattr(settings, 'device', 'auto')
        )
        
        # Load the model
        await model_manager.load_model()
        
        api_logger.info("✅ Model loaded successfully")
        api_logger.info(f"Model format: {model_manager.model_format}")
        api_logger.info(f"Model path: {model_manager.model_path}")
        
    except Exception as e:
        api_logger.error(f"❌ Failed to initialize model: {e}")
        raise
    
    yield
    
    # Shutdown
    api_logger.info("🛑 Shutting down Bean Lesion Classification API...")
    if model_manager:
        await model_manager.cleanup()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    # Load configuration
    try:
        config = load_api_config()
        settings = get_settings()
    except Exception as e:
        # Fallback configuration if config files are missing
        api_logger.warning(f"Could not load configuration: {e}, using defaults")
        config = None
        settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="Bean Lesion Classification API",
        description="AI-powered bean leaf disease classification system supporting both ONNX and PyTorch models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, config)
    
    # Add routes
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])
    
    # Add global exception handler
    setup_exception_handlers(app)
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Bean Lesion Classification API",
            "version": "1.0.0",
            "status": "running",
            "model_format": model_manager.model_format if model_manager else "not_loaded",
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


def setup_middleware(app: FastAPI, config: Optional[Dict[str, Any]]):
    """
    Set up middleware for the FastAPI application.
    """
    # CORS middleware
    if config and 'cors' in config:
        cors_config = config['cors']
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get('allow_origins', ["*"]),
            allow_credentials=cors_config.get('allow_credentials', True),
            allow_methods=cors_config.get('allow_methods', ["*"]),
            allow_headers=cors_config.get('allow_headers', ["*"]),
        )
    else:
        # Default CORS settings
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*"]
    )


def setup_exception_handlers(app: FastAPI):
    """
    Set up global exception handlers.
    """
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc: HTTPException):
        """Handle HTTP exceptions."""
        api_logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "http_error"
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Handle general exceptions."""
        api_logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "type": "server_error"
                }
            }
        )


def get_model_manager() -> ModelManager:
    """
    Dependency to get the global model manager instance.
    """
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    return model_manager


# Create the app instance
app = create_app()


def main():
    """
    Main function to run the API server.
    """
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Get settings
    settings = get_settings()
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()