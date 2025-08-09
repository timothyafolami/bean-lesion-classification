"""
Configuration management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .logging_config import utils_logger


class TrainingConfig(BaseModel):
    """Training configuration model."""
    model: Dict[str, Any]
    training: Dict[str, Any]
    data: Dict[str, Any]
    augmentation: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    device: str = "cuda"
    seed: int = 42


class APIConfig(BaseModel):
    """API configuration model."""
    server: Dict[str, Any]
    model: Dict[str, Any]
    upload: Dict[str, Any]
    cors: Dict[str, Any]
    logging: Dict[str, Any]
    monitoring: Dict[str, Any]


class InferenceConfig(BaseModel):
    """Inference configuration model."""
    model: Dict[str, Any]
    preprocessing: Dict[str, Any]
    postprocessing: Dict[str, Any]
    performance: Dict[str, Any]


class Settings(BaseSettings):
    """Environment-based settings."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Paths
    config_dir: str = Field(default="config", env="CONFIG_DIR")
    models_dir: str = Field(default="models", env="MODELS_DIR")
    logs_dir: str = Field(default="logs", env="LOGS_DIR")
    data_dir: str = Field(default=".", env="DATA_DIR")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    
    # Model Settings
    model_path: Optional[str] = Field(default=None, env="MODEL_PATH")
    model_format: str = Field(default="auto", env="MODEL_FORMAT")  # auto, onnx, pytorch
    architecture: str = Field(default="efficientnet_b0", env="ARCHITECTURE")
    device: str = Field(default="auto", env="DEVICE")  # auto, cpu, cuda, mps
    
    # Security Configuration
    cors_origins: str = Field(default="http://localhost:3000", env="CORS_ORIGINS")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=8001, env="METRICS_PORT")
    prometheus_enabled: bool = Field(default=False, env="PROMETHEUS_ENABLED")
    
    # Grafana Configuration
    grafana_password: str = Field(default="admin123", env="GRAFANA_PASSWORD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        utils_logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        utils_logger.info(f"Loaded configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        utils_logger.error(f"Error parsing YAML config file {config_path}: {e}")
        raise


def load_training_config(config_path: str = "config/training_config.yaml") -> TrainingConfig:
    """Load and validate training configuration."""
    config_data = load_yaml_config(config_path)
    return TrainingConfig(**config_data)


def load_api_config(config_path: str = "config/api_config.yaml") -> APIConfig:
    """Load and validate API configuration."""
    config_data = load_yaml_config(config_path)
    return APIConfig(**config_data)


def load_inference_config(config_path: str = "config/inference_config.yaml") -> InferenceConfig:
    """Load and validate inference configuration."""
    config_data = load_yaml_config(config_path)
    return InferenceConfig(**config_data)


def get_settings() -> Settings:
    """Get application settings from environment variables."""
    return Settings()


def ensure_directories(settings: Settings) -> None:
    """Ensure required directories exist."""
    directories = [
        settings.models_dir,
        settings.logs_dir,
        "data/processed",
        "data/raw"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        utils_logger.debug(f"Ensured directory exists: {directory}")


# Global settings instance
settings = get_settings()