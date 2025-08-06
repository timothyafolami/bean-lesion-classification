# Bean Lesion Classification - Development Makefile

.PHONY: help install install-dev setup clean test lint format run-api run-train docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Initial project setup"
	@echo "  clean        - Clean up generated files"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  run-api      - Run FastAPI server"
	@echo "  run-train    - Run training pipeline"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run with Docker Compose"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Setup
setup: install-dev
	@echo "Setting up project..."
	@mkdir -p models logs data/processed data/raw experiments
	@cp .env.example .env
	@echo "Project setup complete!"
	@echo "Please edit .env file with your configuration"

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Running
run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-train:
	python -m src.training.train

# Docker
docker-build:
	docker build -t bean-classification-api -f docker/Dockerfile.api .
	docker build -t bean-classification-frontend -f docker/Dockerfile.frontend ./frontend

docker-run:
	docker-compose up --build

docker-stop:
	docker-compose down

# Development helpers
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Model operations
convert-model:
	python -m src.inference.convert --model-path models/best_model.pth --output-path models/best_model.onnx

benchmark:
	python -m src.inference.benchmark --model-path models/best_model.onnx

# Data operations
prepare-data:
	python -m src.training.data_preparation

validate-data:
	python -m src.training.data_validation