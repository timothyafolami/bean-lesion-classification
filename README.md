# Bean Lesion Classification System

An end-to-end machine learning system for classifying bean leaf diseases using deep learning. The system supports training multiple CNN architectures, ONNX optimization for fast inference, and provides both API and web interfaces for real-time classification.

## End-to-End Pipeline Overview

This project implements a robust, production-grade pipeline for bean lesion classification, following best practices in machine learning and software engineering. The pipeline includes:

- **Data Validation & EDA**: Automated scripts validate CSVs, check image integrity, analyze class balance, and generate dataset statistics and visualizations. EDA notebooks and scripts help understand data distribution and quality.
- **Flexible Data Loading**: Custom PyTorch `Dataset` and `DataLoader` classes with configurable augmentations, memory optimization, and error handling.
- **Model Training**: Modular training scripts support multiple architectures (ResNet, EfficientNet, VGG, DenseNet), with options for hyperparameter tuning, early stopping, learning rate scheduling, mixed precision, and experiment tracking. Each run is logged and reproducible.
- **Model Evaluation & Comparison**: Automated evaluation scripts compare models on accuracy, F1, per-class metrics, and inference speed. Visualizations (confusion matrix, ROC curves) are generated for each model.
- **ONNX Conversion & Validation**: Trained PyTorch models are exported to ONNX format for optimized inference. Output consistency between PyTorch and ONNX is validated.
- **FastAPI Backend**: A production-ready REST API wraps the ONNX model for fast, scalable inference. Endpoints support single and batch image prediction, with robust error handling, logging, and CORS for frontend integration.
- **React Frontend**: A modern React app (with Material-UI/Tailwind) provides a drag-and-drop interface for uploading single or multiple images, displaying predictions and confidence scores, and visualizing results. No login required.
- **Testing & Quality**: Comprehensive unit and integration tests cover data loading, model inference, and API endpoints. Linting and formatting are enforced for code quality.
- **Dockerization**: Both backend and frontend are containerized for easy deployment. Docker Compose scripts orchestrate the full stack.
- **Monitoring & Logging**: Training, inference, and API logs are collected. Metrics (accuracy, latency, resource usage) are tracked for production monitoring.
- **Software Engineering Practices**: The codebase is modular, well-documented, and follows best practices (env configs, version control, CI-ready structure, clear folder organization).

See below for details on each pipeline stage and how to run or extend them.

## Features

- **Multi-Architecture Training**: Support for ResNet, EfficientNet, VGG, and DenseNet
- **ONNX Optimization**: Convert PyTorch models to optimized ONNX format for faster inference
- **FastAPI Backend**: Production-ready REST API with comprehensive error handling
- **React Frontend**: Modern web interface with drag-and-drop image upload
- **Batch Processing**: Support for single and multiple image classification
- **Comprehensive Testing**: Unit and integration tests with good coverage
- **Docker Support**: Containerized deployment for easy scaling
- **Monitoring**: Built-in logging and metrics for production monitoring

## Disease Classes

The system classifies bean leaves into three categories:
- **Healthy**: Normal, disease-free bean leaves
- **Angular Leaf Spot**: Caused by *Pseudocercospora griseola*
- **Bean Rust**: Caused by *Uromyces appendiculatus*

## Project Structure

```
bean-lesion-classification/
├── src/
│   ├── training/          # Model training pipeline
│   ├── inference/         # ONNX inference engine
│   ├── api/              # FastAPI backend
│   └── utils/            # Utility functions
├── frontend/             # React web application
├── config/               # Configuration files
├── models/               # Trained models and checkpoints
├── tests/                # Test suites
├── logs/                 # Application logs
├── data/                 # Dataset (train/val folders)
├── docker/               # Docker configurations
└── docs/                 # Documentation

```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- CUDA-compatible GPU (optional, for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bean-lesion-classification
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   make install-dev
   # or manually:
   pip install -r requirements.txt
   ```

4. **Initial setup**
   ```bash
   make setup
   ```

5. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Data Loading Pipeline

### Dataset Architecture

The data loading pipeline is built with PyTorch's DataLoader and includes:

- **BeanLesionDataset**: Custom dataset class with validation and statistics
- **Flexible Transforms**: Configurable augmentation pipeline
- **Batch Processing**: Efficient multi-threaded data loading
- **Memory Optimization**: Smart caching and preprocessing
- **Error Handling**: Robust handling of corrupted or missing images

### Data Loading Features

```python
from src.training.dataset import create_data_loaders, BeanLesionDataset

# Create data loaders with custom configuration
train_loader, val_loader = create_data_loaders(
    train_csv="train.csv",
    val_csv="val.csv",
    batch_size=32,
    image_size=(224, 224),
    augmentation_config=augmentation_config,
    num_workers=4,
    root_dir="."
)
```

**Key Features:**
- **Automatic Validation**: Checks image integrity during loading
- **Class Distribution**: Automatic calculation of class balance
- **Flexible Transforms**: Training vs validation transform pipelines
- **Memory Efficient**: Optimized for large datasets
- **Multi-Processing**: Parallel data loading for faster training

### Image Preprocessing

The pipeline automatically handles:

1. **Image Loading**: PIL-based loading with error handling
2. **Resizing**: Intelligent resizing maintaining aspect ratios
3. **Normalization**: ImageNet-standard normalization
4. **Augmentation**: Configurable training-time augmentations
5. **Tensor Conversion**: Automatic PyTorch tensor conversion

### Data Pipeline Testing

Test your data pipeline before training:

```bash
# Test basic dataset functionality
python -m src.training.test_data_pipeline

# Validate data integrity
python -c "
from src.training.data_validation import validate_dataset
results = validate_dataset()
print(f'Validation Status: {results[\"summary\"][\"overall_status\"]}')
"

# Generate dataset statistics
python -c "
from src.training.data_statistics import generate_dataset_statistics
stats = generate_dataset_statistics()
print(f'Total Samples: {stats[\"summary\"][\"dataset_size\"][\"total_samples\"]}')
"
```

## Data Pipeline and Training

### Data Structure

The system expects the following data structure:

```
bean-lesion-classification/
├── train/
│   ├── healthy/
│   ├── angular_leaf_spot/
│   └── bean_rust/
├── val/
│   ├── healthy/
│   ├── angular_leaf_spot/
│   └── bean_rust/
├── train.csv          # Training metadata
├── val.csv            # Validation metadata
└── classname.txt      # Class names mapping
```

### Data Validation and Statistics

Before training, validate your dataset:

```bash
# Run comprehensive data validation
python -m src.training.data_validation

# Generate dataset statistics
python -m src.training.data_statistics

# Test data pipeline
python -m src.training.test_data_pipeline
```

**Data Validation Features:**
- CSV file structure validation
- Image integrity checking
- Class balance analysis
- Directory structure verification
- Comprehensive reporting with visualizations

### Training Pipeline

#### 1. Single Model Training

Train a single model with the main training script:

```bash
# Train EfficientNet-B0 (recommended)
python train_model.py --architecture efficientnet_b0 --epochs 10

# Train ResNet50 with custom settings
python train_model.py --architecture resnet50 --epochs 15 --batch-size 16 --learning-rate 0.0005

# Train with different architectures
python train_model.py --architecture densenet121 --epochs 20 --pretrained
```

**Available Architectures:**
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`, `efficientnet_b4`
- `vgg11`, `vgg13`, `vgg16`, `vgg19`
- `densenet121`, `densenet161`, `densenet169`, `densenet201`

**Training Options:**
```bash
python train_model.py --help

Options:
  --architecture     Model architecture (default: efficientnet_b0)
  --epochs          Number of epochs (default: 10)
  --batch-size      Batch size (default: 32)
  --learning-rate   Learning rate (default: 0.001)
  --pretrained      Use pretrained weights (default: True)
  --save-dir        Directory to save models (default: models)
  --seed            Random seed (default: 42)
```

#### 2. Multi-Architecture Training

Compare multiple architectures automatically:

```bash
# Train and compare multiple architectures
python train_multiple_architectures.py
```

This will train:
- EfficientNet-B0 (balanced accuracy/speed)
- ResNet50 (reliable baseline)
- DenseNet121 (parameter efficient)
- ResNet18 (fastest training)

#### 3. Model Comparison and Selection

After training, analyze and compare your models:

```bash
# Compare all trained models
python check_best_model.py
```

**Output includes:**
- Performance ranking by validation accuracy
- Detailed metrics for each model
- Per-class performance analysis
- Training time comparison
- Recommendations for production use
- Next steps for ONNX conversion

### Training Features

#### Advanced Training Capabilities

- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Step, plateau, and cosine annealing schedulers
- **Data Augmentation**: Configurable augmentation pipeline
- **Mixed Precision**: Automatic mixed precision for faster training
- **Checkpointing**: Comprehensive model checkpoints with full state
- **Metrics Tracking**: Real-time training metrics and visualization

#### Data Augmentation

Configured in `config/training_config.yaml`:

```yaml
augmentation:
  horizontal_flip: 0.5      # 50% chance of horizontal flip
  vertical_flip: 0.3        # 30% chance of vertical flip
  rotation: 15              # Random rotation up to 15 degrees
  brightness: 0.2           # Brightness variation
  contrast: 0.2             # Contrast variation
  saturation: 0.2           # Saturation variation
  hue: 0.1                  # Hue variation
```

#### Training Configuration

Customize training in `config/training_config.yaml`:

```yaml
model:
  architectures: ["resnet50", "efficientnet_b0"]
  num_classes: 3
  pretrained: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 10
  weight_decay: 0.0001

optimizer:
  type: "adam"              # adam, adamw, sgd
  momentum: 0.9             # for SGD

scheduler:
  type: "step"              # step, plateau, cosine
  step_size: 20
  gamma: 0.1
```

### Training Output

After training, you'll find in `models/{architecture}_{timestamp}/`:

```
models/efficientnet_b0_20250806_035425/
├── best_model.pth              # Best model checkpoint
├── training_config.json        # Training configuration
├── training_history.json       # Loss/accuracy history
├── training_history.png        # Training plots
├── final_results.json          # Complete results
└── evaluation/                 # Evaluation plots
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── evaluation_metrics.json
```

### Model Performance Tracking

The training system automatically tracks:

- **Training Metrics**: Loss, accuracy per epoch
- **Validation Metrics**: Loss, accuracy, F1-score, precision, recall
- **Per-Class Metrics**: Individual class performance
- **Training Time**: Total and per-epoch timing
- **Model Size**: Parameter count and file size
- **Hardware Usage**: GPU/CPU utilization

### Quick Training Examples

```bash
# Quick test training (2 epochs)
python train_simple.py

# Production training with EfficientNet-B0
python train_model.py --architecture efficientnet_b0 --epochs 25

# Fast training with ResNet18
python train_model.py --architecture resnet18 --epochs 15 --batch-size 64

# High-accuracy training with larger model
python train_model.py --architecture efficientnet_b3 --epochs 30 --batch-size 16
```

### Training Monitoring and Debugging

#### Real-time Monitoring

During training, monitor progress through:

```bash
# Watch training logs in real-time
tail -f logs/training.log

# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi

# Check training progress
ls -la models/  # See active training directories
```

#### Training Visualization

The system automatically generates:

- **Training Curves**: Loss and accuracy over epochs
- **Learning Rate Schedule**: LR changes during training
- **Validation Metrics**: Per-epoch validation performance
- **Confusion Matrix**: Final model performance breakdown

#### Debugging Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python train_model.py --batch-size 16

# Use smaller image size (modify config/training_config.yaml)
# Change image_size: [224, 224] to [128, 128]
```

**Poor Convergence:**
```bash
# Try different learning rate
python train_model.py --learning-rate 0.0001

# Use different optimizer (modify config)
# Change optimizer type to 'adamw' or 'sgd'
```

**Overfitting:**
```bash
# Increase data augmentation (modify config)
# Add more dropout (modify model config)
# Use early stopping (automatic)
```

#### Performance Optimization

**For Faster Training:**
- Use smaller models (ResNet18, EfficientNet-B0)
- Increase batch size if memory allows
- Use multiple workers: `num_workers=4`
- Enable mixed precision training

**For Better Accuracy:**
- Use larger models (ResNet50, EfficientNet-B2)
- Train for more epochs
- Use stronger data augmentation
- Ensemble multiple models

### Training Best Practices

1. **Start Small**: Begin with ResNet18 or EfficientNet-B0 for quick validation
2. **Use Pretrained Weights**: Always use `--pretrained` for better convergence
3. **Monitor Overfitting**: Watch validation vs training loss curves
4. **Adjust Batch Size**: Reduce if you get CUDA out of memory errors
5. **Early Stopping**: Let the system stop automatically when performance plateaus
6. **Compare Architectures**: Use multi-architecture training to find the best model
7. **Validate Data First**: Always run data validation before training
8. **Save Experiments**: Each training run creates a timestamped directory
9. **Monitor Resources**: Keep an eye on GPU/CPU usage during training
10. **Test Pipeline**: Use `train_simple.py` to test setup before long training runs

## Model Evaluation and Analysis

### Comprehensive Model Evaluation

After training, the system provides detailed evaluation metrics and visualizations:

```bash
# Evaluate a specific model
python evaluate_model.py --model-path models/efficientnet_b0_20250806_035425/best_model.pth

# Compare multiple models
python compare_models.py --models-dir models/

# Generate evaluation report
python generate_evaluation_report.py
```

**Evaluation Features:**
- **Confusion Matrix**: Visual breakdown of classification performance
- **ROC Curves**: Per-class and macro-averaged ROC analysis
- **Precision-Recall Curves**: Detailed performance for each disease class
- **Class-wise Metrics**: Precision, recall, F1-score for each class
- **Error Analysis**: Misclassified samples with confidence scores
- **Performance vs Model Size**: Trade-off analysis for deployment decisions

### Model Comparison Dashboard

The system automatically generates comparison reports:

```bash
# Generate interactive comparison dashboard
python create_model_dashboard.py

# Export comparison to PDF report
python export_model_report.py --format pdf
```

**Dashboard includes:**
- Performance ranking across all metrics
- Training time vs accuracy trade-offs
- Model size vs inference speed analysis
- Resource utilization comparison
- Deployment readiness assessment

## ONNX Conversion and Optimization

### Converting PyTorch Models to ONNX

Convert trained models for optimized inference:

```bash
# Convert best model to ONNX
python convert_to_onnx.py --model-path models/efficientnet_b0_20250806_035425/best_model.pth

# Convert with specific optimization
python convert_to_onnx.py --model-path models/resnet50_20250806_035425/best_model.pth --optimize --quantize

# Batch convert all models
python batch_convert_onnx.py --models-dir models/
```

**ONNX Conversion Features:**
- **Automatic Optimization**: Graph optimization for faster inference
- **Quantization Support**: INT8 quantization for reduced model size
- **Validation**: Automatic output comparison between PyTorch and ONNX
- **Batch Processing**: Convert multiple models simultaneously
- **Metadata Preservation**: Model info and class mappings included

### ONNX Model Validation

Ensure ONNX models maintain accuracy:

```bash
# Validate ONNX model accuracy
python validate_onnx_model.py --onnx-path models/efficientnet_b0.onnx --test-data val/

# Compare PyTorch vs ONNX inference
python compare_pytorch_onnx.py --pytorch-model models/efficientnet_b0_20250806_035425/best_model.pth --onnx-model models/efficientnet_b0.onnx

# Benchmark ONNX performance
python benchmark_onnx.py --model-path models/efficientnet_b0.onnx
```

**Validation includes:**
- **Accuracy Preservation**: Ensure no accuracy loss during conversion
- **Output Consistency**: Verify identical outputs for same inputs
- **Performance Benchmarking**: Measure inference speed improvements
- **Memory Usage**: Compare memory footprint between formats

### ONNX Runtime Optimization

Optimize ONNX models for production:

```python
# Example ONNX optimization configuration
import onnxruntime as ort

# Configure optimization settings
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

# Enable specific optimizations
sess_options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
sess_options.add_session_config_entry('session.use_env_allocators', '1')
```

## Inference Pipeline

### High-Performance Inference Engine

The inference pipeline is optimized for production use:

```python
from src.inference.onnx_predictor import ONNXPredictor

# Initialize predictor with optimizations
predictor = ONNXPredictor(
    model_path="models/efficientnet_b0.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    enable_profiling=True
)

# Single image prediction
result = predictor.predict_single("path/to/image.jpg")

# Batch prediction
results = predictor.predict_batch(["image1.jpg", "image2.jpg", "image3.jpg"])
```

**Inference Features:**
- **Multi-Provider Support**: CUDA, CPU, and other execution providers
- **Batch Processing**: Efficient batch inference for multiple images
- **Memory Management**: Optimized memory usage for large batches
- **Preprocessing Pipeline**: Consistent image preprocessing
- **Error Handling**: Robust error handling for production use
- **Performance Monitoring**: Built-in latency and throughput tracking

### Inference Optimization Strategies

**For CPU Inference:**
```bash
# Optimize for CPU deployment
python optimize_for_cpu.py --model-path models/efficientnet_b0.onnx --threads 4

# Enable Intel MKL optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**For GPU Inference:**
```bash
# Optimize for GPU deployment
python optimize_for_gpu.py --model-path models/efficientnet_b0.onnx --batch-size 32

# Enable TensorRT optimization (if available)
python convert_to_tensorrt.py --onnx-path models/efficientnet_b0.onnx
```

## FastAPI Backend Architecture

### Production-Ready API Design

The FastAPI backend follows best practices for production deployment:

```python
# Example API structure
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.models import PredictionResponse, BatchPredictionResponse
from src.api.dependencies import get_predictor
from src.api.middleware import LoggingMiddleware, RateLimitMiddleware

app = FastAPI(
    title="Bean Lesion Classification API",
    description="Production-ready API for bean disease classification",
    version="1.0.0"
)

# Add middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, calls=100, period=60)
```

**API Features:**
- **Async Processing**: Non-blocking request handling
- **Rate Limiting**: Prevent API abuse
- **Request Validation**: Pydantic models for type safety
- **Error Handling**: Comprehensive error responses
- **Health Checks**: Monitoring endpoints for deployment
- **API Documentation**: Auto-generated OpenAPI docs
- **Metrics Collection**: Prometheus-compatible metrics

### API Security and Monitoring

```bash
# Enable API monitoring
python -m src.api.monitoring --enable-metrics --enable-tracing

# Configure rate limiting
export API_RATE_LIMIT=100
export API_RATE_WINDOW=60

# Enable request logging
export LOG_LEVEL=INFO
export LOG_FORMAT=json
```

**Security Features:**
- **Input Validation**: File type and size validation
- **Rate Limiting**: Per-IP request limiting
- **CORS Configuration**: Configurable cross-origin policies
- **Request Logging**: Comprehensive request/response logging
- **Health Monitoring**: Endpoint health and dependency checks

## React Frontend Architecture

### Modern React Application

The frontend is built with modern React practices:

```javascript
// Example component structure
import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from 'react-query';
import { PredictionResults } from './components/PredictionResults';
import { ImageUpload } from './components/ImageUpload';
import { LoadingSpinner } from './components/LoadingSpinner';

const BeanClassificationApp = () => {
  const [predictions, setPredictions] = useState([]);
  
  const predictMutation = useMutation(
    (files) => apiClient.predictBatch(files),
    {
      onSuccess: (data) => setPredictions(data.results),
      onError: (error) => console.error('Prediction failed:', error)
    }
  );

  return (
    <div className="app">
      <ImageUpload onUpload={predictMutation.mutate} />
      {predictMutation.isLoading && <LoadingSpinner />}
      {predictions.length > 0 && <PredictionResults results={predictions} />}
    </div>
  );
};
```

**Frontend Features:**
- **Drag & Drop Upload**: Intuitive file upload interface
- **Real-time Predictions**: Immediate feedback on uploads
- **Responsive Design**: Mobile-friendly interface
- **Error Handling**: User-friendly error messages
- **Progress Indicators**: Upload and processing progress
- **Results Visualization**: Interactive prediction results
- **Batch Processing**: Support for multiple image uploads

### Frontend Performance Optimization

```javascript
// Performance optimizations
import { lazy, Suspense } from 'react';
import { memo } from 'react';

// Code splitting
const PredictionResults = lazy(() => import('./components/PredictionResults'));

// Memoized components
const ImagePreview = memo(({ image, prediction }) => {
  return (
    <div className="image-preview">
      <img src={image.preview} alt="Upload preview" />
      <div className="prediction-overlay">
        <span className="class-name">{prediction.class_name}</span>
        <span className="confidence">{(prediction.confidence * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
});
```

## Deployment and DevOps

### Docker Containerization

Complete containerization for all components:

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Frontend Dockerfile
FROM node:16-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

### Docker Compose Orchestration

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/efficientnet_b0.onnx
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    environment:
      - REACT_APP_API_URL=http://localhost:8000

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend
```

### Production Deployment

```bash
# Production deployment commands
make docker-build-prod
make docker-deploy-prod

# Health check
make health-check

# Scale services
docker-compose up --scale backend=3

# Monitor logs
docker-compose logs -f backend
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      - name: Run linting
        run: |
          flake8 src/
          black --check src/
          isort --check-only src/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t bean-classification-backend .
          docker build -t bean-classification-frontend ./frontend
      - name: Deploy to production
        run: |
          # Add deployment commands here
```

## Monitoring and Observability

### Application Monitoring

```python
# Monitoring setup
from prometheus_client import Counter, Histogram, generate_latest
import logging
import time

# Metrics collection
prediction_counter = Counter('predictions_total', 'Total predictions made', ['model', 'class'])
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
error_counter = Counter('prediction_errors_total', 'Total prediction errors', ['error_type'])

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Monitoring

```bash
# Monitor system performance
python -m src.monitoring.performance_monitor

# Generate performance report
python -m src.monitoring.generate_report --period 24h

# Real-time monitoring dashboard
python -m src.monitoring.dashboard --port 9090
```

**Monitoring Features:**
- **Request Metrics**: Latency, throughput, error rates
- **Model Performance**: Accuracy drift, prediction confidence
- **System Metrics**: CPU, memory, disk usage
- **Custom Dashboards**: Grafana-compatible metrics
- **Alerting**: Configurable alerts for anomalies

### Logging and Debugging

```python
# Structured logging
import structlog

logger = structlog.get_logger()

# Log prediction events
logger.info(
    "prediction_completed",
    model="efficientnet_b0",
    class_predicted="healthy",
    confidence=0.95,
    processing_time=0.15,
    image_size="224x224"
)
```

## Testing Strategy

### Comprehensive Test Suite

```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests
make test-integration   # Integration tests
make test-e2e          # End-to-end tests
make test-performance  # Performance tests

# Generate coverage report
make test-coverage
```

**Test Coverage:**
- **Data Pipeline Tests**: Dataset loading, validation, transforms
- **Model Tests**: Training, evaluation, ONNX conversion
- **API Tests**: Endpoint functionality, error handling
- **Frontend Tests**: Component rendering, user interactions
- **Integration Tests**: Full pipeline end-to-end
- **Performance Tests**: Load testing, stress testing

### Test Data Management

```bash
# Generate test data
python scripts/generate_test_data.py --samples 100

# Validate test data
python scripts/validate_test_data.py

# Clean test artifacts
make clean-test-data
```

## Performance Optimization

### Model Optimization Techniques

1. **Quantization**: Reduce model size with minimal accuracy loss
2. **Pruning**: Remove unnecessary model parameters
3. **Knowledge Distillation**: Train smaller models from larger ones
4. **Batch Processing**: Optimize for batch inference
5. **Caching**: Cache frequent predictions

### Infrastructure Optimization

```bash
# Optimize for production
python optimize_production.py --target cpu --batch-size 32
python optimize_production.py --target gpu --batch-size 64

# Enable caching
export ENABLE_PREDICTION_CACHE=true
export CACHE_TTL=3600

# Configure worker processes
export WORKERS=4
export WORKER_CONNECTIONS=1000
```