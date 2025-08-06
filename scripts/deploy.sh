#!/bin/bash

# Bean Classification Deployment Script

set -e

# Configuration
ENVIRONMENT=${1:-production}
REGISTRY=${DOCKER_REGISTRY:-}
TAG=${IMAGE_TAG:-latest}
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"

echo "🚀 Starting deployment for environment: $ENVIRONMENT"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command_exists docker; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env.${ENVIRONMENT}" ]; then
    echo "❌ Environment file .env.${ENVIRONMENT} not found"
    exit 1
fi

# Copy environment file
echo "📄 Setting up environment configuration..."
cp ".env.${ENVIRONMENT}" .env

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Models directory not found. Creating empty directory..."
    mkdir -p models
fi

# Build images
echo "🔨 Building Docker images..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE build --no-cache
else
    docker-compose build --no-cache
fi

# Tag images if registry is specified
if [ -n "$REGISTRY" ]; then
    echo "🏷️  Tagging images for registry..."
    docker tag bean-classification-backend:latest $REGISTRY/bean-classification-backend:$TAG
    docker tag bean-classification-frontend:latest $REGISTRY/bean-classification-frontend:$TAG
    
    echo "📤 Pushing images to registry..."
    docker push $REGISTRY/bean-classification-backend:$TAG
    docker push $REGISTRY/bean-classification-frontend:$TAG
fi

# Stop existing services
echo "🛑 Stopping existing services..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE down
else
    docker-compose down
fi

# Start services
echo "🚀 Starting services..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f $COMPOSE_FILE -f $PROD_COMPOSE_FILE up -d
else
    docker-compose up -d
fi

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Performing health checks..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    echo "Health check attempt $attempt/$max_attempts..."
    
    # Check backend health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo "✅ Backend is healthy"
        backend_healthy=true
        break
    else
        echo "❌ Backend health check failed"
        backend_healthy=false
    fi
    
    attempt=$((attempt + 1))
    sleep 10
done

# Check frontend health
if curl -f http://localhost/health >/dev/null 2>&1; then
    echo "✅ Frontend is healthy"
    frontend_healthy=true
else
    echo "❌ Frontend health check failed"
    frontend_healthy=false
fi

# Final status
if [ "$backend_healthy" = true ] && [ "$frontend_healthy" = true ]; then
    echo "🎉 Deployment successful!"
    echo "📊 Service status:"
    docker-compose ps
    echo ""
    echo "🌐 Application URLs:"
    echo "   Frontend: http://localhost"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    
    if [ "$ENVIRONMENT" = "production" ]; then
        echo "   Prometheus: http://localhost:9090"
        echo "   Grafana: http://localhost:3000"
    fi
    
    exit 0
else
    echo "❌ Deployment failed - services are not healthy"
    echo "📋 Service logs:"
    docker-compose logs --tail=50
    exit 1
fi