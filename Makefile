# Bean Classification Docker Management

.PHONY: help build up down logs clean test

# Default environment
ENV ?= development

help: ## Show this help message
	@echo "Bean Classification Docker Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build --no-cache

build-backend: ## Build only backend image
	@echo "Building backend image..."
	docker-compose build --no-cache backend

build-frontend: ## Build only frontend image
	@echo "Building frontend image..."
	docker-compose build --no-cache frontend

up: ## Start all services in development mode
	@echo "Starting services in $(ENV) mode..."
	@if [ "$(ENV)" = "production" ]; then \
		docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d; \
	else \
		docker-compose up -d; \
	fi

up-dev: ## Start services in development mode with logs
	@echo "Starting services in development mode..."
	docker-compose up

up-prod: ## Start services in production mode
	@echo "Starting services in production mode..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

up-monitoring: ## Start services with monitoring stack
	@echo "Starting services with monitoring..."
	docker-compose --profile monitoring up -d

down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose down

down-volumes: ## Stop all services and remove volumes
	@echo "Stopping services and removing volumes..."
	docker-compose down -v

logs: ## Show logs for all services
	docker-compose logs -f

logs-backend: ## Show backend logs
	docker-compose logs -f backend

logs-frontend: ## Show frontend logs
	docker-compose logs -f frontend

status: ## Show service status
	docker-compose ps

restart: ## Restart all services
	@echo "Restarting services..."
	docker-compose restart

restart-backend: ## Restart backend service
	docker-compose restart backend

restart-frontend: ## Restart frontend service
	docker-compose restart frontend

shell-backend: ## Open shell in backend container
	docker-compose exec backend /bin/bash

shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

test: ## Run tests in containers
	@echo "Running tests..."
	docker-compose exec backend python -m pytest tests/ -v
	docker-compose exec frontend npm test

clean: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-all: ## Clean up all Docker resources including images
	@echo "Cleaning up all Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f

health: ## Check service health
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health && echo "Backend: OK" || echo "Backend: FAIL"
	@curl -f http://localhost/health && echo "Frontend: OK" || echo "Frontend: FAIL"

deploy: ## Deploy to production
	@echo "Deploying to production..."
	@make build ENV=production
	@make up-prod

backup-models: ## Backup model files
	@echo "Backing up models..."
	docker run --rm -v $(PWD)/models:/source -v $(PWD)/backups:/backup alpine tar czf /backup/models-$(shell date +%Y%m%d-%H%M%S).tar.gz -C /source .

restore-models: ## Restore model files (specify BACKUP_FILE)
	@echo "Restoring models from $(BACKUP_FILE)..."
	docker run --rm -v $(PWD)/models:/target -v $(PWD)/backups:/backup alpine tar xzf /backup/$(BACKUP_FILE) -C /target