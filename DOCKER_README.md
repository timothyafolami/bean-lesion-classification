# Bean Classification - Docker Deployment Guide

This guide covers containerization and deployment of the Bean Lesion Classification system using Docker and Docker Compose.

## 🏗️ Architecture

The application consists of:
- **Backend**: FastAPI application with ONNX inference engine
- **Frontend**: React application served by Nginx
- **Optional Services**: Redis (caching), Prometheus (metrics), Grafana (monitoring)

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Make (optional, for convenience commands)

## 🚀 Quick Start

### Development Mode

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd bean-lesion-classification
   cp .env.example .env
   ```

2. **Build and start services**:
   ```bash
   # Using Make (recommended)
   make build
   make up-dev

   # Or using Docker Compose directly
   docker-compose build
   docker-compose up
   ```

3. **Access the application**:
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Production Mode

1. **Setup production environment**:
   ```bash
   cp .env.production .env
   # Edit .env with your production settings
   ```

2. **Deploy**:
   ```bash
   # Using deployment script (recommended)
   ./scripts/deploy.sh production

   # Or using Make
   make deploy

   # Or using Docker Compose directly
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

## 📁 File Structure

```
├── Dockerfile.backend          # Backend container definition
├── Dockerfile.frontend         # Frontend container definition
├── docker-compose.yml          # Main compose configuration
├── docker-compose.prod.yml     # Production overrides
├── nginx.conf                  # Nginx configuration
├── .dockerignore              # Docker build context exclusions
├── .env.example               # Environment template
├── .env.development           # Development settings
├── .env.production            # Production settings
├── Makefile                   # Convenience commands
└── scripts/
    ├── deploy.sh              # Deployment script
    └── health_check.py        # Health check utility
```

## 🔧 Configuration

### Environment Variables

Key configuration options:

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `API_WORKERS` | Number of API workers | `1` (dev), `4` (prod) |
| `UPLOAD_MAX_SIZE` | Max upload size in bytes | `10485760` (10MB) |
| `BATCH_MAX_SIZE` | Max batch size | `10` |
| `LOG_LEVEL` | Logging level | `info` |
| `CORS_ORIGINS` | Allowed CORS origins | `http://localhost:3000` |
| `CACHE_ENABLED` | Enable Redis caching | `false` |
| `PROMETHEUS_ENABLED` | Enable metrics | `false` |

### Service Profiles

Use Docker Compose profiles to enable optional services:

```bash
# Enable caching
docker-compose --profile cache up -d

# Enable monitoring
docker-compose --profile monitoring up -d

# Enable all optional services
docker-compose --profile cache --profile monitoring up -d
```

## 🛠️ Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available commands |
| `make build` | Build all images |
| `make up` | Start services |
| `make up-dev` | Start with logs (development) |
| `make up-prod` | Start in production mode |
| `make up-monitoring` | Start with monitoring stack |
| `make down` | Stop services |
| `make logs` | Show service logs |
| `make status` | Show service status |
| `make health` | Check service health |
| `make clean` | Clean up resources |
| `make deploy` | Deploy to production |

## 🏥 Health Checks

### Automated Health Checks

Both services include health checks:
- **Backend**: `GET /health` endpoint
- **Frontend**: Nginx health endpoint

### Manual Health Check

```bash
# Check all services
make health

# Check individual services
curl http://localhost:8000/health  # Backend
curl http://localhost/health       # Frontend

# Detailed health check
python scripts/health_check.py
```

## 📊 Monitoring (Optional)

Enable monitoring stack with Prometheus and Grafana:

```bash
# Start with monitoring
make up-monitoring

# Access monitoring tools
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin123)
```

## 🔒 Security Considerations

### Production Security

1. **Environment Variables**: Never commit `.env` files with secrets
2. **User Permissions**: Containers run as non-root users
3. **Network Security**: Services communicate via internal Docker network
4. **Resource Limits**: Production compose includes resource constraints
5. **Security Headers**: Nginx includes security headers

### Recommended Production Setup

```bash
# Use secrets management
docker secret create api_key /path/to/api_key.txt

# Enable firewall
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # Block direct backend access

# Use reverse proxy with SSL
# Configure nginx or traefik for HTTPS termination
```

## 🚨 Troubleshooting

### Common Issues

1. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # Use different host port
   ```

2. **Memory Issues**:
   ```bash
   # Check container resources
   docker stats
   
   # Increase memory limits in docker-compose.prod.yml
   ```

3. **Model Loading Errors**:
   ```bash
   # Check model files
   ls -la models/
   
   # Check backend logs
   make logs-backend
   ```

4. **Build Failures**:
   ```bash
   # Clean build cache
   docker builder prune
   
   # Rebuild without cache
   make build
   ```

### Debug Commands

```bash
# Enter container shell
make shell-backend
make shell-frontend

# View detailed logs
docker-compose logs -f --tail=100 backend

# Check container processes
docker-compose exec backend ps aux

# Monitor resource usage
docker stats bean-classification-backend
```

## 📈 Performance Tuning

### Backend Optimization

1. **Worker Processes**: Adjust `API_WORKERS` based on CPU cores
2. **Memory**: Monitor memory usage and adjust container limits
3. **Caching**: Enable Redis for inference result caching
4. **ONNX Optimization**: Use optimized ONNX models

### Frontend Optimization

1. **Nginx Caching**: Static assets cached for 1 year
2. **Gzip Compression**: Enabled for text-based assets
3. **Bundle Optimization**: Production builds are minified

### Database/Storage

1. **Volume Mounts**: Use named volumes for persistence
2. **Backup Strategy**: Regular model and data backups
3. **Log Rotation**: Configure log rotation to prevent disk fill

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          ./scripts/deploy.sh production
        env:
          DOCKER_REGISTRY: ${{ secrets.DOCKER_REGISTRY }}
          IMAGE_TAG: ${{ github.sha }}
```

### Deployment Strategies

1. **Blue-Green Deployment**: Use multiple compose files
2. **Rolling Updates**: Update services one at a time
3. **Health Checks**: Ensure services are healthy before switching traffic

## 📝 Maintenance

### Regular Tasks

1. **Update Images**: Regularly update base images
2. **Clean Resources**: Run `make clean` periodically
3. **Monitor Logs**: Check for errors and warnings
4. **Backup Models**: Regular model file backups
5. **Security Updates**: Keep Docker and dependencies updated

### Backup and Recovery

```bash
# Backup models
make backup-models

# Restore models
make restore-models BACKUP_FILE=models-20231201-120000.tar.gz

# Backup volumes
docker run --rm -v bean-classification_logs:/source -v $(pwd)/backups:/backup alpine tar czf /backup/logs-backup.tar.gz -C /source .
```