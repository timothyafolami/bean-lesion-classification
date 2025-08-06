# Bean Classification - Monitoring & Observability Guide

This guide covers the comprehensive monitoring and observability setup for the Bean Lesion Classification system using Prometheus, Grafana, and structured logging.

## 🔍 Overview

The monitoring system provides:
- **Metrics Collection**: Prometheus metrics for API, inference, and system performance
- **Structured Logging**: JSON and text-based logging with multiple levels
- **Visualization**: Grafana dashboards for real-time monitoring
- **Alerting**: Prometheus alert rules for proactive issue detection
- **Health Checks**: Detailed health endpoints for system status

## 📊 Metrics Collected

### API Metrics
- `api_requests_total` - Total API requests by method, endpoint, status code
- `api_request_duration_seconds` - Request duration histogram
- `api_request_size_bytes` - Request size histogram
- `api_response_size_bytes` - Response size histogram

### Inference Metrics
- `inference_requests_total` - Total inference requests by model format and type
- `inference_duration_seconds` - Inference duration histogram
- `inference_preprocessing_duration_seconds` - Preprocessing time histogram
- `inference_confidence_score` - Model confidence score distribution

### Image Processing Metrics
- `images_processed_total` - Total images processed (success/failed)
- `image_size_bytes` - Uploaded image size distribution
- `batch_size_images` - Batch size distribution

### System Metrics
- `active_inference_sessions` - Number of active ONNX sessions
- `memory_usage_bytes` - Memory usage by component
- `model_load_time_seconds` - Model loading time
- `errors_total` - Error count by type and component

### Application Status
- `app_status` - Current application status (starting/ready/degraded/error)
- `model_info` - Model information and metadata

## 🚀 Quick Start

### 1. Enable Monitoring

```bash
# Start with monitoring stack
docker-compose --profile monitoring up -d

# Or using Make
make up-monitoring
```

### 2. Access Monitoring Tools

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **API Metrics**: http://localhost:8000/monitoring/metrics
- **Health Check**: http://localhost:8000/monitoring/health/detailed

### 3. View Dashboards

1. Open Grafana at http://localhost:3000
2. Login with admin/admin123
3. Navigate to "Bean Classification API Monitoring" dashboard

## 📈 Monitoring Endpoints

### Metrics Endpoint
```bash
# Prometheus metrics
curl http://localhost:8000/monitoring/metrics
```

### Health Endpoints
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health with system metrics
curl http://localhost:8000/monitoring/health/detailed

# Application statistics
curl http://localhost:8000/monitoring/stats

# Performance metrics
curl http://localhost:8000/monitoring/performance
```

### Log Management
```bash
# Recent logs (if file logging enabled)
curl http://localhost:8000/monitoring/logs/recent?lines=50

# Monitoring configuration
curl http://localhost:8000/monitoring/config
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `METRICS_ENABLED` | Enable metrics collection | `true` |
| `PROMETHEUS_ENABLED` | Enable Prometheus integration | `false` |
| `LOG_LEVEL` | Logging level | `info` |
| `LOG_FORMAT` | Log format (json/text) | `text` |
| `LOG_FILE` | Log file path | `./logs/app.log` |

### Logging Configuration

```python
# Setup structured logging
from src.monitoring.logging_config import setup_logging

setup_logging(
    log_level="INFO",
    log_format="json",  # or "text"
    log_file="./logs/app.log"
)
```

### Custom Metrics

```python
from src.monitoring.metrics import metrics_collector

# Record custom metrics
metrics_collector.record_inference(
    model_format="onnx",
    prediction_type="single",
    duration=0.5,
    confidence=0.95,
    predicted_class="healthy"
)

# Update system metrics
metrics_collector.update_memory_usage("api", 1024*1024*100)  # 100MB
metrics_collector.update_active_sessions(4)
```

## 🚨 Alerting

### Alert Rules

The system includes predefined alert rules for:

- **High Error Rate**: >10% error rate for 2 minutes
- **High Response Time**: >5s 95th percentile for 3 minutes
- **High Memory Usage**: >2GB for 5 minutes
- **Inference Failures**: >5% failure rate for 2 minutes
- **No Active Sessions**: Model not loaded for 1 minute
- **Slow Batch Processing**: >30s 90th percentile for 5 minutes

### Custom Alerts

Add custom alert rules to `monitoring/alert_rules.yml`:

```yaml
- alert: CustomAlert
  expr: your_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert description"
    description: "Detailed alert message"
```

## 📊 Grafana Dashboards

### Pre-built Dashboard

The system includes a comprehensive dashboard with:

1. **API Overview**: Request rate, response time, error rate
2. **Inference Performance**: Duration, confidence scores, throughput
3. **System Resources**: Memory usage, active sessions
4. **Image Processing**: Processing rates, batch sizes
5. **Error Tracking**: Error rates by type and component

### Custom Dashboards

Create custom dashboards using available metrics:

```json
{
  "targets": [
    {
      "expr": "rate(api_requests_total[5m])",
      "legendFormat": "{{method}} {{endpoint}}"
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Metrics Not Appearing**
   ```bash
   # Check if metrics endpoint is accessible
   curl http://localhost:8000/monitoring/metrics
   
   # Verify Prometheus configuration
   docker-compose logs prometheus
   ```

2. **High Memory Usage Alerts**
   ```bash
   # Check memory usage
   curl http://localhost:8000/monitoring/performance
   
   # Monitor memory over time
   docker stats bean-classification-backend
   ```

3. **Slow Inference Alerts**
   ```bash
   # Check inference metrics
   curl http://localhost:8000/monitoring/stats
   
   # Review model performance
   curl http://localhost:8000/model/info
   ```

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test alert rules
curl http://localhost:9090/api/v1/rules

# View Grafana logs
docker-compose logs grafana

# Check application logs
docker-compose logs backend | tail -100
```

## 📝 Log Analysis

### Structured Logging

Logs are available in both JSON and text formats:

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "logger": "src.api.routes",
  "message": "Prediction completed successfully",
  "method": "POST",
  "endpoint": "/predict/single",
  "duration_seconds": 0.5,
  "confidence": 0.95,
  "predicted_class": "healthy"
}
```

### Log Aggregation

For production, consider integrating with:
- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Fluentd** for log forwarding
- **Loki** for Grafana-native log aggregation

## 🔒 Security Considerations

### Metrics Security

1. **Access Control**: Restrict access to metrics endpoints
2. **Sensitive Data**: Avoid logging sensitive information
3. **Network Security**: Use HTTPS in production
4. **Authentication**: Secure Grafana with proper authentication

### Example Security Configuration

```yaml
# docker-compose.yml
services:
  grafana:
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_AUTH_ANONYMOUS_ENABLED=false
```

## 📈 Performance Optimization

### Metrics Performance

1. **Sampling**: Use histogram buckets appropriate for your use case
2. **Cardinality**: Avoid high-cardinality labels
3. **Retention**: Configure appropriate data retention policies
4. **Aggregation**: Use recording rules for expensive queries

### Example Recording Rules

```yaml
# prometheus-rules.yml
groups:
  - name: bean_classification_recording
    rules:
      - record: api:request_rate_5m
        expr: rate(api_requests_total[5m])
      
      - record: inference:duration_p95_5m
        expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m]))
```

## 🔄 Maintenance

### Regular Tasks

1. **Log Rotation**: Ensure log files don't fill disk space
2. **Metrics Cleanup**: Remove unused metrics and dashboards
3. **Alert Tuning**: Adjust alert thresholds based on usage patterns
4. **Dashboard Updates**: Keep dashboards relevant and useful

### Backup and Recovery

```bash
# Backup Grafana dashboards
docker exec grafana grafana-cli admin export-dashboard > backup.json

# Backup Prometheus data
docker run --rm -v prometheus_data:/source -v $(pwd)/backups:/backup alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz -C /source .
```

## 🎯 Best Practices

1. **Start Simple**: Begin with basic metrics and expand gradually
2. **Focus on SLIs**: Monitor Service Level Indicators that matter
3. **Actionable Alerts**: Only alert on conditions that require action
4. **Documentation**: Keep runbooks for common alert scenarios
5. **Regular Review**: Periodically review and update monitoring setup

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [FastAPI Monitoring Best Practices](https://fastapi.tiangolo.com/advanced/monitoring/)