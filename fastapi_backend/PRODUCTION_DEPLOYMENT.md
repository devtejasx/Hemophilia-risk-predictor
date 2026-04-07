# FastAPI Backend - Production Deployment & Operations

## 📋 Pre-Deployment Checklist

### **Code Quality**
- [ ] All tests passing locally
- [ ] Code coverage ≥ 80%
- [ ] No linting errors (`pylint`, `flake8`)
- [ ] Type checking passes (`mypy`)
- [ ] All docstrings complete
- [ ] No hardcoded secrets in code
- [ ] No debug prints or debug mode enabled

### **Configuration**
- [ ] `.env` file created and Git-ignored
- [ ] `.env.example` has all required variables
- [ ] Environment-specific configs created
- [ ] Database migrations tested
- [ ] API keys secured (never in Git)

### **Infrastructure**
- [ ] Database backup strategy defined
- [ ] Monitoring and logging configured
- [ ] Error alerting configured
- [ ] Rate limiting tested
- [ ] CORS settings verified
- [ ] SSL/TLS certificates ready

### **Documentation**
- [ ] API documentation complete
- [ ] README updated
- [ ] Architecture documented
- [ ] Deployment guide written
- [ ] Runbook created for common issues

---

## 🚀 Deployment Strategies

### **Strategy 1: Docker + Heroku (Easiest)**

```bash
# 1. Create Heroku app
heroku create hemophilia-api-prod

# 2. Add PostgreSQL addon
heroku addons:create heroku-postgresql:standard-0

# 3. Set environment variables
heroku config:set OPENAI_API_KEY=sk-xxxxx
heroku config:set ENVIRONMENT=production

# 4. Deploy
git push heroku main

# 5. Verify
heroku open
heroku logs --tail
```

### **Strategy 2: Docker + AWS ECS (Scalable)**

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name hemophilia-api

# 2. Build and push image
docker build -t hemophilia-api:latest .
docker tag hemophilia-api:latest <AWS_ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/hemophilia-api:latest
docker push <AWS_ACCOUNT>.dkr.ecr.us-east-1.amazonaws.com/hemophilia-api:latest

# 3. Create ECS task definition, service, and load balancer (via AWS Console)

# 4. Configure RDS database for PostgreSQL

# 5. Deploy via CLI or Console
```

### **Strategy 3: Docker + DigitalOcean App Platform (Balanced)**

```bash
# 1. Creates app.yaml
cat > app.yaml << EOF
name: hemophilia-api
services:
  - name: api
    image:
      registry: docker
      registry_name: my-registry
      repository: hemophilia-api
      tag: latest
    http_port: 8000
    health_check:
      http_path: /health
    envs:
      - key: ENVIRONMENT
        value: production
EOF

# 2. Deploy
doctl apps create --spec app.yaml

# 3. Add managed database
```

### **Strategy 4: Kubernetes (Enterprise)**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hemophilia-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hemophilia-api
  template:
    metadata:
      labels:
        app: hemophilia-api
    spec:
      containers:
      - name: api
        image: hemophilia-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secret
              key: openai-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: hemophilia-api
spec:
  selector:
    app: hemophilia-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📊 Monitoring & Logging

### **Application Metrics (Prometheus)**

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

active_patients = Gauge(
    'active_patients_total',
    'Total active patients'
)

# Middleware
from fastapi import Request
from contextlib import asynccontextmanager

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

### **Structured Logging**

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for production"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Configure logging
handler = logging.FileHandler('app.log')
handler.setFormatter(JSONFormatter())

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### **Error Tracking (Sentry)**

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
    environment=os.getenv("ENVIRONMENT")
)

# Automatic error tracking
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    sentry_sdk.capture_exception(exc)
    return {"error": "Internal server error"}
```

### **Health Dashboard**

```python
# health.py
from datetime import datetime
import psutil
import os

def get_system_health():
    """Get system health metrics"""
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'uptime_seconds': int(time.time() - start_time),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'database_connected': check_database(),
        'models_loaded': prediction_service.models_loaded(),
        'cache_size_mb': get_cache_size(),
    }

@app.get("/health")
async def health():
    """Comprehensive health check"""
    return get_system_health()
```

---

## 🔄 Continuous Integration/Deployment

### **GitHub Actions Workflow**

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint
      run: |
        pylint fastapi_backend/
        flake8 fastapi_backend/
    
    - name: Type check
      run: mypy fastapi_backend/
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        OPENAI_API_KEY: sk-test-key
      run: pytest --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/hemophilia-api:latest
    
    - name: Deploy to Heroku
      run: |
        git remote add heroku https://git.heroku.com/hemophilia-api-prod.git
        git push heroku main
      env:
        HEROKU_API_KEY: ${{ secrets.HEROKU_API_KEY }}
```

---

## 📈 Scaling & Performance

### **Horizontal Scaling**

```yaml
# docker-compose.yml with scaling
version: '3.8'

services:
  api:
    build: .
    deploy:
      replicas: 3  # Run 3 instances
      update_config:
        parallelism: 1
        delay: 10s
    ports:
      - "8000-8002:8000"
    depends_on:
      - db
      - cache

  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: password

  cache:
    image: redis:7
    
  lb:  # Load balancer
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### **Caching Strategy**

```python
# cache.py
from functools import wraps
import hashlib
import json
import redis

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_result(ttl: int = 3600):
    """
    Cache decorator with TTL
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Compute and cache
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
            return result
        
        return wrapper
    return decorator

# Usage
@cache_result(ttl=3600)
def get_high_risk_patients():
    """Cache for 1 hour"""
    return analytics_service.get_high_risk_patients()
```

### **Database Connection Pooling**

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600,
    echo=False
)

def get_db():
    """Get database session"""
    with engine.connect() as conn:
        yield conn
```

---

## 🚨 Incident Response

### **Runbook: API Not Responding**

```markdown
## Symptom
API endpoints returning 503 or timing out

## Investigation
1. Check service status
   ```bash
   docker-compose ps
   systemctl status hemophilia-api
   ```

2. Check logs
   ```bash
   docker logs hemophilia-api
   tail -f /var/log/app.log
   ```

3. Check resources
   ```bash
   docker stats
   df -h
   free -h
   ```

4. Check database
   ```bash
   psql -h localhost -U postgres -d hemophilia
   SELECT count(*) FROM patients;
   ```

## Resolution
- Restart service: `docker-compose restart api`
- Scale up: `docker-compose up -d --scale api=5`
- Check database: Verify connections not exhausted
- Clear cache: `redis-cli FLUSHALL`
```

### **Runbook: High Memory Usage**

```markdown
## Symptom
Memory usage > 80% or API running slowly

## Investigation
1. Identify memory leak
   ```bash
   docker exec hemophilia-api python -m memory_profiler app.py
   ```

2. Check for long-running requests
   ```bash
   tail -f app.log | grep duration
   ```

3. Check cache size
   ```bash
   redis-cli INFO memory
   ```

## Resolution
- Restart service to clear caches
- Reduce model cache size
- Implement request timeout limits
- Scale horizontally
```

---

## 📋 Backup & Recovery

### **Database Backup Strategy**

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/hemophilia"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/hemophilia_$DATE.sql"

# Create directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump postgresql://postgres:password@localhost:5432/hemophilia > $BACKUP_FILE

# Compress
gzip $BACKUP_FILE

# Upload to S3
aws s3 cp $BACKUP_FILE.gz s3://hemophilia-backups/

# Keep only last 30 days
find $BACKUP_DIR -mtime +30 -delete

echo "✅ Backup completed: $BACKUP_FILE.gz"
```

### **Cron Job for Daily Backups**

```bash
# Add to crontab -e
0 2 * * * /path/to/backup.sh  # Run daily at 2 AM

# Verify
crontab -l
```

### **Restore from Backup**

```bash
# Restore database
gzip -dc backup_file.sql.gz | psql postgresql://postgres:password@localhost:5432/hemophilia

# Verify restore
psql postgresql://postgres:password@localhost:5432/hemophilia -c "SELECT COUNT(*) FROM patients;"
```

---

## 🔐 Security Hardening

### **Production Security Checklist**

- [ ] All secrets in environment variables
- [ ] Database credentials never exposed
- [ ] API keys rotated regularly
- [ ] HTTPS/TLS enabled
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS protection headers set
- [ ] CSRF tokens implemented
- [ ] Regular security audits
- [ ] Dependencies updated regularly
- [ ] WAF (Web Application Firewall) enabled

### **Security Headers**

```python
# main.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# HTTPS redirect
app.add_middleware(HTTPSRedirectMiddleware)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.example.com", "*.example.com"]
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

---

## 📞 Support & Escalation

### **Support Contacts**

```
🏥 Production Issues:
  - Engineering: @dev-team
  - On-Call: Check PagerDuty
  - Slack: #production-alerts

📞 Contact Information:
  - API Lead: dev@example.com
  - DevOps: devops@example.com
  - Database: dba@example.com
```

### **Escalation Path**

```
1. Detection (monitoring/alerts)
   ↓
2. Initial Response (on-call engineer)
   ↓
3. Investigation & Mitigation (team)
   ↓
4. Resolution & Root Cause Analysis
   ↓
5. Post-Incident Review & Prevention
```

---

## ✅ Post-Deployment Validation

```bash
#!/bin/bash
# validate-deployment.sh

echo "🔍 Validating deployment..."

# 1. Health check
echo "1️⃣ Health check..."
curl -f http://localhost:8000/health || exit 1

# 2. Database connectivity
echo "2️⃣ Database check..."
curl -f http://localhost:8000/api/v1/patients/ || exit 1

# 3. Model loading
echo "3️⃣ Model loading..."
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "M",
    "severity": "severe",
    "mutation": "intron22",
    "dose_intensity": 50.0,
    "exposure_days": 365,
    "fviii_inhibitor": false
  }' || exit 1

# 4. Performance baseline
echo "4️⃣ Performance check..."
ab -n 100 -c 10 http://localhost:8000/health

echo "✅ Deployment validated successfully!"
```

---

## 📊 Summary

✅ Multiple deployment strategies  
✅ Comprehensive monitoring  
✅ CI/CD pipeline setup  
✅ Scaling strategies  
✅ Incident response guides  
✅ Backup & recovery procedures  
✅ Security hardening  
✅ Post-deployment validation  

**Production-ready operations manual!**
