# FastAPI Backend Docker Deployment Guide

## 🐳 Docker Setup

### **Dockerfile** (Production-ready)

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **.dockerignore**

```
__pycache__
*.pyc
*.pyo
.git
.gitignore
.env
.DS_Store
*.db
dist/
build/
*.egg-info/
.pytest_cache/
htmlcov/
.coverage
*.log
venv/
env/
.vscode/
.idea/
```

---

## 🎯 Docker Compose Setup

### **docker-compose.yml** (With PostgreSQL)

```yaml
version: '3.8'

services:
  # FastAPI Backend
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: hemophilia-api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/hemophilia
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    networks:
      - hemophilia-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: hemophilia-db
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=hemophilia
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - hemophilia-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Cache (Optional)
  redis:
    image: redis:7-alpine
    container_name: hemophilia-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - hemophilia-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Adminer (Database UI)
  adminer:
    image: adminer:latest
    container_name: hemophilia-adminer
    ports:
      - "8080:8080"
    depends_on:
      - db
    networks:
      - hemophilia-network
    restart: unless-stopped

networks:
  hemophilia-network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
```

### **init.sql** (Database initialization)

```sql
-- Create tables
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10),
    severity VARCHAR(50),
    mutation VARCHAR(100),
    dose_intensity FLOAT,
    exposure_days INT,
    fviii_inhibitor BOOLEAN DEFAULT FALSE,
    risk_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id) ON DELETE CASCADE,
    risk_score FLOAT NOT NULL,
    severity_category VARCHAR(50),
    explanation TEXT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_patients_severity ON patients(severity);
CREATE INDEX idx_patients_mutation ON patients(mutation);
CREATE INDEX idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX idx_conversations_patient_id ON conversations(patient_id);

-- Create views for analytics
CREATE VIEW patient_statistics AS
SELECT
    COUNT(*) as total_patients,
    AVG(age) as avg_age,
    MIN(risk_score) as min_risk,
    MAX(risk_score) as max_risk,
    AVG(risk_score) as avg_risk
FROM patients;
```

---

## 🚀 Building and Running

### **Build Docker Image**

```bash
# Build image
docker build -t hemophilia-api:latest .

# Build with specific tag
docker build -t hemophilia-api:v1.0.0 .

# Build with build args
docker build --build-arg ENVIRONMENT=production -t hemophilia-api:latest .
```

### **Run Single Container**

```bash
# Build and run
docker build -t hemophilia-api . && docker run -p 8000:8000 hemophilia-api

# Run with environment file
docker run -p 8000:8000 --env-file .env hemophilia-api

# Run with volume mounts
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  hemophilia-api

# Run in background
docker run -d -p 8000:8000 --name api hemophilia-api

# View logs
docker logs api
docker logs -f api  # Follow logs
```

### **Docker Compose Operations**

```bash
# Start all services
docker-compose up -d

# Start with rebuild
docker-compose up -d --build

# Start specific service
docker-compose up -d db

# View logs
docker-compose logs -f api

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart specific service
docker-compose restart api

# Rebuild service
docker-compose up -d --build api

# Execute command in container
docker-compose exec api bash
docker-compose exec api python -m pytest

# View service status
docker-compose ps
```

---

## 🔍 Verification & Testing

### **Health Check**

```bash
# Inside container
curl http://localhost:8000/health

# From host
curl http://localhost:8000/health

# Verbose output
curl -v http://localhost:8000/health
```

### **Test Database Connection**

```bash
# From container
docker-compose exec api python -c "
from database import DatabaseConnection
db = DatabaseConnection()
print('Database connected successfully!')
"

# Direct connection
docker-compose exec db psql -U postgres -d hemophilia -c "SELECT COUNT(*) FROM patients;"
```

### **View Container Info**

```bash
# Container logs
docker logs hemophilia-api

# Container stats
docker stats hemophilia-api

# Container inspect
docker inspect hemophilia-api

# Network details
docker network inspect hemophilia-network

# Volume details
docker volume inspect hemophilia_postgres_data
```

---

## 🛡️ Security Best Practices

### **Environment Variables**

```bash
# .env (Git-ignored)
DATABASE_URL=postgresql://postgres:secure_password@db:5432/hemophilia
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
ENVIRONMENT=production
LOG_LEVEL=WARNING
SECRET_KEY=your-secret-key-here
```

### **Improved Dockerfile (Security)**

```dockerfile
# Use specific version
FROM python:3.11.4-slim

WORKDIR /app

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Don't run as root
RUN useradd -m -u 1000 appuser
USER appuser

# Copy requirements
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy app
COPY --chown=appuser:appuser . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Network Security**

```yaml
# docker-compose.yml - Better security
services:
  api:
    networks:
      - hemophilia-network
    # No port exposure until needed
    ports:
      - "127.0.0.1:8000:8000"  # Localhost only
    secrets:
      - openai_key

  db:
    networks:
      - hemophilia-network
    ports: []  # No external access
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password

secrets:
  openai_key:
    file: ./secrets/openai_key.txt
  db_password:
    file: ./secrets/db_password.txt

networks:
  hemophilia-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: br-hemophilia
```

---

## 📦 Deployment to Cloud

### **Heroku Deployment**

```bash
# Install Heroku CLI
# Login
heroku login

# Create app
heroku create hemophilia-api

# Set environment variables
heroku config:set OPENAI_API_KEY=sk-xxxxx
heroku config:set DATABASE_URL=postgresql://...

# Push code
git push heroku main

# View logs
heroku logs --tail

# Scale dynos
heroku ps:scale web=1
```

### **Docker Hub Publish**

```bash
# Login
docker login

# Tag image
docker tag hemophilia-api:latest yourusername/hemophilia-api:latest

# Push
docker push yourusername/hemophilia-api:latest

# Pull from registry
docker pull yourusername/hemophilia-api:latest
```

### **AWS ECS Deployment**

```bash
# Build for ECS
docker build -t 123456789.dkr.ecr.region.amazonaws.com/hemophilia-api:latest .

# Push to ECR
aws ecr get-login-password --region region | docker login --username AWS --password-stdin 123456789.dkr.ecr.region.amazonaws.com
docker push 123456789.dkr.ecr.region.amazonaws.com/hemophilia-api:latest

# Create ECS task definition, service, and cluster through AWS Console or CLI
```

---

## 🔧 Troubleshooting

### **Port Already in Use**

```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>

# Use different port
docker run -p 9000:8000 hemophilia-api
```

### **Container Won't Start**

```bash
# Check logs
docker logs hemophilia-api

# Check if port is available
docker port hemophilia-api

# Rebuild without cache
docker build --no-cache -t hemophilia-api .

# Run with debug
docker run -it hemophilia-api bash
```

### **Database Connection Failed**

```bash
# Check database service
docker-compose ps

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U postgres -d hemophilia -c "SELECT 1"

# Rebuild database
docker-compose down -v
docker-compose up -d db
docker-compose exec db psql -U postgres -f init.sql
```

### **Out of Memory**

```bash
# Check memory usage
docker stats

# Limit memory in docker-compose
services:
  api:
    mem_limit: 512m
    memswap_limit: 1g

# Increase Docker memory in settings
```

---

## 📊 Performance Optimization

### **Multi-Stage Build (Smaller Image)**

```dockerfile
# Reduces image size from 1GB+ to ~500MB

FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **Image Size Reduction**

```dockerfile
# Remove unnecessary packages
RUN apt-get install --no-install-recommends -y <package>

# Use alpine images
FROM python:3.11-alpine  # ~40MB vs 900MB for slim

# Keep layers minimal
RUN apt-get update && \
    apt-get install -y package && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
```

### **Resource Limits**

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

---

## ✅ Pre-Deployment Checklist

- [ ] `.env` file created and Git-ignored
- [ ] `OPENAI_API_KEY` configured
- [ ] Database connection tested
- [ ] ML models accessible
- [ ] All tests passing
- [ ] Health endpoint working
- [ ] Logs configured
- [ ] Error handling verified
- [ ] CORS settings correct
- [ ] Rate limiting enabled
- [ ] Docker image builds successfully
- [ ] Docker Compose stack starts
- [ ] Database migrations run
- [ ] API responds to requests

---

## 📝 Summary

✅ Production-ready Dockerfile  
✅ Docker Compose with PostgreSQL  
✅ Security best practices  
✅ Cloud deployment guides  
✅ Troubleshooting guide  
✅ Performance optimization  
✅ Pre-deployment checklist  

**Ready for containerization and deployment!**
