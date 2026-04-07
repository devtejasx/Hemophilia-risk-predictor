# Production Deployment Guide - Hemophilia AI Platform

## 📋 Overview

This guide covers deploying the production-ready Hemophilia AI Platform with all security, performance, and reliability features.

---

## 🚀 Quick Start (Local Development)

### 1. **Clone and Setup Environment**

```bash
# Clone repository
git clone https://github.com/devtejasx/Hemophilia-risk-predictor.git
cd Hemophilia-risk-predictor

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. **Configure Environment**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Add your OpenAI API key
OPENAI_API_KEY=sk-...
SECRET_KEY=your-secure-random-key
```

### 3. **Run Production Backend**

```bash
# Terminal 1: Start FastAPI backend
python api_production.py

# Or with Uvicorn directly
uvicorn api_production:app --host 0.0.0.0 --port 8000 --workers 4

# Or with Gunicorn (recommended for production)
gunicorn -k uvicorn.workers.UvicornWorker api_production:app --bind 0.0.0.0:8000 --workers 4
```

### 4. **Run Frontend (New Terminal)**

```bash
# Terminal 2: Start Streamlit
streamlit run app.py
```

Access the application:
- 🌐 Frontend: http://localhost:8501
- 🔌 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

---

## 🔧 Production Environment Setup

### 1. **Environment Variables**

Create `.env` with all required variables:

```env
# === Environment ===
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# === API Configuration ===
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_URL=https://your-domain.com

# === Security ===
SECRET_KEY=generate-with-openssl-rand-hex-32
OPENAI_API_KEY=sk-your-api-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# === Database ===
DATABASE_URL=sqlite:///./hemophilia_clinic.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/hemophilia_db

# === Rate Limiting ===
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_PERIOD_SECONDS=60

# === Logging ===
LOG_FILE=logs/app.log
LOG_LEVEL=INFO

# === Cache ===
CACHE_ENABLED=True
CACHE_TTL_SECONDS=3600

# === ML Models ===
MODELS_PATH=./models
```

### 2. **Generate Secure Secret Key**

```bash
# Linux/macOS
python -c "import secrets; print(secrets.token_hex(32))"

# Windows PowerShell
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. **Create Required Directories**

```bash
mkdir logs
mkdir models
mkdir -p data/backups
```

---

## 🐳 Docker Deployment

### 1. **Create Dockerfile**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run with Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "api_production:app", \
     "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120"]
```

### 2. **Create docker-compose.yml**

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - DEBUG=False
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=hemophilia
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 3. **Run with Docker Compose**

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

---

## 🚢 Cloud Deployment

### **Heroku**

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create your-app-name

# Add environment variables
heroku config:set OPENAI_API_KEY=sk-...
heroku config:set SECRET_KEY=...
heroku config:set ENVIRONMENT=production

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### **AWS EC2**

```bash
# SSH into instance
ssh -i key.pem ubuntu@your-instance.com

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv

# Clone and setup
git clone <repo>
cd Hemophilia-risk-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env

# Install systemd service
sudo cp hemophilia.service /etc/systemd/system/
sudo systemctl enable hemophilia
sudo systemctl start hemophilia

# Setup Nginx reverse proxy
sudo apt install nginx
sudo cp nginx.conf /etc/nginx/sites-available/hemophilia
sudo ln -s /etc/nginx/sites-available/hemophilia /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### **Google Cloud Run**

```bash
# Configure GCP
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy hemophilia-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 120 \
  --set-env-vars OPENAI_API_KEY=sk-...,SECRET_KEY=...
```

---

## 🔒 Security Best Practices

### 1. **Use Environment Variables**
- Never hardcode secrets
- Rotate API keys regularly
- Use `.env` only for local development

### 2. **Enable HTTPS**
```bash
# Use Let's Encrypt with Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d your-domain.com
```

### 3. **Set Up Firewall**
```bash
# Allow only necessary ports
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw enable
```

### 4. **Database Security**
```bash
# For PostgreSQL, create restricted user
CREATE ROLE hemophilia_app WITH LOGIN PASSWORD 'strong_password';
GRANT CONNECT ON DATABASE hemophilia TO hemophilia_app;
GRANT USAGE ON SCHEMA public TO hemophilia_app;
```

### 5. **API Rate Limiting**
Enabled by default in production config - 100 requests/minute per IP

### 6. **CORS Configuration**
Update allowed origins in `config.py`:
```python
CORS_ORIGINS = [
    "https://your-domain.com",
    "https://www.your-domain.com"
]
```

---

## 📊 Monitoring & Logging

### 1. **Check Application Logs**

```bash
# View logs
tail -f logs/app.log

# Find errors
grep ERROR logs/app.log

# Check API requests
grep "API Request" logs/app.log
```

### 2. **Monitor Performance**

```bash
# Check prediction cache stats
# Via Python shell
from cache_layer import prediction_cache
print(prediction_cache.get_stats())

# Check model cache
from cache_layer import model_cache
print(model_cache.get_cache_info())
```

### 3. **Health Checks**

```bash
# Check API health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready
```

---

## 🗄️ Database Management

### 1. **Initialize Database**

```bash
python -c "from database import init_database; init_database()"
```

### 2. **Backup Database**

```bash
# SQLite backup
cp hemophilia_clinic.db backups/hemophilia_clinic_$(date +%Y%m%d_%H%M%S).db

# PostgreSQL backup
pg_dump hemophilia > backups/hemophilia_$(date +%Y%m%d_%H%M%S).sql
```

### 3. **Restore Database**

```bash
# SQLite restore
cp backups/hemophilia_clinic_*.db hemophilia_clinic.db

# PostgreSQL restore
psql hemophilia < backups/hemophilia_*.sql
```

---

## 📈 Performance Optimization

### 1. **Enable Caching**

All enabled by default:
- Prediction cache (1 hour TTL)
- Model cache (in-memory)
- Feature cache (24 hours TTL)

### 2. **Database Connection Pooling**

Configure in `config.py`:
```python
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10
```

### 3. **Load Balancing**

For multiple instances:
```bash
# Use Nginx as reverse proxy
upstream hemophilia_backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://hemophilia_backend;
    }
}
```

---

## 🧪 Testing Before Deployment

### 1. **Run Health Checks**

```bash
python -m pytest tests/ -v
```

### 2. **Test API Endpoints**

```bash
# Get API docs
curl http://localhost:8000/docs

# Test health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "dose_intensity": 50.0,
    "exposure_days": 365,
    "severity": "severe",
    "mutation": "intron22"
  }'
```

### 3. **Load Testing**

```bash
# Install locust
pip install locust

# Create locustfile.py with test scenarios
# Run tests
locust -f locustfile.py
```

---

## 🚨 Troubleshooting

### **Issue: API won't start**

```bash
# Check port is available
lsof -i :8000

# Check environment variables
echo $OPENAI_API_KEY

# Check logs
tail -f logs/error.log
```

### **Issue: Models won't load**

```bash
# Verify model files exist
ls -la models/

# Test model loading
python -c "import joblib; joblib.load('models/rf.pkl')"
```

### **Issue: Memory errors**

```bash
# Increase available memory
# Or reduce batch size in config.py
BATCH_SIZE_PREDICTIONS = 16  # From 32

# Enable model caching
CACHE_ENABLED = True
```

---

## 📞 Support & Documentation

- **API Documentation**: http://localhost:8000/docs
- **GitHub**: https://github.com/devtejasx/Hemophilia-risk-predictor
- **Issues**: Create GitHub issue with logs and configuration

---

## ✅ Production Readiness Checklist

- [ ] Environment variables configured
- [ ] Secret key generated and secured
- [ ] Database initialized and backed up
- [ ] HTTPS/SSL certificates installed
- [ ] Firewall rules configured
- [ ] API health checks passing
- [ ] Logging enabled and monitored
- [ ] Rate limiting configured
- [ ] CORS origins restricted
- [ ] Database connection pooling enabled
- [ ] Caching configured
- [ ] Backup strategy implemented
- [ ] Monitoring/alerting setup
- [ ] Load balancing configured (if multiple instances)
- [ ] CI/CD pipeline configured
- [ ] Documentation updated
- [ ] Team trained on deployment
- [ ] Rollback plan documented
