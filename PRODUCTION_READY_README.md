# 🏥 Hemophilia AI Platform - Production Ready System

**Status**: ✅ **PRODUCTION READY** - Full-stack clinical decision support system with enterprise-grade security, performance, and reliability.

---

## 📦 What's Included

### ✨ **New Production-Grade Modules**

1. **`config.py`** - Centralized configuration management
   - Environment-based settings (development, staging, production)
   - Application, API, database, security, logging, cache configuration
   - Flexible for all deployment environments

2. **`logging_config.py`** - Enterprise logging infrastructure
   - Structured JSON logging for log aggregation
   - Rotating file handlers with automatic cleanup
   - Separate error logs for debugging
   - Context-aware logging helpers

3. **`models_schema.py`** - Type-safe API contracts
   - 30+ Pydantic models for request/response validation
   - Comprehensive enums (Severity, Mutation, UserRole, etc.)
   - Automatic documentation generation
   - Built-in data validation and error handling

4. **`security.py`** - Production-grade authentication
   - Password hashing with bcrypt
   - JWT token management (access + refresh tokens)
   - Role-Based Access Control (RBAC) with 5 roles
   - API key management for service-to-service auth
   - Session management with expiration

5. **`api_production.py`** - Enhanced FastAPI backend
   - Full REST API with 15+ endpoints
   - Request/response validation with Pydantic
   - JWT authentication on protected routes
   - Role-based access control middleware
   - Comprehensive error handling
   - Request ID tracking
   - Rate limiting (100 req/min per IP)
   - CORS configuration
   - Health check endpoints

6. **`cache_layer.py`** - Multi-level caching system
   - Prediction result caching (1-hour TTL)
   - Model caching (prevents repeated disk loads)
   - Feature engineering cache
   - Batch prediction caching
   - Cache statistics and cleanup utilities

7. **`.env.example`** - Comprehensive configuration template
   - 40+ configuration options
   - Production-safe defaults
   - Clear documentation
   - Optional services (email, monitoring)

8. **`PRODUCTION_DEPLOYMENT_GUIDE.md`** - Complete deployment documentation
   - Local development setup
   - Docker containerization
   - Multiple cloud deployment options (Heroku, AWS, GCP)
   - Security best practices
   - Monitoring and logging strategies
   - Database management
   - Performance optimization

---

## 🚀 Getting Started

### **Quick Local Setup**

```bash
# 1. Clone repository
git clone https://github.com/devtejasx/Hemophilia-risk-predictor.git
cd Hemophilia-risk-predictor

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings (especially OPENAI_API_KEY)

# 5. Start production backend
python api_production.py

# 6. Start frontend (new terminal)
streamlit run app.py
```

Access:
- 🌐 Frontend: http://localhost:8501
- 🔌 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 💚 Health: http://localhost:8000/health

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│            STREAMLIT FRONTEND (Port 8501)                │
│  • Multi-page UI (Patient Form, Results, History, etc)  │
│  • Real-time visualizations & SHAP explanations         │
│  • Doctor Dashboard with analytics                      │
│  • GPT-4 powered chatbot interface                      │
└────────────┬────────────────────────────────────────────┘
             │ HTTP Requests
             ▼
┌─────────────────────────────────────────────────────────┐
│         FASTAPI BACKEND (Port 8000)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Endpoints:                                       │   │
│  │  🔒 /api/v1/auth/* - Authentication (JWT)       │   │
│  │  🧠 /api/v1/predict - ML Predictions            │   │
│  │  💬 /api/v1/chat - AI Chatbot                   │   │
│  │  👥 /api/v1/patients - Patient CRUD             │   │
│  │  📊 /api/v1/analytics - Dashboard Stats         │   │
│  │  ❤️ /health - Health Check                      │   │
│  └──────────────────────────────────────────────────┘   │
│  • Rate limiting (100 req/min)                          │
│  • JWT authentication & RBAC                            │
│  • Structured request/response validation               │
│  • Comprehensive error handling                         │
│  • JSON logging with request tracking                   │
│  • Multi-level caching                                  │
└────────────┬────────────────────────────────────────────┘
             │ Read/Write
             ▼
┌─────────────────────────────────────────────────────────┐
│              DATABASE LAYER                              │
│  • SQLite (development) / PostgreSQL (production)       │
│  • 7 tables: patients, conversations, doctor_notes, ... │
│  • Connection pooling & transaction management          │
│  • Audit logging for compliance                         │
└─────────────────────────────────────────────────────────┘
             │
┌─────────────────────────────────────────────────────────┐
│          ML INFERENCE ENGINE                             │
│  • Random Forest + XGBoost ensemble                     │
│  • SHAP explainability (waterfall, summary plots)      │
│  • Prediction caching (1-hour TTL)                     │
│  • Batch processing support                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│          EXTERNAL SERVICES                               │
│  🔐 OpenAI API (GPT-4 Chatbot)                          │
│  📊 Sentry (Error tracking - optional)                  │
│  📧 Email Service (Notifications - optional)            │
└─────────────────────────────────────────────────────────┘
```

---

## 🔐 Security Features

✅ **Authentication & Authorization**
- JWT tokens (access + refresh)
- Role-based access control (5 roles)
- Password hashing with bcrypt
- Session management with expiration

✅ **API Security**
- Rate limiting (100 req/min per IP)
- CORS configuration
- Request ID tracking
- API key management for services

✅ **Data Protection**
- Audit logging for compliance
- Patient data encryption-ready
- Password security validation
- Environment variable management

✅ **Production-Grade**
- Structured logging with JSON
- Error tracking ready (Sentry integration)
- Health checks
- Security headers

---

## 🚀 Deployment Options

### **1. Local/Docker (Recommended for testing)**
```bash
docker-compose up -d
```

### **2. Heroku (Easiest cloud option)**
```bash
heroku create your-app
heroku config:set OPENAI_API_KEY=sk-...
git push heroku main
```

### **3. AWS EC2**
- Full setup instructions in PRODUCTION_DEPLOYMENT_GUIDE.md
- Nginx reverse proxy
- SSL/HTTPS with Let's Encrypt
- Systemd service management

### **4. Google Cloud Run**
```bash
gcloud run deploy hemophilia-api --source .
```

See **PRODUCTION_DEPLOYMENT_GUIDE.md** for detailed instructions.

---

## 📊 API Endpoints

### **Authentication**
```
POST   /api/v1/auth/register       - Register new user
POST   /api/v1/auth/login          - Login (returns JWT)
POST   /api/v1/auth/refresh        - Refresh access token
```

### **Predictions**
```
POST   /api/v1/predict             - Generate risk prediction
GET    /api/v1/predict/{id}        - Get prediction history
```

### **Chat**
```
POST   /api/v1/chat                - Send chat message
GET    /api/v1/chat/{patient_id}   - Get chat history
```

### **Patients**
```
POST   /api/v1/patients            - Create patient
GET    /api/v1/patients/{id}       - Get patient
PUT    /api/v1/patients/{id}       - Update patient
DELETE /api/v1/patients/{id}       - Delete patient
GET    /api/v1/patients            - List all patients
```

### **Analytics**
```
GET    /api/v1/analytics/dashboard - Dashboard statistics
GET    /api/v1/analytics/trends    - Patient trends
GET    /api/v1/analytics/cohorts   - Cohort analysis
```

### **Health**
```
GET    /health                     - Health status
GET    /ready                      - Readiness check
```

Full documentation: http://localhost:8000/docs (Swagger UI)

---

## 🧪 Testing

### **API Testing**

```bash
# Get bearer token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"doctor1","password":"Password1!"}'

# Use token for authenticated requests
TOKEN="eyJ0eXAiOiJKV1QiLCJhbGc..."
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "dose_intensity": 75.0,
    "exposure_days": 500,
    "severity": "severe",
    "mutation": "intron22"
  }'
```

### **Load Testing**

```bash
pip install locust
locust -f locustfile.py --users 100 --spawn-rate 10
```

---

## 📈 Performance & Optimization

### **Caching Strategy**
- **Predictions**: 1-hour TTL (configurable)
- **Models**: In-memory (loaded once)
- **Features**: 24-hour cache per patient
- **Batch results**: Custom TTL

### **Database Optimization**
- Connection pooling (5 connections)
- Query optimization with indexes
- Automatic cleanup of old records
- Backup rotation (30 days)

### **API Optimization**
- Gzip compression for responses
- Request batching support
- Async processing for long tasks
- Load balancing ready

---

## 📋 Configuration Reference

### **Key Settings (config.py)**

```python
# Environment
ENV: production/staging/development
DEBUG: False (production)

# API
API_HOST: 0.0.0.0
API_PORT: 8000
FRONTEND_URL: http://localhost:8501

# Security
ALGORITHM: HS256
ACCESS_TOKEN_EXPIRE_MINUTES: 30
REFRESH_TOKEN_EXPIRE_DAYS: 7

# Database
DATABASE_URL: sqlite:///./hemophilia_clinic.db
DB_POOL_SIZE: 5
DB_MAX_OVERFLOW: 10

# Rate Limiting
RATE_LIMIT_REQUESTS: 100
RATE_LIMIT_PERIOD_SECONDS: 60

# Cache
CACHE_TTL_SECONDS: 3600
CACHE_ENABLED: True

# Logging
LOG_LEVEL: INFO
LOG_FILE: logs/app.log
```

See PRODUCTION_DEPLOYMENT_GUIDE.md for complete reference.

---

## 🔍 Monitoring & Logging

### **View Logs**
```bash
# All logs
tail -f logs/app.log

# Errors only
grep ERROR logs/app.log | tail -20

# API requests
grep "API Request" logs/app.log
```

### **Check Cache Stats**
```python
from cache_layer import prediction_cache
print(prediction_cache.get_stats())
# Output: {'size': 42, 'hits': 156, 'misses': 28, 'hit_rate': '84.86%', 'ttl_seconds': 3600}
```

### **Monitor API Health**
```bash
curl http://localhost:8000/health
# Output: {"status": "healthy", "timestamp": "2024-04-07T..."}
```

---

## 🛠️ Troubleshooting

### **Issue: `ModuleNotFoundError: No module named 'openai'`**
```bash
pip install -r requirements.txt
```

### **Issue: `OpenAI API key not configured`**
```bash
# Set in .env
OPENAI_API_KEY=sk-your_key_here
```

### **Issue: Models won't load (Memory Error)**
```python
# config.py - Reduce batch size
BATCH_SIZE_PREDICTIONS = 16  # from 32
```

### **Issue: Database is locked**
```bash
# Remove stale connections
rm hemophilia_clinic.db
python -c "from database import init_database; init_database()"
```

---

## 📚 Documentation Files

- **README.md** - Original overview
- **ARCHITECTURE.md** - System design
- **FASTAPI_ARCHITECTURE.md** - API architecture
- **PRODUCTION_DEPLOYMENT_GUIDE.md** - Deployment instructions
- **CLINICAL_ASSISTANT_GUIDE.md** - AI modes documentation
- **SHAP_EXPLAINABILITY_GUIDE.md** - Model explanation
- **QUICKSTART.md** - Quick start guide

---

## 🎯 Production Readiness Checklist

- ✅ Environment configuration system
- ✅ Structured logging with rotation
- ✅ Type-safe API contracts (Pydantic)
- ✅ JWT authentication
- ✅ Role-based access control
- ✅ Rate limiting
- ✅ CORS security
- ✅ Error handling & tracking
- ✅ Request ID tracing
- ✅ Multi-level caching
- ✅ Database connection pooling
- ✅ Health check endpoints
- ✅ Comprehensive API documentation
- ✅ Docker containerization ready
- ✅ Cloud deployment instructions
- ✅ Security best practices
- ✅ Monitoring integration points

---

## 🚀 Next Steps

1. **Configure Environment**: Edit `.env` with your API keys
2. **Test Locally**: Run development setup
3. **Choose Deployment**: Pick cloud provider (Docker/Heroku/AWS/GCP)
4. **Deploy**: Follow PRODUCTION_DEPLOYMENT_GUIDE.md
5. **Monitor**: Set up logging & alerting (Sentry, DataDog optional)
6. **Scale**: Use load balancing for multi-instance deployment

---

## 📞 Support

- 🐛 **Issues**: GitHub Issues
- 📖 **Docs**: See documentation files
- 🔗 **API Docs**: http://localhost:8000/docs
- 📧 **Contact**: Your contact info

---

## 📄 License

See LICENSE file for details

---

**Built with ❤️ for clinical decision support**

Last Updated: April 2024 | Version: 2.0.0 Production Release
