# FastAPI Medical AI Platform - Complete Developer Handbook

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Navigation](#quick-navigation)
3. [Development Environment](#development-environment)
4. [System Architecture](#system-architecture)
5. [Feature Modules](#feature-modules)
6. [Common Tasks](#common-tasks)
7. [Code Organization](#code-organization)
8. [Testing Strategy](#testing-strategy)
9. [Deployment & Operations](#deployment--operations)
10. [Troubleshooting](#troubleshooting)
11. [Resources & References](#resources--references)

---

## 📖 Project Overview

### **What is This?**

A **production-ready FastAPI backend** for a medical AI platform that provides:
- 🧬 **Hemophilia Risk Prediction** - ML models (Random Forest + XGBoost)
- 💬 **Clinical Chatbot** - OpenAI GPT-4 integration
- 👤 **Patient Management** - Full CRUD with PostgreSQL/SQLite
- 📊 **Analytics Dashboard** - Real-time statistics and insights

### **Tech Stack**

```
Frontend: Streamlit/React → Backend: FastAPI → Database: PostgreSQL/SQLite
                          ↓
                   ML Services (SHAP, Scikit-learn)
                   AI Services (OpenAI GPT-4)
                   Redis Cache
```

### **Key Stats**

- **25+ API endpoints** across 4 routers
- **4 microservices** handling different domains
- **30+ Pydantic models** for data validation
- **100% async/await** for performance
- **Clean architecture** with separation of concerns
- **Production-ready** with error handling, logging, monitoring

---

## 🗂️ Quick Navigation

### **I Want To...**

| Goal | Start Here | Time |
|------|-----------|------|
| Get started quickly | [QUICK_START.md](QUICK_START.md) | 10 min |
| Understand architecture | [ARCHITECTURE.md](ARCHITECTURE.md) | 20 min |
| Build a client/integrate | [API_CLIENT_GUIDE.md](API_CLIENT_GUIDE.md) | 15 min |
| Write tests | [TESTING_GUIDE.md](TESTING_GUIDE.md) | 30 min |
| Deploy to production | [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) | 30 min |
| Use Docker | [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) | 20 min |
| Understand full API | [README.md](README.md) | 20 min |
| Debug an issue | [Troubleshooting](#troubleshooting) | varies |

---

## 🛠️ Development Environment

### **Prerequisites**

```bash
# Required
- Python 3.9+ (check: python --version)
- Git (check: git --version)
- Virtual environment capable system

# Optional
- Docker & Docker Compose
- PostgreSQL (default is SQLite)
- Redis (for caching)
```

### **Setup (5 minutes)**

```bash
# 1. Clone & enter
git clone <repo> && cd fastapi_backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env with your API keys

# 5. Run
uvicorn main:app --reload

# 6. Access
# http://localhost:8000/docs (Swagger UI)
```

### **Verification**

```bash
# API health
curl http://localhost:8000/health

# Swagger docs
open http://localhost:8000/docs

# Create patient (test data)
curl -X POST http://localhost:8000/api/v1/patients -H "Content-Type: application/json" \
  -d '{"name":"Test","age":30,"gender":"M","severity":"severe","mutation":"intron22","dose_intensity":50.0,"exposure_days":365,"fviii_inhibitor":false}'
```

---

## 🏗️ System Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────┐
│                 HTTP Clients                         │
│    (Streamlit, Browser, Mobile, CLI, Testing)       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              API Layer (FastAPI)                     │
│  - predictions_router.py ( /api/v1/predictions)    │
│  - patients_router.py    ( /api/v1/patients)       │
│  - chat_router.py        ( /api/v1/chat)           │
│  - analytics_router.py   ( /api/v1/analytics)      │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         Services Layer (Business Logic)              │
│  - PredictionService    (ML inference)              │
│  - ChatService          (OpenAI API)                │
│  - PatientService       (CRUD)                      │
│  - AnalyticsService     (Statistics)                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│        Data Layer (External Services)                │
│  ├─ PostgreSQL/SQLite Database                      │
│  ├─ ML Models (Pickle files)                        │
│  ├─ OpenAI API                                      │
│  ├─ Redis Cache (optional)                          │
│  └─ External APIs                                   │
└─────────────────────────────────────────────────────┘
```

### **Request Flow (Example: Prediction)**

```
POST /api/v1/predictions
  ↓
predictions_router.create_prediction()
  ↓
PredictionService.predict()
  ├─ Load models (RF, XGB)
  ├─ Prepare features
  ├─ Run ensemble
  ├─ Calculate SHAP
  └─ Return result
  ↓
Save to database (optional)
  ↓
Return JSON response
```

### **Data Model**

```
patients
├─ id, name, age, gender
├─ severity, mutation
├─ dose_intensity, exposure_days
├─ fviii_inhibitor
└─ timestamps

predictions
├─ id, patient_id
├─ risk_score, severity_category
├─ explanation, model_version
└─ timestamps

conversations
├─ id, patient_id
├─ user_message, ai_response
└─ timestamps
```

---

## 🔧 Feature Modules

### **1. Predictions (ML Models)**

**File**: `services/prediction_service.py`

**Capabilities**:
- Load RF + XGBoost ensemble models
- Single & batch predictions
- SHAP feature importance
- Risk categorization

**Usage**:
```python
service = PredictionService(db)
result = service.predict(patient_data)
# {
#   "risk_score": 0.75,
#   "severity_category": "HIGH",
#   "explanation": "...",
#   "feature_importance": {...}
# }
```

### **2. Chat (AI Clinical Support)**

**File**: `services/chat_service.py`

**Capabilities**:
- OpenAI GPT-4 integration
- Clinical system prompts
- Patient context injection
- Conversation history

**Usage**:
```python
service = ChatService()
response = service.get_clinical_response(patient_id, message)
# "Based on your condition, the recommended..."
```

### **3. Patient Management (CRUD)**

**File**: `services/patient_service.py`

**Capabilities**:
- Create/Read/Update/Delete
- Search by severity/mutation
- Pagination support
- Data validation

**Usage**:
```python
service = PatientService(db)
patient = service.create_patient(patient_data)
patients = service.search_patients("severe")
```

### **4. Analytics (Statistics)**

**File**: `services/analytics_service.py`

**Capabilities**:
- Dashboard statistics
- Risk trends over time
- Distribution analysis
- High-risk patient ranking

**Usage**:
```python
service = AnalyticsService(db)
dashboard = service.get_dashboard_stats()
# {"total_patients": 1000, "average_risk": 0.45, ...}
```

---

## 📋 Common Tasks

### **Task 1: Add a New Endpoint**

```python
# 1. Define model in models/__init__.py
class NewRequest(BaseModel):
    field1: str
    field2: int

# 2. Add service method in services/new_service.py
def process_data(self, data: NewRequest) -> dict:
    # Business logic
    return result

# 3. Create router endpoint in routers/new_router.py
@router.post("/new-endpoint")
async def new_endpoint(data: NewRequest, service: NewService = Depends(...)):
    return service.process_data(data)

# 4. Register router in main.py
app.include_router(new_router.router)

# 5. Test it
curl -X POST http://localhost:8000/api/v1/new-endpoint -H "Content-Type: application/json" -d '{...}'
```

### **Task 2: Add Database Field**

```python
# 1. Update Pydantic model in models/__init__.py
class PatientResponse(BaseModel):
    id: int
    name: str
    new_field: str  # Add here

# 2. Update database schema in database/__init__.py
CREATE TABLE patients (
    # ... existing fields
    new_field VARCHAR(100),  # Add here
);

# 3. Update service in services/patient_service.py
def create_patient(self, data: PatientCreate):
    # Map new_field
    query = "INSERT INTO patients (..., new_field) VALUES (..., ?)"

# 4. Run migrations
# (For production, use Alembic)
```

### **Task 3: Add Authentication**

```python
# 1. Add to config.py
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# 2. Create auth service
class AuthService:
    def create_token(self, data: dict):
        to_encode = data.copy()
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return token

# 3. Add dependency
def get_current_user(token: str = Depends(HTTPBearer())):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload

# 4. Protect endpoint
@router.post("/protected")
async def protected(user = Depends(get_current_user)):
    return {"user": user}
```

### **Task 4: Add Caching**

```python
# 1. Use @cache_result decorator
from functools import lru_cache

@lru_cache(maxsize=128)
def get_high_risk_patients():
    return analytics_service.get_high_risk_patients()

# 2. Or use Redis
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def get_cached(key):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None

def set_cache(key, value, ttl=3600):
    redis_client.setex(key, ttl, json.dumps(value))
```

---

## 🗂️ Code Organization

### **Directory Structure**

```
fastapi_backend/
├── main.py                    [200 lines] Main FastAPI app
├── config.py                  [35 lines]  Settings
├── exceptions.py              [70 lines]  Exception classes
│
├── models/
│   └── __init__.py           [240 lines] 30+ Pydantic models
│
├── database/
│   └── __init__.py           [150 lines] Database wrapper
│
├── services/
│   ├── prediction_service.py [200 lines] ML inference
│   ├── chat_service.py       [140 lines] OpenAI integration
│   ├── patient_service.py    [250 lines] CRUD operations
│   └── analytics_service.py  [220 lines] Statistics
│
├── routers/
│   ├── predictions_router.py [110 lines] /predictions endpoints
│   ├── chat_router.py        [100 lines] /chat endpoints
│   ├── patients_router.py    [180 lines] /patients endpoints
│   └── analytics_router.py   [150 lines] /analytics endpoints
│
├── Documentation
│   ├── README.md                         Main docs
│   ├── QUICK_START.md                    Getting started
│   ├── ARCHITECTURE.md                   Design patterns
│   ├── API_CLIENT_GUIDE.md               Client code
│   ├── TESTING_GUIDE.md                  Testing
│   ├── DOCKER_DEPLOYMENT.md              Containerization
│   ├── PRODUCTION_DEPLOYMENT.md          Operations
│   └── DEVELOPER_HANDBOOK.md             This file
│
├── Configuration
│   ├── requirements.txt                  Dependencies
│   ├── .env.example                      Env template
│   ├── Dockerfile                        Container
│   └── docker-compose.yml                Services
│
└── Root Files
    ├── .gitignore
    └── README.md
```

### **Naming Conventions**

```python
# Files
- snake_case for files: prediction_service.py
- PascalCase for classes: class PredictionService
- snake_case for functions: def get_prediction()

# Variables
- snake_case for variables: patient_data, risk_score
- UPPER_CASE for constants: MAX_PATIENTS = 1000

# Database
- lowercase for tables: patients, predictions
- snake_case for columns: fviii_inhibitor, exposure_days
```

### **Import Organization**

```python
# Put in order:
# 1. Standard library
import json
import os

# 2. Third-party
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends

# 3. Local
from models import PatientResponse
from services import patient_service
```

---

## 🧪 Testing Strategy

### **Test Organization**

```
tests/
├── conftest.py                  Shared fixtures
├── test_models.py               Pydantic validation
├── test_services/
│   ├── test_prediction_service.py
│   ├── test_chat_service.py
│   ├── test_patient_service.py
│   └── test_analytics_service.py
├── test_routers/
│   ├── test_predictions_router.py
│   ├── test_chat_router.py
│   ├── test_patients_router.py
│   └── test_analytics_router.py
├── test_database.py             Database operations
└── test_integration.py          End-to-end flows
```

### **Quick Test Commands**

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_models.py

# Run with coverage
pytest --cov=. --cov-report=html

# Run and stop on first failure
pytest -x

# Run in verbose mode
pytest -v

# Run specific test
pytest tests/test_models.py::TestPatientModel::test_valid_creation
```

### **Test Coverage Goals**

```
Target: 80%+ overall
├─ Models: 100% (validation is critical)
├─ Services: 90%+ (business logic)
├─ Routers: 85%+ (endpoints)
├─ Database: 90%+ (data integrity)
└─ Integration: 70%+ (complex flows)
```

---

## 🚀 Deployment & Operations

### **Local Development**

```bash
# Terminal 1: Start API
cd fastapi_backend
source venv/bin/activate
uvicorn main:app --reload

# Terminal 2: Run tests
pytest --watch

# Terminal 3: Monitor logs
tail -f app.log | jq .
```

### **staging/Production**

**Option 1: Docker Compose** (Recommended for getting started)
```bash
docker-compose up -d
docker-compose logs -f api
```

**Option 2: Heroku**
```bash
git push heroku main
heroku logs --tail
```

**Option 3: Cloud (AWS, GCP, Azure)**
See `PRODUCTION_DEPLOYMENT.md` for detailed guides.

### **Health Monitoring**

```bash
# Check health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready

# Monitor logs
docker-compose logs -f api

# Check metrics
curl http://localhost:8000/metrics
```

---

## 🔍 Troubleshooting

### **Common Issues**

| Issue | Cause | Solution |
|-------|-------|----------|
| Port 8000 in use | Another process | `lsof -i :8000` then `kill -9 <pid>` |
| Import error | Missing package | `pip install -r requirements.txt` |
| Database not found | Connection string wrong | Check `.env` DATABASE_URL |
| OpenAI error | Missing/wrong API key | Check `.env` OPENAI_API_KEY |
| Model not found | Wrong path | Check `.env` model paths, verify files exist |
| Tests failing | DB state dirty | Delete test DB, run fresh |

### **Debug Mode**

```python
# Add to main.py
import logging

logging.basicConfig(level=logging.DEBUG)

# Or set via environment
export LOGLEVEL=DEBUG
```

### **Performance Issues**

```bash
# Monitor resource usage
docker stats

# Check slow queries
psql -d hemophilia -c "SELECT query, calls, total_time FROM pg_stat_statements ORDER BY total_time DESC;"

# Profile code
python -m cProfile -s cumulative main.py

# Check for memory leaks
docker exec hemophilia-api python -m memory_profiler app.py
```

---

## 📚 Resources & References

### **Official Documentation**

- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Pydantic Docs](https://docs.pydantic.dev)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Scikit-learn Docs](https://scikit-learn.org)

### **Articles & Tutorials**

- [FastAPI Best Practices](https://fastapi.tiangolo.com/advanced/)
- [Clean Code in Python](https://realpython.com/clean-code-python/)
- [SQL Best Practices](https://www.postgresql.org/docs/current/sql.html)
- [Testing Best Practices](https://pytest.org/en/7.1.x/goodpractices.html)

### **Tools**

- **API Testing**: Postman, Insomnia, Thunder Client
- **Database**: DBeaver, pgAdmin, SQLiteStudio
- **Monitoring**: Prometheus, Grafana, Datadog
- **Logging**: ELK Stack, Splunk, CloudWatch
- **CI/CD**: GitHub Actions, GitLab CI, Jenkins

### **Community**

- GitHub Issues for bug reports
- Discussions for feature requests
- Email for security issues

---

## 🎯 Development Workflow

### **Daily Workflow**

```
1. Start of day
   → Pull latest code: git pull
   → Update libs: pip install -r requirements.txt
   → Run tests: pytest

2. During work
   → Create feature branch: git checkout -b feature/my-feature
   → Write code focusing on one task
   → Run tests frequently: pytest
   → Commit regularly: git commit -m "..."

3. End of work
   → Run full test suite: pytest --cov
   → Check code quality: pylint, flake8
   → Push to remote: git push origin feature/my-feature
   → Create pull request for review

4. Before release
   → Merge to main after review
   → Run integration tests
   → Deploy to staging
   → Test in staging environment
   → Deploy to production
```

### **Git Workflow**

```bash
# Feature development
git checkout -b feature/patient-search
# ... make changes ...
git add .
git commit -m "feat(patients): add search by mutation"
git push origin feature/patient-search
# Create pull request on GitHub

# After review & merge
git checkout main
git pull
git branch -d feature/patient-search
```

### **Commit Message Format**

```
type(scope): description

feat(patients): add search by severity
fix(predictions): handle missing features
docs(api): update endpoint examples
test(routers): add patient CRUD tests
refactor(services): simplify prediction logic
```

---

## 📞 Support & Help

### **Ask Questions About**

- **Architecture**: See `ARCHITECTURE.md`
- **Endpoints**: See `README.md` or `/docs`
- **Testing**: See `TESTING_GUIDE.md`
- **Deployment**: See `PRODUCTION_DEPLOYMENT.md`
- **Integration**: See `API_CLIENT_GUIDE.md`

### **Report Issues**

Create GitHub issue with:
1. Description
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment info (OS, Python version, etc.)

### **Get Help**

```bash
# Check documentation
grep -r "your-question" docs/

# Run with verbose mode
pytest -vv
uvicorn main:app --log-level debug

# Debug specific function
python -c "from module import func; help(func)"
```

---

## ✅ Quick Checklist

### **Before Coding**
- [ ] Latest code pulled (`git pull`)
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests passing (`pytest`)

### **While Coding**
- [ ] Following code style (PEP 8)
- [ ] Adding docstrings
- [ ] Writing tests
- [ ] Adding type hints
- [ ] Committing frequently

### **Before Commit**
- [ ] All tests pass (`pytest`)
- [ ] Code formatted (`black`)
- [ ] Linting clean (`pylint`, `flake8`)
- [ ] Types checked (`mypy`)
- [ ] No secrets in code

### **Before Merging**
- [ ] PR reviewed
- [ ] Merge conflicts resolved
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Integration tests pass

---

## 🎓 Learning Path

```
Beginner → Intermediate → Advanced

Beginner (Week 1)
  1. Read QUICK_START.md
  2. Run server locally
  3. Test endpoints in Swagger UI
  4. Create test patients
  5. Review README.md

Intermediate (Weeks 2-3)
  1. Study ARCHITECTURE.md
  2. Write simple API client
  3. Add unit tests
  4. Read all service code
  5. Deploy to Docker

Advanced (Weeks 4+)
  1. Modify services for custom logic
  2. Add new endpoints
  3. Implement caching
  4. Deploy to production
  5. Set up monitoring
```

---

## 🎉 Summary

**You now have everything needed to:**

✅ Understand the codebase  
✅ Set up development environment  
✅ Write and test code  
✅ Deploy to production  
✅ Troubleshoot issues  
✅ Integrate with other systems  
✅ Scale the application  

**Next Steps:**
1. Start with [QUICK_START.md](QUICK_START.md) (10 minutes)
2. Run the server locally
3. Explore Swagger UI (`/docs`)
4. Read [ARCHITECTURE.md](ARCHITECTURE.md) for deeper understanding
5. Write your first test
6. Deploy!

---

**Happy coding! 🚀**

*For questions or issues, refer to specific documentation or GitHub issues.*
