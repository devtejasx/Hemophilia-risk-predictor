# FastAPI Backend - Complete Quick Start Guide

## 📚 Documentation Map

```
fastapi_backend/
├── README.md                        ← Main overview (start here)
├── QUICK_START.md                   ← This file: get running in 10 minutes
├── ARCHITECTURE.md                  ← Design patterns & architecture
├── API_CLIENT_GUIDE.md              ← How to use the API
├── TESTING_GUIDE.md                 ← Testing & quality assurance
├── DOCKER_DEPLOYMENT.md             ← Containerization & deployment
├── PRODUCTION_DEPLOYMENT.md         ← Production operations & scaling
│
├── main.py                          ← FastAPI application entry
├── config.py                        ← Settings & configuration
├── exceptions.py                    ← Exception handling
│
├── models/                          ← Data validation schemas
│   └── __init__.py                  (30+ Pydantic models)
│
├── database/                        ← Database layer
│   └── __init__.py                  (SQLite/PostgreSQL wrapper)
│
├── services/                        ← Business logic
│   ├── prediction_service.py        (ML predictions)
│   ├── chat_service.py              (OpenAI integration)
│   ├── patient_service.py           (CRUD operations)
│   └── analytics_service.py         (Statistics)
│
├── routers/                         ← API endpoints
│   ├── predictions_router.py        (/api/v1/predictions)
│   ├── chat_router.py               (/api/v1/chat)
│   ├── patients_router.py           (/api/v1/patients)
│   └── analytics_router.py          (/api/v1/analytics)
│
├── requirements.txt                 ← Python dependencies
├── .env.example                     ← Configuration template
└── Dockerfile                       ← Container image
```

---

## ⚡ 10-Minute Quick Start

### **Step 1: Prerequisites (1 minute)**

```bash
# Check Python version (need 3.9+)
python --version

# Clone repository
git clone <your-repo>
cd fastapi_backend
```

### **Step 2: Setup Environment (2 minutes)**

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 3: Configure Application (2 minutes)**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# (Add your OpenAI API key, model paths, etc.)
```

### **Step 4: Initialize Database (1 minute)**

```bash
# SQLite will auto-create
# Or for PostgreSQL, create database first:
# createdb hemophilia

# App will auto-initialize tables on startup
```

### **Step 5: Run Application (1 minute)**

```bash
# Start development server
uvicorn main:app --reload

# Server runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### **Step 6: Test It Works (2 minutes)**

```bash
# In another terminal
# Create a patient
curl -X POST http://localhost:8000/api/v1/patients \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "age": 35,
    "gender": "M",
    "severity": "severe",
    "mutation": "intron22",
    "dose_intensity": 50.0,
    "exposure_days": 365,
    "fviii_inhibitor": false
  }'

# Make prediction
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
  }'
```

**✅ Done! API is running.**

---

## 🔧 Common Development Tasks

### **Run Tests**

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### **Check Code Quality**

```bash
# Install dev tools
pip install pylint flake8 black mypy

# Format code
black fastapi_backend/

# Lint
pylint fastapi_backend/
flake8 fastapi_backend/

# Type check
mypy fastapi_backend/
```

### **View API Documentation**

```
# Swagger UI
http://localhost:8000/docs

# ReDoc
http://localhost:8000/redoc

# OpenAPI JSON
http://localhost:8000/openapi.json
```

### **Access Database**

```bash
# SQLite
sqlite3 patients.db

# PostgreSQL
psql postgresql://user:password@localhost/hemophilia

# View tables
SELECT * FROM patients LIMIT 5;
SELECT * FROM predictions;
SELECT * FROM conversations;
```

---

## 📖 Feature Overview

### **🧑‍⚕️ Patient Management**
- Create, read, update, delete patients
- Search by severity or mutation type
- Track patient history
- **Endpoints**: `/api/v1/patients`

### **🧠 ML Predictions**
- Risk score prediction using ensemble models
- Feature importance analysis
- Batch predictions
- Prediction history per patient
- **Endpoints**: `/api/v1/predictions`

### **💬 AI Chat**
- Clinical decision support via GPT-4
- Patient context-aware responses
- Conversation history tracking
- **Endpoints**: `/api/v1/chat`

### **📊 Analytics**
- Patient statistics dashboard
- Risk trends over time
- Mutation distribution
- Severity distribution
- High-risk patient ranking
- **Endpoints**: `/api/v1/analytics`

---

## 🎯 Workflow Examples

### **Example 1: Single Prediction**

```python
import requests

api_url = "http://localhost:8000"

# 1. Create patient
patient = requests.post(
    f"{api_url}/api/v1/patients",
    json={
        "name": "Alice Smith",
        "age": 28,
        "gender": "F",
        "severity": "moderate",
        "mutation": "intron1",
        "dose_intensity": 40.0,
        "exposure_days": 200,
        "fviii_inhibitor": True
    }
).json()

patient_id = patient["id"]

# 2. Get prediction
prediction = requests.post(
    f"{api_url}/api/v1/predictions",
    json={
        "age": 28,
        "gender": "F",
        "severity": "moderate",
        "mutation": "intron1",
        "dose_intensity": 40.0,
        "exposure_days": 200,
        "fviii_inhibitor": True
    },
    params={"patient_id": patient_id}
).json()

print(f"Risk Score: {prediction['risk_score']:.2%}")
print(f"Category: {prediction['severity_category']}")

# 3. Chat with AI
chat_response = requests.post(
    f"{api_url}/api/v1/chat",
    json={
        "patient_id": patient_id,
        "message": "What treatment options are available?"
    }
).json()

print(f"AI: {chat_response['response']}")
```

### **Example 2: Batch Analysis**

```python
# Load patient data
import pandas as pd

patients_df = pd.read_csv("patients.csv")

# Create patients
for _, row in patients_df.iterrows():
    patient_data = row.to_dict()
    requests.post(f"{api_url}/api/v1/patients", json=patient_data)

# Get predictions for all
predictions = requests.post(
    f"{api_url}/api/v1/predictions/batch",
    json=[row.to_dict() for _, row in patients_df.iterrows()]
).json()

# Analyze results
results_df = pd.DataFrame(predictions)
print(results_df.describe())
```

### **Example 3: Dashboard**

```python
# Get statistics
dashboard = requests.get(
    f"{api_url}/api/v1/analytics/dashboard"
).json()

print(f"Total Patients: {dashboard['total_patients']}")
print(f"Average Risk: {dashboard['average_risk_score']:.2%}")

# Get high-risk patients
high_risk = requests.get(
    f"{api_url}/api/v1/analytics/high-risk?limit=10"
).json()

for patient in high_risk:
    print(f"  {patient['name']}: {patient['risk_score']:.2%}")
```

---

## 🐳 Docker Quick Start

### **One-Command Deployment**

```bash
# Build and run with Docker Compose
docker-compose up -d

# API runs at http://localhost:8000
# Database at localhost:5432
# Admin UI at http://localhost:8080

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

### **Manual Docker**

```bash
# Build image
docker build -t hemophilia-api .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxxxx \
  hemophilia-api

# View logs
docker logs <container_id>
```

---

## 🚀 Deployment

### **To Heroku**

```bash
# One command
git push heroku main

# View logs
heroku logs --tail
```

### **To Docker Hub**

```bash
docker build -t yourusername/hemophilia-api .
docker push yourusername/hemophilia-api
```

### **To AWS/GCP/Azure**

See `DOCKER_DEPLOYMENT.md` for detailed cloud deployment guides.

---

## 🔍 Troubleshooting

### **Port 8000 Already in Use**

```bash
# Find what's using it
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn main:app --port 9000
```

### **ModuleNotFoundError**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
which python  # or: where python (Windows)
```

### **Database Connection Error**

```bash
# Check database running
psql --version  # PostgreSQL
sqlite3 <path>  # SQLite

# Check connection string in .env
cat .env | grep DATABASE_URL
```

### **OpenAI API Error**

```bash
# Verify API key
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows

# Check rate limits
# See OpenAI dashboard: https://platform.openai.com
```

### **ML Models Not Found**

```bash
# Verify model paths in .env
cat .env | grep MODEL

# Check files exist
ls -la models/

# Download if needed
python download_models.py
```

---

## 📊 Next Steps

### **Development**
1. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) to understand design
2. Read [`TESTING_GUIDE.md`](TESTING_GUIDE.md) to write tests
3. Make code changes and run `pytest`

### **Deployment**
1. Read [`DOCKER_DEPLOYMENT.md`](DOCKER_DEPLOYMENT.md) for containerization
2. Read [`PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md) for production
3. Follow deployment strategy for your platform

### **Integration**
1. Read [`API_CLIENT_GUIDE.md`](API_CLIENT_GUIDE.md) for client code
2. Integrate with frontend (Streamlit, React, etc.)
3. Test end-to-end workflows

### **Advanced**
- Implement custom authentication
- Add more ML models
- Implement caching layer
- Scale to Kubernetes
- Set up monitoring & alerting

---

## 📞 Quick Reference

### **Key Files**
| File | Purpose |
|------|---------|
| `main.py` | Application entry point |
| `config.py` | Settings & configuration |
| `models/__init__.py` | Data validation schemas |
| `services/` | Business logic |
| `routers/` | API endpoints |

### **Key Endpoints**
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/patients` | Create patient |
| GET | `/api/v1/patients/{id}` | Get patient |
| POST | `/api/v1/predictions` | Make prediction |
| POST | `/api/v1/chat` | Send message |
| GET | `/api/v1/analytics/dashboard` | Get statistics |

### **Key Commands**
```bash
uvicorn main:app --reload           # Run dev
pytest                               # Run tests
docker-compose up -d                # Run with Docker
docker-compose logs -f api          # View logs
git push heroku main                # Deploy to Heroku
```

---

## 💡 Pro Tips

1. **Use Swagger UI** (`/docs`) for interactive API testing during development
2. **Check logs frequently** - `docker-compose logs -f` or `tail -f app.log`
3. **Run tests before committing** - avoid breaking the CI/CD
4. **Keep `.env` out of Git** - use `.env.example` instead
5. **Monitor API responses** - check response times and error rates
6. **Backup database regularly** - especially before major changes
7. **Use git branches** - develop features on separate branches
8. **Document API changes** - update this guide when adding features

---

## 🎓 Learning Path

**Beginner**
1. Run the server locally
2. Test endpoints with Swagger UI
3. Read API responses
4. Create test patients

**Intermediate**
1. Write API client code
2. Create integration tests
3. Understand database schema
4. Explore services layer

**Advanced**
1. Modify services for custom logic
2. Add new endpoints
3. Implement caching
4. Deploy to production

---

## ✅ Verification Checklist

After setup, verify:

- [ ] `uvicorn main:app --reload` starts without errors
- [ ] Swagger UI accessible at `http://localhost:8000/docs`
- [ ] Can create patient via POST `/api/v1/patients`
- [ ] Can make prediction via POST `/api/v1/predictions`
- [ ] Can send message via POST `/api/v1/chat`
- [ ] Database creates tables automatically
- [ ] All tests pass with `pytest`
- [ ] Can access documentation at `/docs`

---

## 🚀 Summary

✅ Understand architecture → Read [`ARCHITECTURE.md`](ARCHITECTURE.md)  
✅ Set up locally → Follow 10-minute quick start above  
✅ Write integration code → Read [`API_CLIENT_GUIDE.md`](API_CLIENT_GUIDE.md)  
✅ Test thoroughly → Follow [`TESTING_GUIDE.md`](TESTING_GUIDE.md)  
✅ Deploy to production → Follow [`PRODUCTION_DEPLOYMENT.md`](PRODUCTION_DEPLOYMENT.md)  

**You're ready to build! 🎉**
