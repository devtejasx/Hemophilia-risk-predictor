# FastAPI Backend for Medical AI Platform

**Professional-grade clinical decision support REST API**

## 📋 Overview

This is a production-ready FastAPI backend for a hemophilia AI platform featuring:

- **ML Predictions**: Random Forest + XGBoost ensemble with SHAP explanability
- **AI Chat**: OpenAI GPT-4 integration for clinical decision support
- **Patient Management**: Complete CRUD operations with SQLite
- **Analytics Dashboard**: System-wide statistics and trends
- **Clean Architecture**: Routers, services, models separation
- **Type Safety**: Pydantic models for validation
- **Error Handling**: Comprehensive exception handling
- **Dependency Injection**: Database connections as dependencies

---

## 🏗️ Project Structure

```
fastapi_backend/
├── main.py                          # Main FastAPI app & routes
├── config.py                        # Configuration settings
├── exceptions.py                    # Custom exceptions
├── requirements.txt                 # Dependencies
│
├── models/
│   └── __init__.py                 # Pydantic schemas (PatientBase, PredictionInput, etc)
│
├── database/
│   ├── __init__.py                 # Database connection manager
│   └── connection.py               # Connection utilities
│
├── services/
│   ├── __init__.py
│   ├── prediction_service.py       # ML model management & predictions
│   ├── chat_service.py             # OpenAI GPT-4 integration
│   ├── patient_service.py          # Patient CRUD operations
│   └── analytics_service.py        # Dashboard statistics
│
└── routers/
    ├── __init__.py
    ├── predictions_router.py       # /api/v1/predictions endpoints
    ├── chat_router.py              # /api/v1/chat endpoints
    ├── patients_router.py          # /api/v1/patients endpoints
    └── analytics_router.py         # /api/v1/analytics endpoints
```

---

## 🚀 Quick Start

### 1. **Setup Virtual Environment**

```bash
cd fastapi_backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Configure Environment**

```bash
# Copy from root
cp .env.example .env

# Edit .env with your settings
OPENAI_API_KEY=sk-...
RF_MODEL_PATH=../rf.pkl
XGB_MODEL_PATH=../xgb.pkl
```

### 4. **Run Server**

```bash
python main.py
```

Or with auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger)
- **ReDoc**: http://localhost:8000/redoc

---

## 📊 API Endpoints

### **Health & Status**

```
GET  /                       Root information
GET  /health                 Health status
GET  /ready                  Readiness check
```

### **Predictions** (`/api/v1/predictions`)

```
POST   /                          Generate single prediction
POST   /batch                     Batch predictions (multiple patients)
GET    /patient/{patient_id}      Get patient prediction history
```

**Request (POST /):**
```json
{
  "age": 35,
  "dose_intensity": 75.0,
  "exposure_days": 500,
  "severity": "severe",
  "mutation": "intron22",
  "hemoglobin": 12.5,
  "white_blood_cells": 7.2,
  "platelets": 250
}
```

**Response:**
```json
{
  "risk_score": 0.68,
  "risk_category": "High",
  "confidence": 0.85,
  "top_features": [
    {
      "feature": "age_first_treatment",
      "importance_score": 0.18,
      "impact": "High"
    }
  ],
  "explanation": "Risk prediction based on 3 key factors",
  "timestamp": "2024-04-07T10:30:00"
}
```

### **Chat** (`/api/v1/chat`)

```
POST   /                          Send message to AI
GET    /patient/{patient_id}      Get conversation history
GET    /health                    Chat service status
```

**Request (POST /):**
```json
{
  "patient_id": 1,
  "message": "What is the recommended dose for this patient?",
  "mode": "treatment"
}
```

**Response:**
```json
{
  "message_id": "uuid-string",
  "response": "Based on the patient context, the recommended approach would be...",
  "confidence": 0.85,
  "timestamp": "2024-04-07T10:30:00"
}
```

### **Patients** (`/api/v1/patients`)

```
POST                            Create patient
GET     /                       List patients (paginated)
GET     /{patient_id}           Get patient details
PUT     /{patient_id}           Update patient
DELETE  /{patient_id}           Delete patient
GET     /search/by-severity     Search by severity
GET     /search/by-mutation     Search by mutation
```

**Request (POST):**
```json
{
  "name": "John Doe",
  "age": 35,
  "gender": "M",
  "severity": "severe",
  "mutation": "intron22",
  "dose_intensity": 75.0,
  "exposure_days": 500
}
```

**Response:**
```json
{
  "id": 1,
  "name": "John Doe",
  "age": 35,
  "gender": "M",
  "severity": "severe",
  "mutation": "intron22",
  "dose_intensity": 75.0,
  "exposure_days": 500,
  "risk_score": null,
  "created_at": "2024-04-07T10:30:00",
  "updated_at": null
}
```

### **Analytics** (`/api/v1/analytics`)

```
GET  /dashboard              Dashboard summary statistics
GET  /trends?days=30        Risk trends over time
GET  /mutations             Statistics by mutation type
GET  /severity              Patient distribution by severity
GET  /high-risk?limit=10    Highest risk patients
```

---

## 🔑 Key Features

### 1. **Pydantic Models** (Type Safety)

All endpoints use Pydantic models for automatic validation:

```python
from models import PredictionInput

# Automatic validation on request
# Automatic documentation in Swagger
# Serialization of responses
```

### 2. **Services Layer** (Business Logic)

Separation of concerns:

```python
# prediction_service.py - ML logic
# chat_service.py - AI logic
# patient_service.py - Database logic
# analytics_service.py - Statistics logic
```

### 3. **Dependency Injection** (Testability)

```python
# Database is injected as dependency
@router.get("/{patient_id}")
async def get_patient(patient_id: int, db = Depends(get_db)):
    ...
```

### 4. **Exception Handling** (Robustness)

```python
# Custom exceptions
class PredictionException(MedicalAIException): ...
class PatientNotFound(MedicalAIException): ...
class ChatException(MedicalAIException): ...

# Global exception handlers
# Proper HTTP status codes
# Descriptive error messages
```

### 5. **Routers** (Organization)

Each domain has its own router:

```python
app.include_router(predictions_router.router)
app.include_router(chat_router.router)
app.include_router(patients_router.router)
app.include_router(analytics_router.router)
```

---

## 🗄️ Database Schema

### **patients** table

```sql
CREATE TABLE patients (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    severity TEXT NOT NULL,
    mutation TEXT NOT NULL,
    dose_intensity REAL NOT NULL,
    exposure_days INTEGER NOT NULL,
    risk_score REAL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
```

### **conversations** table

```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    created_at TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
```

### **predictions** table

```sql
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL,
    risk_score REAL NOT NULL,
    risk_category TEXT NOT NULL,
    confidence REAL NOT NULL,
    model_version TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
```

---

## 🔐 Configuration

Edit `config.py` for settings:

```python
# Application
APP_NAME = "Medical AI Platform API"
API_HOST = "0.0.0.0"
API_PORT = 8000

# Database
DATABASE_PATH = "hemophilia_clinic.db"

# OpenAI
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4"

# Models
RF_MODEL_PATH = "rf.pkl"
XGB_MODEL_PATH = "xgb.pkl"
COLUMNS_PATH = "columns.pkl"
```

---

## 📝 Code Examples

### **Using the Client**

```python
import requests

API_URL = "http://localhost:8000"

# 1. Create patient
patient = requests.post(
    f"{API_URL}/api/v1/patients",
    json={
        "name": "John Doe",
        "age": 35,
        "gender": "M",
        "severity": "severe",
        "mutation": "intron22",
        "dose_intensity": 75.0,
        "exposure_days": 500
    }
).json()
print(f"Created patient: {patient['id']}")

# 2. Get prediction
prediction = requests.post(
    f"{API_URL}/api/v1/predictions",
    json={
        "age": 35,
        "dose_intensity": 75.0,
        "exposure_days": 500,
        "severity": "severe",
        "mutation": "intron22"
    },
    params={"patient_id": patient['id']}
).json()
print(f"Risk score: {prediction['risk_score']:.2f}")

# 3. Chat with AI
response = requests.post(
    f"{API_URL}/api/v1/chat",
    json={
        "patient_id": patient['id'],
        "message": "What treatment do you recommend?",
        "mode": "treatment"
    }
).json()
print(f"AI Response: {response['response']}")

# 4. Get analytics
stats = requests.get(
    f"{API_URL}/api/v1/analytics/dashboard"
).json()
print(f"Total patients: {stats['total_patients']}")
```

---

## 🧪 Testing

### **Health Check**

```bash
curl http://localhost:8000/health
```

### **Create Patient**

```bash
curl -X POST http://localhost:8000/api/v1/patients \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Patient",
    "age": 30,
    "gender": "M",
    "severity": "severe",
    "mutation": "intron22",
    "dose_intensity": 50.0,
    "exposure_days": 365
  }'
```

### **Get Prediction**

```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "dose_intensity": 50.0,
    "exposure_days": 365,
    "severity": "severe",
    "mutation": "intron22"
  }'
```

---

## 🐳 Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t medical-ai-api .
docker run -p 8000:8000 medical-ai-api
```

---

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative Docs**: http://localhost:8000/redoc (ReDoc)
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## ✅ Production Checklist

- [ ] Environment variables configured
- [ ] Models files available (rf.pkl, xgb.pkl)
- [ ] OpenAI API key set
- [ ] Database initialized
- [ ] Tests passing
- [ ] CORS configured for production domain
- [ ] Error handling tested
- [ ] Rate limiting configured
- [ ] Logging enabled
- [ ] Health checks passing

---

## 🐛 Troubleshooting

### **Models won't load**

```bash
# Verify model files exist
ls -la *.pkl

# Check model paths in .env
cat .env | grep MODEL
```

### **Chat service not available**

```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Verify OpenAI library installed
pip install openai
```

### **Database errors**

```bash
# Reinitialize database
rm hemophilia_clinic.db
# Restart server to recreate
```

---

## 📞 Support

- **Issues**: GitHub Issues
- **Docs**: Swagger UI at `/docs`
- **Email**: your-email@example.com

---

**Built with ❤️ for clinical decision support**

Version: 1.0.0 | Last Updated: April 2024
