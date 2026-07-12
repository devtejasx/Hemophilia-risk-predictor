# FastAPI Backend Implementation Guide

## 🏗️ Architecture Overview

The application is now split into:
- **Backend (FastAPI)**: REST API server handling all business logic
- **Frontend (Streamlit)**: User interface consuming the REST API

### Backend Structure
```
backend/
├── main.py                      # FastAPI app and route registration
├── models.py                    # Pydantic request/response models
├── ml_utils.py                  # ML prediction logic
├── gpt_utils.py                 # Clinical AI chat logic
├── requirements.txt             # Backend dependencies
└── routers/                      # Organized endpoints
    ├── __init__.py
    ├── predict.py              # /predict endpoints
    ├── chat.py                 # /chat endpoints
    ├── patients.py             # /patients endpoints
    └── analytics.py            # /analytics endpoints
```

## 🚀 Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start FastAPI Server

**Option A: Development mode (with auto-reload)**
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: Production mode**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║  🏥 Hemophilia AI Platform - FastAPI Backend                  ║
║  ─────────────────────────────────────────────────────────────║
║  Starting on: http://localhost:8000                           ║
║  Documentation: http://localhost:8000/docs                    ║
```

### 3. Access API Documentation

**Interactive API Docs (Swagger UI):**
```
http://localhost:8000/docs
```

**Alternative OpenAPI Docs:**
```
http://localhost:8000/redoc
```

### 4. Start Streamlit Frontend (in another terminal)

```bash
streamlit run app.py
```

The Streamlit app will now call the FastAPI backend for all operations.

## 📋 API Endpoints

### 1. Prediction API - `/predict`

**POST /predict** - Predict inhibitor risk
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 12,
    "dose": 2000,
    "exposure": 90,
    "severity": "Moderate",
    "mutation": "Intron 22 Inversion",
    "ethnicity": "Caucasian",
    "blood_type": "O+",
    "previous_inhibitor": false,
    "treatment_adherence": 85
  }'
```

**Response:**
```json
{
  "risk_score": 0.42,
  "risk_category": "Medium",
  "confidence": 0.85,
  "contributing_factors": [
    {"factor": "Severity", "weight": 0.35}
  ],
  "recommendations": [
    "Increase monitoring frequency to every 6 months"
  ],
  "model_used": "ensemble"
}
```

**Query Parameters:**
- `age`: Patient age (0-120)
- `dose`: Treatment dose in units (required)
- `exposure`: Days of treatment (required)
- `severity`: Mild, Moderate, or Severe (required)
- `mutation`: Gene mutation name (required)
- `ethnicity`, `blood_type`, `hla_typing`: Optional clinical parameters
- And 12 more optional clinical parameters

---

### 2. Chat API - `/chat`

**POST /chat** - Query clinical AI assistant
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What could cause persistent joint swelling?",
    "mode": "diagnosis_support",
    "patient_data": {
      "name": "John Smith",
      "age": 12,
      "severity": "Moderate",
      "mutation": "Intron 22 Inversion",
      "bleeding_episodes": 12
    }
  }'
```

**Response:**
```json
{
  "response": "Based on your patient's profile, hemarthrosis (bleeding into joints) is the most likely cause...",
  "mode_used": "diagnosis_support",
  "disclaimer": "⚠️ AI-generated suggestions are for educational discussion only...",
  "sources": ["OpenAI GPT-4", "Clinical Guidelines"],
  "confidence": 0.85
}
```

**Modes:**
- `diagnosis_support` - Interpret symptoms
- `treatment_recommendation` - Optimize treatment
- `risk_explanation` - Explain risk scores
- `monitoring_analysis` - Guide monitoring

**Query Parameters:**
- `question`: User question (required)
- `mode`: Clinical mode (default: diagnosis_support)
- `patient_data`: Optional patient context (Dict)
- `conversation_history`: Optional message history (List)

**GET /chat/modes** - Get available modes
```bash
curl "http://localhost:8000/chat/modes"
```

**GET /chat/definitions** - Get medical definitions
```bash
curl "http://localhost:8000/chat/definitions?terms=inhibitor&terms=hemarthrosis"
```

---

### 3. Patient Management API - `/patients`

**GET /patients** - List all patients
```bash
curl "http://localhost:8000/patients?skip=0&limit=10&severity=Moderate"
```

**GET /patients/{id}** - Get specific patient
```bash
curl "http://localhost:8000/patients/1"
```

**POST /patients** - Create new patient
```bash
curl -X POST "http://localhost:8000/patients" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {
      "name": "Jane Doe",
      "age": 8,
      "severity": "Severe",
      "mutation": "Missense",
      "dose": 1500
    },
    "notes": "New patient admission"
  }'
```

**PUT /patients/{id}** - Update patient
```bash
curl -X PUT "http://localhost:8000/patients/1" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_data": {...},
    "updated_by": 5
  }'
```

**DELETE /patients/{id}** - Delete patient
```bash
curl -X DELETE "http://localhost:8000/patients/1"
```

**GET /patients/{id}/history** - Get patient history
```bash
curl "http://localhost:8000/patients/1/history?limit=20"
```

---

### 4. Analytics API - `/analytics`

**GET /analytics/dashboard** - Get dashboard statistics
```bash
curl "http://localhost:8000/analytics/dashboard?days=30"
```

**Response:**
```json
{
  "dashboard_stats": {
    "total_patients": 150,
    "predictions_this_month": 45,
    "average_risk_score": 0.385,
    "high_risk_count": 23,
    "treatment_adherence_avg": 82.5,
    "inhibitor_rate": 0.18
  },
  "risk_distribution": {
    "low": 60,
    "medium": 55,
    "high": 30,
    "critical": 5
  },
  "predictions_trend": [],
  "top_factors": [
    {"factor": "Previous Inhibitor History", "weight": 0.15, "count": 27}
  ],
  "recommendations": [...]
}
```

**GET /analytics/risk-distribution** - Risk score distribution
```bash
curl "http://localhost:8000/analytics/risk-distribution"
```

**GET /analytics/severity-breakdown** - Patients by severity
```bash
curl "http://localhost:8000/analytics/severity-breakdown"
```

**GET /analytics/adherence-metrics** - Treatment adherence stats
```bash
curl "http://localhost:8000/analytics/adherence-metrics"
```

**GET /analytics/export** - Export analytics data
```bash
curl "http://localhost:8000/analytics/export?format=json"
```

---

### 5. Health Check - `/health`

**GET /health** - Check API status
```bash
curl "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-02T10:30:45.123456",
  "version": "1.0.0",
  "database": "connected",
  "ml_models": "loaded"
}
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:
```env
# OpenAI API
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=sqlite:///hemophilia_clinic.db

# FastAPI
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_ENV=development  # development or production

# CORS
ALLOW_ORIGINS=http://localhost:8501,http://localhost:3000
```

### CORS Configuration

Edit `backend/main.py` to allow specific origins:

```python
origins = [
    "http://localhost:8501",   # Streamlit
    "http://localhost:3000",   # React
    "http://localhost:8000",   # FastAPI
    "https://yourdomain.com",  # Production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📡 Frontend Integration (Streamlit)

The Streamlit app now uses the API:

```python
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"

# Make prediction
response = requests.post(
    f"{API_BASE_URL}/predict",
    json={
        "age": 12,
        "dose": 2000,
        "exposure": 90,
        "severity": "Moderate",
        "mutation": "Intron 22 Inversion"
    }
)
prediction = response.json()

# Chat with AI
response = requests.post(
    f"{API_BASE_URL}/chat",
    json={
        "question": "What's my risk?",
        "mode": "risk_explanation",
        "patient_data": patient_context
    }
)
chat_response = response.json()

# Get patients
response = requests.get(f"{API_BASE_URL}/patients?limit=20")
patients = response.json()

# Get analytics
response = requests.get(f"{API_BASE_URL}/analytics/dashboard?days=30")
analytics = response.json()
```

---

## 🧪 Testing the API

### Using cURL (Command Line)

```bash
# Health check
curl http://localhost:8000/health

# Get available chat modes
curl http://localhost:8000/chat/modes

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 12, "dose": 2000, "exposure": 90, "severity": "Moderate", "mutation": "Intron 22"}'

# Chat query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What should I monitor?", "mode": "monitoring_analysis"}'
```

### Using Swagger UI

1. Go to: `http://localhost:8000/docs`
2. Click "Try it out" on any endpoint
3. Enter parameters
4. Click "Execute"

### Using Python Requests

```python
import requests

# Prediction example
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 12,
        "dose": 2000,
        "exposure": 90,
        "severity": "Moderate",
        "mutation": "Intron 22 Inversion"
    }
)
print(response.json())

# Chat example
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "question": "Is this patient high risk?",
        "mode": "risk_explanation"
    }
)
print(response.json())
```

---

## 🔍 Troubleshooting

### Issue: "Connection refused" on localhost:8000

**Solution:** Ensure FastAPI is running
```bash
netstat -tuln | grep 8000  # Check if port is in use
python -m uvicorn backend.main:app --reload
```

### Issue: CORS Error in Streamlit

**Solution:** Update CORS origins in `backend/main.py`:
```python
origins = ["*"]  # For development
```

### Issue: Database connection error

**Solution:** Ensure database.py is initialized
```bash
python -c "from database import init_database; init_database()"
```

### Issue: OpenAI API key error

**Solution:** Set environment variable
```bash
export OPENAI_API_KEY=sk-...  # Linux/Mac
set OPENAI_API_KEY=sk-...     # Windows CMD
$env:OPENAI_API_KEY="sk-..."  # Windows PowerShell
```

---

## 📚 File Organization

```
project/
├── backend/                    # NEW: FastAPI backend
│   ├── __init__.py
│   ├── main.py               # Main FastAPI app
│   ├── models.py             # Pydantic models
│   ├── ml_utils.py           # ML logic
│   ├── gpt_utils.py          # Chat logic
│   ├── requirements.txt       # Dependencies
│   └── routers/              # Route handlers
│       ├── __init__.py
│       ├── predict.py
│       ├── chat.py
│       ├── patients.py
│       └── analytics.py
├── app.py                     # Streamlit frontend (updated)
├── database.py               # Database operations (shared)
├── evaluation.py             # ML evaluation (shared)
├── clinical_assistant.py     # Clinical AI (shared)
├── requirements.txt          # Frontend dependencies
└── [other files]
```

---

## 🚀 Deployment

### Local Development
```bash
# Terminal 1: FastAPI Backend
cd backend && python -m uvicorn main:app --reload

# Terminal 2: Streamlit Frontend
streamlit run app.py
```

### Production with Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app --bind 0.0.0.0:8000
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11

WORKDIR /app
COPY requirements.txt backend/requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📊 API Response Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad request (validation error) |
| 404 | Not found |
| 500 | Server error |

---

**Version:** 1.0.0  
**Last Updated:** 2026-04-02  
**Status:** Production-Ready ✅

