# FastAPI Backend Architecture - Complete Overview

## System Architecture

### Before (Monolithic)
```
┌─────────────────────────────────────┐
│      Streamlit Frontend              │
│  ┌──────────────────────────────┐   │
│  │  ML Predictions              │   │
│  │  GPT/Chat Logic              │   │
│  │  Patient Management          │   │
│  │  Analytics/Dashboard         │   │
│  │  Database Connections        │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### After (Microservices with API)
```
┌──────────────────────────────────────────────────────────────┐
│                 Streamlit Frontend (Port 8501)                 │
│  • UI Components                                               │
│  • Calls FastAPI backend                                       │
│  • Browser HTTP requests to backend                            │
└────────────┬─────────────────────────────────────────────────┘
             │ HTTP requests
             ↓
┌────────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  /predict    → ML Prediction Logic                       │  │
│  │  /chat       → Clinical AI Chat (GPT)                    │  │
│  │  /patients   → Patient CRUD Operations                   │  │
│  │  /analytics  → Dashboard Analytics & Stats              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  ML Models   │  Database   │  GPT API   │  Encryption    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
             │ Read/Write
             ↓
┌────────────────────────────────────────────────────────────────┐
│                    SQLite Database                              │
└────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Component Breakdown

### 1. Frontend Layer (Streamlit)

**File:** `app.py`

**Responsibilities:**
- User interface rendering
- Form input collection
- Data visualization
- Chart/dashboard display
- Session state management

**Changes Made:**
- ✅ Replaced direct function calls with HTTP requests
- ✅ Points to `http://localhost:8000` for API
- ✅ Maintains session state for user context
- ✅ Handles API responses and errors gracefully

**Example Code:**
```python
import requests

API_URL = "http://localhost:8000"

# Make prediction via API
response = requests.post(
    f"{API_URL}/predict",
    json={
        "age": age,
        "dose": dose,
        "exposure": exposure,
        "severity": severity,
        "mutation": mutation
    }
)
result = response.json()
```

---

### 2. API Layer (FastAPI)

**File:** `backend/main.py`

**Responsibilities:**
- Route requests to appropriate handlers
- Request validation (Pydantic)
- Response formatting
- Error handling
- Middleware & CORS

**Key Features:**
✅ Auto-generated OpenAPI docs (`/docs`)
✅ CORS enabled for Streamlit & React
✅ Request logging
✅ Exception handling
✅ Health check endpoint

**Health Check:**
```
GET /health → Status of API, database, ML models
```

---

### 3. Business Logic Layer

#### 3a. ML Prediction Module

**File:** `backend/ml_utils.py`

**Functions:**
```python
def predict_inhibitor_risk() → Dict[risk_score, category, factors]
def calculate_clinical_adjustment() → Float (risk adjustment)
def generate_recommendations() → List[str]
def get_feature_importance() → Dict[feature, weight]
```

**Integrated from App:** `app.py` predict_inhibitor_risk() function

**Changes:**
- ✅ Extracted from Streamlit app
- ✅ Made into reusable functions
- ✅ No Streamlit dependencies
- ✅ Returns structured data (dicts, not st.displays)

---

#### 3b. Clinical AI Chat Module

**File:** `backend/gpt_utils.py`

**Classes & Functions:**
```python
class ClinicalAssistantMode  # Mode constants
class StructuredPromptTemplates  # 4 prompt templates
def get_clinical_response() → Tuple[response, mode]
def get_available_modes() → List[modes]
def get_medical_definitions() → Dict[term, definition]
```

**Integrated from Files:**
- ✅ `clinical_assistant.py` - Extracted fully
- ✅ `gpt_chatbot.py` - create_gpt_response() extracted
- ✅ All prompt templates kept intact
- ✅ Safety disclaimers preserved

---

#### 3c. Database Module

**File:** `database.py` (shared, not moved)

**Functions:**
```python
def init_database() → None
def add_patient() → patient_id
def get_patient() → patient_dict
def get_all_patients() → list[patients]
def update_patient() → None
def delete_patient() → None
```

**Status:**
- ✅ Remains as shared module
- ✅ Used by both Streamlit and FastAPI
- ✅ No changes needed

---

### 4. Router Handlers

#### 4a. Prediction Router

**File:** `backend/routers/predict.py`

**Endpoints:**
```
POST   /predict              → Single prediction
GET    /predict/batch        → Batch predictions
GET    /predict/history/{id} → Patient history
```

**Request Model:**
```python
class PredictionRequest:
    age, dose, exposure, severity, mutation
    [12 optional clinical parameters]
```

**Response Model:**
```python
class PredictionResponse:
    risk_score, risk_category, confidence
    contributing_factors, recommendations
    model_used
```

#### 4b. Chat Router

**File:** `backend/routers/chat.py`

**Endpoints:**
```
POST   /chat                 → Query AI
GET    /chat/modes           → List modes
GET    /chat/definitions     → Medical terms
POST   /chat/feedback        → Submit feedback
```

**Request Model:**
```python
class ChatRequest:
    question, mode, patient_data, conversation_history
```

**Response Model:**
```python
class ChatResponse:
    response, mode_used, disclaimer, sources, confidence
```

#### 4c. Patients Router

**File:** `backend/routers/patients.py`

**Endpoints:**
```
GET    /patients             → List all
POST   /patients             → Create new
GET    /patients/{id}        → Get one
PUT    /patients/{id}        → Update
DELETE /patients/{id}        → Delete
GET    /patients/{id}/history → History
```

#### 4d. Analytics Router

**File:** `backend/routers/analytics.py`

**Endpoints:**
```
GET    /analytics/dashboard           → Dashboard stats
GET    /analytics/risk-distribution   → Risk breakdown
GET    /analytics/severity-breakdown  → Severity stats
GET    /analytics/adherence-metrics   → Adherence stats
GET    /analytics/export              → Data export
```

---

### 5. Data Models Layer

**File:** `backend/models.py`

**Pydantic Models:**
- `PredictionRequest` / `PredictionResponse`
- `ChatRequest` / `ChatResponse`
- `PatientData` / `PatientResponse`
- `AnalyticsRequest` / `AnalyticsResponse`
- `ErrorResponse`
- `HealthResponse`

**Benefits:**
✅ Automatic validation
✅ Type hints
✅ JSON schema generation
✅ Auto-generated OpenAPI docs

---

## 📡 Request/Response Flow

### Example: ML Prediction Flow

```
Streamlit Frontend
    ↓ (HTTP POST /predict)
    ├─→ JSON: {age: 12, dose: 2000, ...}
    
FastAPI Backend (predict router)
    ↓ (Validate with PredictionRequest model)
    ├─→ Parse input, check types
    
ML Utils Module
    ├─→ Load trained models (rf.pkl, xgb.pkl)
    ├─→ Create feature vector
    ├─→ Get RF prediction: 0.45
    ├─→ Get XGBoost prediction: 0.48
    ├─→ Average: 0.465
    ├─→ Apply clinical adjustments
    ├─→ Generate recommendations
    
Prediction Router
    ↓ (Format with PredictionResponse model)
    ├─→ risk_score: 0.42
    ├─→ risk_category: "Medium"
    ├─→ recommendations: [...]
    
FastAPI Backend
    ↓ (HTTP 200 + JSON)

Streamlit Frontend
    ├─→ Display results
    ├─→ Show risk category badge
    ├─→ List recommendations
```

### Example: Chat Flow

```
Streamlit Frontend
    ↓ (HTTP POST /chat)
    ├─→ JSON: {
    │     question: "What should I monitor?",
    │     mode: "monitoring_analysis",
    │     patient_data: {...}
    │   }

FastAPI Backend (chat router)
    ↓ (Validate with ChatRequest model)
    
GPT Utils Module
    ├─→ Select prompt template based on mode
    ├─→ Format patient context: Name, Severity, Risk, ...
    ├─→ Build messages for OpenAI API
    │   ├─→ System: Mode-specific prompt template
    │   ├─→ History: Last 5 conversation messages
    │   └─→ User: Current question
    ├─→ Call OpenAI: POST https://api.openai.com/...
    │   ├─→ Model: gpt-4
    │   ├─→ Temperature: 0.7
    │   └─→ Max tokens: 1000
    ├─→ Parse response
    ├─→ Add safety disclaimer
    
Chat Router
    ↓ (Format with ChatResponse model)
    ├─→ response: "AI text here..."
    ├─→ mode_used: "monitoring_analysis"
    ├─→ disclaimer: "⚠️ For education only..."
    
FastAPI Backend
    ↓ (HTTP 200 + JSON)

Streamlit Frontend
    ├─→ Display response
    ├─→ Show disclaimer
    ├─→ Add feedback buttons
```

---

## 🔒 Security Considerations

### CORS (Cross-Origin Resource Sharing)

**Configured for:**
```python
"http://localhost:8501",      # Streamlit
"http://localhost:3000",      # React frontend
"http://localhost:8000",      # Local testing
```

**Production:** Add your domain
```python
"https://yourdomain.com",
"https://app.yourdomain.com"
```

### Error Handling

**Graceful Errors:**
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }
```

### Environment Variables

```env
OPENAI_API_KEY=sk-...           # OpenAI API key
DATABASE_URL=sqlite:///...      # Database location
FASTAPI_ENV=development         # development/production
ALLOW_ORIGINS=...               # CORS origins
```

---

## 📊 Project Structure After Refactor

```
project_root/
├── backend/                         ← NEW: FastAPI Backend
│   ├── __init__.py
│   ├── main.py                     ← FastAPI app entry point
│   ├── models.py                   ← Pydantic models (200 lines)
│   ├── ml_utils.py                 ← ML logic (300 lines)
│   ├── gpt_utils.py                ← Chat logic (400 lines)
│   ├── requirements.txt            ← Backend dependencies
│   └── routers/                    ← Route handlers
│       ├── __init__.py
│       ├── predict.py              ← /predict endpoints
│       ├── chat.py                 ← /chat endpoints
│       ├── patients.py             ← /patients endpoints
│       └── analytics.py            ← /analytics endpoints
│
├── app.py                          ← UPDATED: Streamlit frontend
├── database.py                     ← Shared database module
├── clinical_assistant.py           ← Kept for reference
├── evaluation.py                   ← ML evaluation (unchanged)
├── requirements.txt                ← Frontend dependencies
│
├── FASTAPI_BACKEND_GUIDE.md       ← Full API documentation
├── FASTAPI_BACKEND_QUICKSTART.md  ← 5-min start guide
├── start_all.py                    ← Start both services
└── [other files...]
```

---

## 🚀 Deployment Scenarios

### Local Development
```bash
Terminal 1: python -m uvicorn backend.main:app --reload
Terminal 2: streamlit run app.py
```

### Docker Containerization
```dockerfile
# Backend container
docker build -f Dockerfile.backend -t hemophilia-backend .
docker run -p 8000:8000 hemophilia-backend

# Frontend container
docker build -f Dockerfile.frontend -t hemophilia-frontend .
docker run -p 8501:8501 hemophilia-frontend
```

### Cloud Deployment (AWS/Heroku/GCP)
```
Backend:   Deploy to Heroku/AWS Lambda/Cloud Run
Frontend:  Deploy to Vercel/Netlify (if React)
Database:  MongoDB Atlas / AWS RDS / Cloud SQL
```

---

## ✨ Benefits of This Architecture

### 1. **Separation of Concerns**
- ✅ Frontend (Streamlit) handles UI only
- ✅ Backend (FastAPI) handles business logic
- ✅ Easy to maintain and modify independently

### 2. **Scalability**
- ✅ Backend can be load-balanced with gunicorn/nginx
- ✅ Multiple frontend instances can use same backend
- ✅ Easier to scale compute-heavy predictions

### 3. **Reliability**
- ✅ Backend failure doesn't crash frontend
- ✅ API requests are atomic
- ✅ Health check endpoint monitors status

### 4. **Integration**
- ✅ Easy to add new frontend (React, Vue, etc.)
- ✅ Backend can serve mobile apps
- ✅ Third-party systems can integrate via API

### 5. **Testing**
- ✅ Backend APIs are easy to test
- ✅ Can mock backend for frontend testing
- ✅ Unit tests for business logic

### 6. **Documentation**
- ✅ Auto-generated API docs (Swagger UI)
- ✅ Clear request/response models
- ✅ Type hints throughout codebase

---

## 📈 Performance Optimization

### Caching
```python
# Cache ML models in memory
@cache
def load_models():
    return rf_model, xgb_model
```

### Async Operations
```python
# All route handlers are async
@router.post("")
async def predict_risk(request: PredictionRequest):
    ...
```

### Database Connection Pooling
```python
# SQLite uses built-in connection manager
# For production, use SQLAlchemy with connection pooling
```

---

## 🔄 Backward Compatibility

**Existing Functions Preserved:**
- ✅ `predict_inhibitor_risk()` - works same way
- ✅ `create_gpt_response()` - wrapped in API
- ✅ `database.py` functions - unchanged
- ✅ `clinical_assistant.py` - fully integrated

**Migration Path:**
1. Start FastAPI backend (separate process)
2. Update Streamlit app to use API
3. All existing functionality works identically
4. No breaking changes to data models

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `FASTAPI_BACKEND_GUIDE.md` | Complete API reference with examples |
| `FASTAPI_BACKEND_QUICKSTART.md` | 5-minute quick start |
| This file | Architecture overview |
| Swagger UI at `/docs` | Interactive API documentation |

---

## 🎯 Next Steps

1. **Start Backend:**
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```

2. **Verify API:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **Open Docs:**
   ```
   http://localhost:8000/docs
   ```

4. **Start Frontend:**
   ```bash
   streamlit run app.py
   ```

5. **Test Integration:**
   - Make prediction in Streamlit
   - Verify call shows in backend logs
   - Check response displays correctly

---

**Status:** ✅ Production Ready  
**Version:** 1.0.0  
**Last Updated:** 2026-04-02

