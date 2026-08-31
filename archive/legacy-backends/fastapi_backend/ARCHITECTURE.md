# FastAPI Backend Architecture Guide

## 🏗️ Clean Architecture Principles

This FastAPI backend implements **Clean Architecture** with clear separation of concerns.

---

## 📊 Layer Architecture

```
┌────────────────────────────────────────┐
│         API Layer (Routers)            │
│  predictions_router.py                 │
│  chat_router.py                        │
│  patients_router.py                    │
│  analytics_router.py                   │
└────────────────────────────────────────┘
         ↓         ↑
┌────────────────────────────────────────┐
│      Services Layer (Business Logic)   │
│  prediction_service.py                 │
│  chat_service.py                       │
│  patient_service.py                    │
│  analytics_service.py                  │
└────────────────────────────────────────┘
         ↓         ↑
┌────────────────────────────────────────┐
│  Data Layer (Database & External APIs) │
│  database/__init__.py                  │
│  External: OpenAI, ML Models           │
└────────────────────────────────────────┘
```

---

## 🔄 Request Flow Example

### **Prediction Flow**

```
HTTP POST /api/v1/predictions
    ↓
predictions_router.create_prediction()
    ↓
PredictionService.predict()
    ├─ Load models from pkl files
    ├─ Prepare features
    ├─ Run RF + XGBoost ensemble
    ├─ Calculate SHAP importance
    └─ Build response
    ↓
Save to database (Optional)
    ↓
Return PredictionOutput (JSON)
```

### **Chat Flow**

```
HTTP POST /api/v1/chat
    ↓
chat_router.send_message()
    ↓
ChatService.get_clinical_response()
    ├─ Build patient context
    ├─ Call OpenAI API
    ├─ Parse response
    └─ Return response
    ↓
Save to conversations table
    ↓
Return ChatResponse (JSON)
```

### **Patient CRUD Flow**

```
HTTP POST /api/v1/patients
    ↓
patients_router.create_patient()
    ↓
PatientService.create_patient()
    ├─ Validate data (Pydantic)
    ├─ Build SQL query
    ├─ Execute insert
    ├─ Get patient ID
    └─ Fetch created patient
    ↓
Return PatientResponse (JSON)
```

---

## 🎯 Design Patterns Used

### **1. Dependency Injection**

```python
# Services depend on database, not on specific implementation
@router.post("/")
async def create_prediction(
    prediction_input: PredictionInput,
    db = Depends(get_db)  # ← Injected dependency
):
    service = PredictionService(db)
    return service.predict(prediction_input)
```

**Benefits**:
- Easy to test (can mock database)
- Loose coupling
- Configuration flexibility

### **2. Service Layer Pattern**

```python
# Routers handle HTTP
# Services handle business logic

# Router (thin layer)
@router.get("/{patient_id}")
async def get_patient(patient_id: int, db = Depends(get_db)):
    return PatientService(db).get_patient(patient_id)

# Service (handles logic)
class PatientService:
    def get_patient(self, patient_id: int):
        # Fetch from DB
        # Transform data
        # Return response
```

**Benefits**:
- Testable business logic
- Reusable across endpoints
- Clear responsibilities

### **3. Pydantic Models**

```python
# Request validation (automatic)
class PredictionInput(BaseModel):
    age: int = Field(..., ge=0, le=150)
    severity: SeverityLevel
    # Validators applied automatically

# Response serialization (automatic)
class PredictionOutput(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
```

**Benefits**:
- Type safety
- Automatic validation
- Auto-documentation
- Serializer/deserializer

### **4. Exception Handling**

```python
# Custom exceptions
class DatabaseException(MedicalAIException):
    def __init__(self, message: str = "Database error"):
        super().__init__(message, 500, "DATABASE_ERROR")

# Usage
try:
    cursor.execute(query)
except sqlite3.Error as e:
    raise DatabaseException(f"Query failed: {e}")

# Handling
@app.exception_handler(DatabaseException)
async def handle_db_error(request, exc):
    return error_response(exc)
```

**Benefits**:
- Consistent error format
- Proper HTTP status codes
- Easy debugging

### **5. Configuration Management**

```python
# config.py - Single source of truth
class Settings:
    OpenAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    RF_MODEL_PATH: str = os.getenv("RF_MODEL_PATH")

# Usage everywhere
from config import settings
client = OpenAI(api_key=settings.OPENAI_API_KEY)
```

**Benefits**:
- Environment-based configuration
- No hardcoded values
- Easy deployment

---

## 📦 Module Responsibilities

### **config.py**
- Application settings
- Environment variables
- Constants

### **exceptions.py**
- Custom exception classes
- HTTP exception mapping
- Error formatting

### **models/__init__.py** (Pydantic Schemas)
- Request models (PatientCreate, PredictionInput)
- Response models (PatientResponse, PredictionOutput)
- Shared enums (SeverityLevel, MutationType)
- Data validation

### **database/__init__.py**
- Database connection management
- SQLite context managers
- Query execution helpers
- Table initialization

### **services/**
Each service handles one domain:

- **prediction_service.py**
  - Load ML models
  - Generate predictions
  - Feature importance calculation
  - Batch predictions

- **chat_service.py**
  - OpenAI API integration
  - Prompt building
  - Response parsing
  - Context injection

- **patient_service.py**
  - CRUD operations
  - Data transformation
  - Search functionality
  - Risk score updates

- **analytics_service.py**
  - Statistical calculations
  - Trend analysis
  - Distribution statistics
  - Data aggregation

### **routers/**
Each router handles HTTP for one domain:

- **predictions_router.py**
  - POST /predictions (single)
  - POST /predictions/batch
  - GET /predictions/patient/{id}

- **chat_router.py**
  - POST /chat (send message)
  - GET /chat/patient/{id} (history)
  - GET /chat/health (service status)

- **patients_router.py**
  - POST / (create)
  - GET / (list)
  - GET /{id} (read)
  - PUT /{id} (update)
  - DELETE /{id} (delete)
  - Search endpoints

- **analytics_router.py**
  - GET /dashboard (summary)
  - GET /trends (time series)
  - GET /mutations (by mutation)
  - GET /severity (by severity)
  - GET /high-risk (top patients)

### **main.py**
- FastAPI app creation
- Router registration
- Middleware setup
- Lifespan events
- Global exception handlers
- Health check endpoints

---

## 🔄 Data Flow Patterns

### **Pattern 1: Stateless Service with Dependency**

```python
# Router receives dependency
@router.post("/")
async def endpoint(data: InputModel, db = Depends(get_db)):
    # Service is created per request (stateless)
    service = SomeService(db)
    return service.do_something(data)

# Service uses dependency
class SomeService:
    def __init__(self, db):
        self.db = db  # Injected
    
    def do_something(self, data):
        # Use self.db
```

**When to use**: Most endpoints (prediction, chat, patients)

### **Pattern 2: Global Singleton Service**

```python
# Service loaded once
prediction_service = PredictionService()

# Router uses global
@router.post("/")
async def endpoint(data: InputModel):
    return prediction_service.predict(data)
```

**When to use**: Expensive initialization (ML models, OpenAI client)

### **Pattern 3: Dependency-Injected Service**

```python
def get_patient_service(db = Depends(get_db)) -> PatientService:
    return PatientService(db)

@router.post("/")
async def create(data: InputModel, service: PatientService = Depends(get_patient_service)):
    return service.create(data)
```

**When to use**: Services needing dependencies

---

## ✅ Best Practices Implemented

### **1. Validation**

```python
# Pydantic validates automatically
class PredictionInput(BaseModel):
    age: int = Field(..., ge=0, le=150)  # Validators
    severity: SeverityLevel  # Enum validation
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
```

### **2. Error Handling**

```python
# Specific exceptions
try:
    patient = db.get(id)
except PatientNotFound:
    raise HTTPException(404, "Patient not found")
except DatabaseException:
    raise HTTPException(500, "Database error")

# Global handlers catch all
@app.exception_handler(Exception)
async def handle_error(request, exc):
    return error_response(exc)
```

### **3. Async/Await**

```python
# All endpoints are async
@router.post("/")
async def endpoint(data: InputModel):
    # Non-blocking I/O
    result = await async_operation()
    return result
```

### **4. Type Hints**

```python
# All functions have type hints
def predict(
    data: PredictionInput,
    db: sqlite3.Connection
) -> PredictionOutput:
    ...
```

### **5. Documentation**

```python
# Docstrings on all functions
def get_patient(self, patient_id: int) -> PatientResponse:
    """
    Get patient by ID
    
    Args:
        patient_id: Patient ID
        
    Returns:
        PatientResponse with all patient data
        
    Raises:
        PatientNotFound: If patient doesn't exist
    """
    ...

# Docstrings on all endpoints
@router.post("/")
async def create_prediction(prediction_input: PredictionInput):
    """
    Generate ML prediction for patient
    
    Args:
        prediction_input: Patient clinical data
        
    Returns:
        Prediction with risk score and explanations
    """
    ...
```

---

## 🧪 Testing Examples

### **Unit Test Example**

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_patient():
    response = client.post(
        "/api/v1/patients",
        json={
            "name": "Test",
            "age": 30,
            "gender": "M",
            "severity": "severe",
            "mutation": "intron22",
            "dose_intensity": 50.0,
            "exposure_days": 365
        }
    )
    assert response.status_code == 201
    assert response.json()["id"] is not None
```

### **Integration Test Example**

```python
def test_prediction_pipeline():
    # 1. Create patient
    patient_response = client.post("/api/v1/patients", json=patient_data)
    patient_id = patient_response.json()["id"]
    
    # 2. Get prediction
    pred_response = client.post(
        f"/api/v1/predictions?patient_id={patient_id}",
        json=prediction_data
    )
    assert pred_response.status_code == 200
    
    # 3. Verify prediction saved to DB
    analytics = client.get("/api/v1/analytics/dashboard")
    assert analytics.json()["total_patients"] > 0
```

---

## 🚀 Performance Considerations

### **1. Connection Pooling** (Future Enhancement)

```python
# Could add SQLAlchemy connection pooling
from sqlalchemy.pool import QueuePool

pool = QueuePool(
    sqlite3.connect,
    max_overflow=10,
    pool_size=5
)
```

### **2. Model Caching**

```python
# Models loaded once (singleton pattern)
prediction_service = PredictionService()  # Loads at startup

# Used for all requests (no reload)
@router.post("/")
async def endpoint(data: InputModel):
    return prediction_service.predict(data)
```

### **3. Batch Operations**

```python
# Batch endpoint for efficiency
@router.post("/batch")
async def batch_predictions(predictions: List[PredictionInput]):
    results = []
    for pred in predictions:
        results.append(prediction_service.predict(pred))
    return results
```

---

## 📚 Further Learning

- **FastAPI**: https://fastapi.tiangolo.com
- **Pydantic**: https://docs.pydantic.dev
- **SQLite**: https://www.sqlite.org
- **Design Patterns**: https://refactoring.guru/design-patterns
- **Clean Code**: Robert C. Martin's "Clean Code"

---

## 🎯 Summary

This FastAPI backend demonstrates:

✅ Clean separation of concerns (routers, services, models)  
✅ Type safety with Pydantic models  
✅ Dependency injection for testability  
✅ Comprehensive error handling  
✅ Configuration management  
✅ Async/await for performance  
✅ ML model integration  
✅ OpenAI API integration  
✅ Database operations  
✅ Analytics & statistics  

**Production-ready code structure** that scales with your application!
