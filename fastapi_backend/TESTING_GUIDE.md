# FastAPI Backend Testing Guide

## 📋 Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Fixtures and configuration
├── test_config.py              # Configuration tests
├── test_exceptions.py          # Exception handling
├── test_models.py              # Pydantic model validation
├── test_services/
│   ├── __init__.py
│   ├── test_prediction_service.py
│   ├── test_chat_service.py
│   ├── test_patient_service.py
│   └── test_analytics_service.py
├── test_routers/
│   ├── __init__.py
│   ├── test_predictions_router.py
│   ├── test_chat_router.py
│   ├── test_patients_router.py
│   └── test_analytics_router.py
├── test_database.py            # Database layer
└── test_integration.py         # End-to-end tests
```

---

## 🔧 Setup & Configuration

### **requirements-dev.txt**
```
# Add to main requirements.txt
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.24.0
pytest-mock>=3.11.1
```

### **conftest.py** (Shared fixtures)

```python
import pytest
import sqlite3
import os
from fastapi.testclient import TestClient
from contextlib import contextmanager
import tempfile

# Import from your app
from main import app
from config import Settings
from database import DatabaseConnection
from services.prediction_service import PredictionService
from services.patient_service import PatientService
from services.chat_service import ChatService
from services.analytics_service import AnalyticsService

# ============ GLOBAL FIXTURES ============

@pytest.fixture(scope="session")
def test_settings():
    """Override settings for testing"""
    settings = Settings()
    settings.DATABASE_URL = "sqlite:///:memory:"
    return settings

@pytest.fixture(scope="function")
def test_db():
    """In-memory SQLite database for testing"""
    # Create temporary database
    db = DatabaseConnection(":memory:")
    db.init_tables()
    yield db
    # No cleanup needed (in-memory DB is auto-deleted)

@pytest.fixture(scope="function")
def client(test_db):
    """Test client with mocked database"""
    def override_get_db():
        return test_db
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def prediction_service(test_db):
    """PredictionService with test DB"""
    return PredictionService(test_db)

@pytest.fixture(scope="function")
def patient_service(test_db):
    """PatientService with test DB"""
    return PatientService(test_db)

@pytest.fixture(scope="function")
def chat_service():
    """ChatService (no DB dependency)"""
    return ChatService()

@pytest.fixture(scope="function")
def analytics_service(test_db):
    """AnalyticsService with test DB"""
    return AnalyticsService(test_db)

# ============ SAMPLE DATA ============

@pytest.fixture
def sample_patient_data():
    """Sample patient for testing"""
    return {
        "name": "John Doe",
        "age": 35,
        "gender": "M",
        "severity": "severe",
        "mutation": "intron22",
        "dose_intensity": 50.0,
        "exposure_days": 365,
        "fviii_inhibitor": False
    }

@pytest.fixture
def sample_prediction_data():
    """Sample prediction input"""
    return {
        "age": 35,
        "gender": "M",
        "severity": "severe",
        "mutation": "intron22",
        "dose_intensity": 50.0,
        "exposure_days": 365,
        "fviii_inhibitor": False
    }

@pytest.fixture
def sample_chat_message():
    """Sample chat message"""
    return {
        "patient_id": 1,
        "message": "What is the best treatment for my condition?"
    }

# ============ MOCKING HELPERS ============

@pytest.fixture
def mock_openai_response(monkeypatch):
    """Mock OpenAI API response"""
    def mock_create(*args, **kwargs):
        class MockResponse:
            choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': 'This is a mock response'})})]
        return MockResponse()
    
    monkeypatch.setattr("openai.ChatCompletion.create", mock_create)

@pytest.fixture
def mock_models(monkeypatch):
    """Mock ML model loading"""
    import pickle
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    rf_model = RandomForestClassifier(max_depth=3)
    rf_model.fit([[1, 2], [3, 4]], [0, 1])
    
    def mock_load(path, mmap_mode=None):
        if "rf" in path:
            return rf_model
        elif "xgb" in path:
            return rf_model  # Use RF as mock XGBoost
        return None
    
    monkeypatch.setattr("joblib.load", mock_load)
```

---

## ✅ Test Examples

### **1. Model Validation Tests** (`test_models.py`)

```python
import pytest
from pydantic import ValidationError
from models import (
    PatientCreate, PredictionInput, ChatMessage,
    SeverityLevel, MutationType
)

class TestPatientModel:
    """Test PatientCreate model validation"""
    
    def test_valid_patient_creation(self):
        """Should accept valid patient data"""
        patient = PatientCreate(
            name="John Doe",
            age=35,
            gender="M",
            severity="severe",
            mutation="intron22",
            dose_intensity=50.0,
            exposure_days=365,
            fviii_inhibitor=False
        )
        assert patient.name == "John Doe"
        assert patient.age == 35
    
    def test_invalid_age(self):
        """Should reject invalid age"""
        with pytest.raises(ValidationError) as exc_info:
            PatientCreate(
                name="John",
                age=-5,  # Invalid
                gender="M",
                severity="severe",
                mutation="intron22",
                dose_intensity=50.0,
                exposure_days=365,
                fviii_inhibitor=False
            )
        assert "age" in str(exc_info.value).lower()
    
    def test_invalid_dose_intensity(self):
        """Should reject negative dose"""
        with pytest.raises(ValidationError):
            PatientCreate(
                name="John",
                age=35,
                gender="M",
                severity="severe",
                mutation="intron22",
                dose_intensity=-10.0,  # Invalid
                exposure_days=365,
                fviii_inhibitor=False
            )
    
    def test_invalid_severity_enum(self):
        """Should reject invalid severity"""
        with pytest.raises(ValidationError):
            PatientCreate(
                name="John",
                age=35,
                gender="M",
                severity="invalid_severity",  # Invalid
                mutation="intron22",
                dose_intensity=50.0,
                exposure_days=365,
                fviii_inhibitor=False
            )

class TestPredictionModel:
    """Test PredictionInput validation"""
    
    def test_valid_prediction_input(self):
        """Should accept valid prediction"""
        pred = PredictionInput(
            age=35,
            gender="M",
            severity="severe",
            mutation="intron22",
            dose_intensity=50.0,
            exposure_days=365,
            fviii_inhibitor=False
        )
        assert pred.age == 35
    
    def test_missing_required_field(self):
        """Should reject missing required fields"""
        with pytest.raises(ValidationError):
            PredictionInput(
                age=35
                # Missing all other required fields
            )
```

### **2. Service Tests** (`test_services/test_patient_service.py`)

```python
import pytest
from databases.exceptions import DatabaseException
from services.patient_service import PatientService
from models import PatientCreate, PatientResponse

class TestPatientService:
    """Test PatientService CRUD operations"""
    
    def test_create_patient_success(self, patient_service, sample_patient_data):
        """Should create patient successfully"""
        created = patient_service.create_patient(
            PatientCreate(**sample_patient_data)
        )
        
        assert created.id is not None
        assert created.name == sample_patient_data["name"]
        assert created.age == sample_patient_data["age"]
    
    def test_get_patient_success(self, patient_service, sample_patient_data):
        """Should retrieve patient"""
        # Create first
        created = patient_service.create_patient(
            PatientCreate(**sample_patient_data)
        )
        
        # Retrieve
        retrieved = patient_service.get_patient(created.id)
        
        assert retrieved.id == created.id
        assert retrieved.name == created.name
    
    def test_get_patient_not_found(self, patient_service):
        """Should raise exception for non-existent patient"""
        with pytest.raises(PatientNotFound):
            patient_service.get_patient(999)
    
    def test_update_patient(self, patient_service, sample_patient_data):
        """Should update patient fields"""
        # Create
        created = patient_service.create_patient(
            PatientCreate(**sample_patient_data)
        )
        
        # Update
        update_data = {"age": 40, "dose_intensity": 75.0}
        updated = patient_service.update_patient(created.id, update_data)
        
        assert updated.age == 40
        assert updated.dose_intensity == 75.0
    
    def test_delete_patient(self, patient_service, sample_patient_data):
        """Should delete patient"""
        # Create
        created = patient_service.create_patient(
            PatientCreate(**sample_patient_data)
        )
        
        # Delete
        patient_service.delete_patient(created.id)
        
        # Verify deleted
        with pytest.raises(PatientNotFound):
            patient_service.get_patient(created.id)
    
    def test_search_by_severity(self, patient_service, sample_patient_data):
        """Should search patients by severity"""
        # Create multiple patients
        patient_service.create_patient(PatientCreate(**sample_patient_data))
        
        data2 = sample_patient_data.copy()
        data2["severity"] = "moderate"
        patient_service.create_patient(PatientCreate(**data2))
        
        # Search
        severe_patients = patient_service.search_patients("severe")
        
        assert len(severe_patients) > 0
        assert all(p.severity == "severe" for p in severe_patients)
    
    def test_get_all_patients(self, patient_service, sample_patient_data):
        """Should retrieve all patients"""
        # Create multiple
        patient_service.create_patient(PatientCreate(**sample_patient_data))
        
        data2 = sample_patient_data.copy()
        data2["name"] = "Jane Doe"
        patient_service.create_patient(PatientCreate(**data2))
        
        # Get all
        all_patients = patient_service.get_all_patients()
        
        assert len(all_patients) >= 2
```

### **3. Router Tests** (`test_routers/test_patients_router.py`)

```python
import pytest
from fastapi import status

class TestPatientsRouter:
    """Test patient endpoints"""
    
    def test_create_patient_endpoint(self, client, sample_patient_data):
        """POST /patients should create patient"""
        response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] is not None
        assert data["name"] == sample_patient_data["name"]
    
    def test_create_patient_validation_error(self, client):
        """POST /patients should validate input"""
        response = client.post(
            "/api/v1/patients",
            json={
                "name": "John",
                "age": -5,  # Invalid
                # Missing other required fields
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_all_patients(self, client, sample_patient_data):
        """GET /patients should list patients"""
        # Create first
        client.post("/api/v1/patients", json=sample_patient_data)
        
        # Get all
        response = client.get("/api/v1/patients")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
    
    def test_get_patient_by_id(self, client, sample_patient_data):
        """GET /patients/{id} should get patient"""
        # Create
        create_response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        patient_id = create_response.json()["id"]
        
        # Get
        response = client.get(f"/api/v1/patients/{patient_id}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == patient_id
    
    def test_get_patient_not_found(self, client):
        """GET /patients/{id} should return 404"""
        response = client.get("/api/v1/patients/999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_update_patient(self, client, sample_patient_data):
        """PUT /patients/{id} should update patient"""
        # Create
        create_response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        patient_id = create_response.json()["id"]
        
        # Update
        response = client.put(
            f"/api/v1/patients/{patient_id}",
            json={"age": 40, "dose_intensity": 75.0}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["age"] == 40
    
    def test_delete_patient(self, client, sample_patient_data):
        """DELETE /patients/{id} should delete patient"""
        # Create
        create_response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        patient_id = create_response.json()["id"]
        
        # Delete
        response = client.delete(f"/api/v1/patients/{patient_id}")
        
        assert response.status_code == status.HTTP_204_NO_CONTENT
        
        # Verify deleted
        get_response = client.get(f"/api/v1/patients/{patient_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_search_by_severity(self, client, sample_patient_data):
        """GET /patients/search/by-severity should filter"""
        # Create
        client.post("/api/v1/patients", json=sample_patient_data)
        
        # Search
        response = client.get(
            "/api/v1/patients/search/by-severity?severity=severe"
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) > 0
        assert all(p["severity"] == "severe" for p in data)
```

### **4. Integration Tests** (`test_integration.py`)

```python
import pytest
from fastapi import status

class TestIntegrationFlows:
    """Test end-to-end workflows"""
    
    def test_patient_creation_to_prediction_flow(self, client, sample_patient_data, sample_prediction_data):
        """Complete flow: Create patient → Make prediction"""
        
        # Step 1: Create patient
        patient_response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        assert patient_response.status_code == status.HTTP_201_CREATED
        patient_id = patient_response.json()["id"]
        
        # Step 2: Make prediction
        pred_response = client.post(
            f"/api/v1/predictions?patient_id={patient_id}",
            json=sample_prediction_data
        )
        assert pred_response.status_code == status.HTTP_200_OK
        pred_data = pred_response.json()
        
        # Verify prediction has required fields
        assert "risk_score" in pred_data
        assert "severity_category" in pred_data
        assert "explanation" in pred_data
    
    def test_patient_to_analytics_flow(self, client, sample_patient_data):
        """Complete flow: Create patients → Check analytics"""
        
        # Create multiple patients
        for i in range(3):
            data = sample_patient_data.copy()
            data["name"] = f"Patient {i}"
            response = client.post("/api/v1/patients", json=data)
            assert response.status_code == status.HTTP_201_CREATED
        
        # Check analytics
        analytics_response = client.get("/api/v1/analytics/dashboard")
        assert analytics_response.status_code == status.HTTP_200_OK
        
        analytics_data = analytics_response.json()
        assert analytics_data["total_patients"] >= 3
    
    def test_patient_update_persistence(self, client, sample_patient_data):
        """Verify patient updates persist in retrieval"""
        
        # Create
        create_response = client.post(
            "/api/v1/patients",
            json=sample_patient_data
        )
        patient_id = create_response.json()["id"]
        
        # Update
        new_age = 50
        update_response = client.put(
            f"/api/v1/patients/{patient_id}",
            json={"age": new_age}
        )
        assert update_response.status_code == status.HTTP_200_OK
        
        # Retrieve and verify
        get_response = client.get(f"/api/v1/patients/{patient_id}")
        assert get_response.json()["age"] == new_age
```

---

## 🏃 Running Tests

### **Run all tests**
```bash
pytest
```

### **Run specific test file**
```bash
pytest tests/test_models.py
```

### **Run specific test class**
```bash
pytest tests/test_routers/test_patients_router.py::TestPatientsRouter
```

### **Run specific test function**
```bash
pytest tests/test_routers/test_patients_router.py::TestPatientsRouter::test_create_patient_endpoint
```

### **Run with coverage**
```bash
pytest --cov=. --cov-report=html
```

### **Run in verbose mode**
```bash
pytest -v
```

### **Run with markers**
```bash
pytest -m "slow" -v
```

### **Stop on first failure**
```bash
pytest -x
```

---

## 📊 Coverage Report

After running `pytest --cov=. --cov-report=html`, open `htmlcov/index.html` to see coverage:

- **Target**: 80%+ overall coverage
- **Critical**: 100% coverage for services
- **Routers**: 85%+ coverage (edge cases matter)

---

## 🎯 Test Coverage Checklist

```
✅ Models (Pydantic validation)
  ├─ Valid input acceptance
  ├─ Invalid input rejection
  └─ Type coercion

✅ Services (Business logic)
  ├─ CRUD operations
  ├─ Exception handling
  ├─ Edge cases
  └─ Data transformation

✅ Routers (HTTP endpoints)
  ├─ Success responses
  ├─ Error responses
  ├─ HTTP status codes
  ├─ Input validation
  └─ Authorization (if applicable)

✅ Database (Data persistence)
  ├─ Create/Read/Update/Delete
  ├─ Transactions
  ├─ Constraints
  └─ Edge cases

✅ Integration (End-to-end flows)
  ├─ Multi-step scenarios
  ├─ Data persistence
  ├─ Cross-service communication
  └─ Error propagation
```

---

## 🐛 Common Issues & Solutions

### **Issue: Tests pass locally but fail in CI**
- **Solution**: Ensure test DB is clean and independent
- **Code**: Use `function` scope for fixtures, not `session`

### **Issue: Async test warnings**
- **Solution**: Use `pytest-asyncio` with proper markers
- **Code**: 
```python
@pytest.mark.asyncio
async def test_async_function():
    ...
```

### **Issue: OpenAI API calls in tests**
- **Solution**: Always mock external APIs
- **Code**: Use `monkeypatch` fixture to replace API calls

### **Issue: Database locked**
- **Solution**: Use in-memory SQLite for tests
- **Code**: `DatabaseConnection(":memory:")`

---

## ✨ Summary

This testing guide provides:

✅ Complete test structure  
✅ Reusable fixtures  
✅ Sample test implementations  
✅ Best practices  
✅ Coverage strategy  
✅ Troubleshooting guide  

**Goal**: Achieve 80%+ test coverage with meaningful, maintainable tests!
