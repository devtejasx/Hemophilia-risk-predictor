# FastAPI Backend - API Client Guide

## 🔗 Complete Client Implementation

### **Installation**

```bash
pip install requests python-dotenv pydantic httpx
```

---

## 📝 Synchronous Client (Requests)

### **api_client.py**

```python
import requests
import json
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with Medical AI Platform API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize API client
        
        Args:
            base_url: API base URL (default: localhost)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments (params, json, etc.)
            
        Returns:
            Response JSON
            
        Raises:
            requests.RequestException: On network error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, 
                url, 
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.text else {}
        
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json()
                raise APIException(f"API Error: {error_detail['detail']}", e.response.status_code)
            except json.JSONDecodeError:
                raise APIException(str(e), e.response.status_code)
        
        except requests.RequestException as e:
            raise APIException(f"Request failed: {str(e)}")
    
    # ============ HEALTH & STATUS ============
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Health status dict
            
        Example:
            >>> client.health_check()
            {'status': 'healthy', 'timestamp': '2024-01-15T10:30:00'}
        """
        return self._request("GET", "/health")
    
    def ready_check(self) -> Dict[str, Any]:
        """Check if API is ready to serve requests"""
        return self._request("GET", "/ready")
    
    # ============ PATIENTS ============
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create new patient
        
        Args:
            patient_data: Patient information dict
            
        Returns:
            Created patient with ID
            
        Example:
            >>> client.create_patient({
            ...     "name": "John Doe",
            ...     "age": 35,
            ...     "gender": "M",
            ...     "severity": "severe",
            ...     "mutation": "intron22",
            ...     "dose_intensity": 50.0,
            ...     "exposure_days": 365,
            ...     "fviii_inhibitor": False
            ... })
        """
        return self._request("POST", "/api/v1/patients", json=patient_data)
    
    def get_patients(self, skip: int = 0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all patients with pagination
        
        Args:
            skip: Number of patients to skip
            limit: Maximum patients to return
            
        Returns:
            List of patients
            
        Example:
            >>> clients = client.get_patients(skip=0, limit=5)
        """
        return self._request(
            "GET", 
            "/api/v1/patients",
            params={"skip": skip, "limit": limit}
        )
    
    def get_patient(self, patient_id: int) -> Dict[str, Any]:
        """
        Get patient by ID
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Patient details
            
        Example:
            >>> patient = client.get_patient(1)
        """
        return self._request("GET", f"/api/v1/patients/{patient_id}")
    
    def update_patient(
        self, 
        patient_id: int, 
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update patient information
        
        Args:
            patient_id: Patient ID
            update_data: Fields to update
            
        Returns:
            Updated patient
            
        Example:
            >>> client.update_patient(1, {"age": 40, "dose_intensity": 75.0})
        """
        return self._request(
            "PUT", 
            f"/api/v1/patients/{patient_id}",
            json=update_data
        )
    
    def delete_patient(self, patient_id: int) -> None:
        """
        Delete patient
        
        Args:
            patient_id: Patient ID
            
        Example:
            >>> client.delete_patient(1)
        """
        self._request("DELETE", f"/api/v1/patients/{patient_id}")
    
    def search_patients_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """
        Search patients by severity level
        
        Args:
            severity: Severity level (severe, moderate, mild)
            
        Returns:
            List of matching patients
            
        Example:
            >>> severe_patients = client.search_patients_by_severity("severe")
        """
        return self._request(
            "GET",
            "/api/v1/patients/search/by-severity",
            params={"severity": severity}
        )
    
    def search_patients_by_mutation(self, mutation: str) -> List[Dict[str, Any]]:
        """
        Search patients by mutation type
        
        Args:
            mutation: Mutation type (intron22, intron1, etc)
            
        Returns:
            List of matching patients
        """
        return self._request(
            "GET",
            "/api/v1/patients/search/by-mutation",
            params={"mutation": mutation}
        )
    
    # ============ PREDICTIONS ============
    
    def predict(
        self, 
        prediction_data: Dict[str, Any],
        patient_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate ML prediction
        
        Args:
            prediction_data: Clinical data for prediction
            patient_id: Optional patient ID to link prediction
            
        Returns:
            Prediction with risk score and explanation
            
        Example:
            >>> pred = client.predict({
            ...     "age": 35,
            ...     "gender": "M",
            ...     "severity": "severe",
            ...     "mutation": "intron22",
            ...     "dose_intensity": 50.0,
            ...     "exposure_days": 365,
            ...     "fviii_inhibitor": False
            ... }, patient_id=1)
        """
        params = {}
        if patient_id:
            params["patient_id"] = patient_id
        
        return self._request(
            "POST",
            "/api/v1/predictions",
            json=prediction_data,
            params=params
        )
    
    def batch_predict(
        self, 
        predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple patients
        
        Args:
            predictions: List of prediction data dicts
            
        Returns:
            List of predictions
            
        Example:
            >>> batch_preds = client.batch_predict([
            ...     {"age": 35, "gender": "M", ...},
            ...     {"age": 45, "gender": "F", ...}
            ... ])
        """
        return self._request(
            "POST",
            "/api/v1/predictions/batch",
            json=predictions
        )
    
    def get_patient_prediction_history(
        self, 
        patient_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history for patient
        
        Args:
            patient_id: Patient ID
            
        Returns:
            List of patient's predictions
        """
        return self._request(
            "GET",
            f"/api/v1/predictions/patient/{patient_id}"
        )
    
    # ============ CHAT ============
    
    def send_message(
        self, 
        patient_id: int, 
        message: str
    ) -> Dict[str, Any]:
        """
        Send message to clinical chatbot
        
        Args:
            patient_id: Patient ID for context
            message: User message
            
        Returns:
            AI response
            
        Example:
            >>> response = client.send_message(
            ...     1, 
            ...     "What is the best treatment for my condition?"
            ... )
        """
        return self._request(
            "POST",
            "/api/v1/chat",
            json={
                "patient_id": patient_id,
                "message": message
            }
        )
    
    def get_chat_history(self, patient_id: int) -> List[Dict[str, Any]]:
        """
        Get conversation history for patient
        
        Args:
            patient_id: Patient ID
            
        Returns:
            List of chat messages
        """
        return self._request(
            "GET",
            f"/api/v1/chat/patient/{patient_id}"
        )
    
    def chat_health(self) -> Dict[str, Any]:
        """Check if chat service is available"""
        return self._request("GET", "/api/v1/chat/health")
    
    # ============ ANALYTICS ============
    
    def get_dashboard(self) -> Dict[str, Any]:
        """
        Get dashboard statistics
        
        Returns:
            Summary statistics
            
        Example:
            >>> dashboard = client.get_dashboard()
            >>> print(f"Total patients: {dashboard['total_patients']}")
        """
        return self._request("GET", "/api/v1/analytics/dashboard")
    
    def get_risk_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Get risk trends over time
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Risk trend data
        """
        return self._request(
            "GET",
            "/api/v1/analytics/trends",
            params={"days": days}
        )
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """Get statistics by mutation type"""
        return self._request("GET", "/api/v1/analytics/mutations")
    
    def get_severity_distribution(self) -> Dict[str, Any]:
        """Get patient distribution by severity"""
        return self._request("GET", "/api/v1/analytics/severity")
    
    def get_high_risk_patients(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get high-risk patients
        
        Args:
            limit: Number of patients to return
            
        Returns:
            List of high-risk patients ranked by risk
        """
        return self._request(
            "GET",
            "/api/v1/analytics/high-risk",
            params={"limit": limit}
        )


class APIException(Exception):
    """API client exception"""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)
```

---

## 🔄 Async Client (HTTPX)

### **async_api_client.py**

```python
import httpx
import json
from typing import Dict, List, Optional, Any
import asyncio

class AsyncAPIClient:
    """Asynchronous API client for high-performance operations"""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000", 
        timeout: int = 30
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make async HTTP request"""
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json() if response.text else {}
        except httpx.HTTPError as e:
            raise APIException(f"API Error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return await self._request("GET", "/health")
    
    async def batch_predict_async(
        self,
        predictions: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Async batch prediction with concurrency control
        
        Args:
            predictions: List of predictions
            batch_size: Process N predictions concurrently
            
        Returns:
            List of results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]
            
            # Concurrent requests
            tasks = [
                self._request("POST", "/api/v1/predictions", json=pred)
                for pred in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        return results
    
    async def close(self):
        """Close client connection"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
```

---

## 🚀 Usage Examples

### **Example 1: Complete Patient Workflow**

```python
from api_client import APIClient, APIException

def main():
    # Initialize client
    client = APIClient(base_url="http://localhost:8000")
    
    try:
        # 1. Check API health
        print("🏥 Checking API health...")
        health = client.health_check()
        print(f"Status: {health['status']}")
        
        # 2. Create patient
        print("\n➕ Creating patient...")
        patient_data = {
            "name": "John Doe",
            "age": 35,
            "gender": "M",
            "severity": "severe",
            "mutation": "intron22",
            "dose_intensity": 50.0,
            "exposure_days": 365,
            "fviii_inhibitor": False
        }
        patient = client.create_patient(patient_data)
        patient_id = patient["id"]
        print(f"✅ Patient created: {patient_id}")
        
        # 3. Get patient details
        print("\n👤 Retrieving patient...")
        retrieved = client.get_patient(patient_id)
        print(f"Name: {retrieved['name']}, Age: {retrieved['age']}")
        
        # 4. Update patient
        print("\n✏️ Updating patient...")
        updated = client.update_patient(
            patient_id,
            {"age": 40, "dose_intensity": 75.0}
        )
        print(f"Updated age: {updated['age']}")
        
        # 5. Generate prediction
        print("\n🧠 Making prediction...")
        prediction = client.predict(patient_data, patient_id)
        print(f"Risk Score: {prediction['risk_score']:.2%}")
        print(f"Category: {prediction['severity_category']}")
        print(f"Explanation: {prediction['explanation']}")
        
        # 6. Chat with AI
        print("\n💬 Sending message to chatbot...")
        chat_response = client.send_message(
            patient_id,
            "What preventive measures can I take?"
        )
        print(f"AI Response: {chat_response['response']}")
        
        # 7. Get analytics
        print("\n📊 Retrieving dashboard...")
        dashboard = client.get_dashboard()
        print(f"Total Patients: {dashboard['total_patients']}")
        print(f"Average Risk: {dashboard['average_risk_score']:.2%}")
        
        # 8. Get high-risk patients
        print("\n⚠️ High-risk patients...")
        high_risk = client.get_high_risk_patients(limit=5)
        for patient in high_risk:
            print(f"  - {patient['name']}: {patient['risk_score']:.2%}")
        
        # 9. Delete patient
        print("\n🗑️ Deleting patient...")
        client.delete_patient(patient_id)
        print("✅ Patient deleted")
        
    except APIException as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

### **Example 2: Batch Operations**

```python
from api_client import APIClient

def batch_example():
    client = APIClient()
    
    # Prepare multiple patients
    patients = [
        {
            "name": f"Patient {i}",
            "age": 30 + i,
            "gender": "M" if i % 2 == 0 else "F",
            "severity": ["mild", "moderate", "severe"][i % 3],
            "mutation": "intron22",
            "dose_intensity": 50.0 + (i * 10),
            "exposure_days": 365,
            "fviii_inhibitor": i % 2 == 0
        }
        for i in range(100)
    ]
    
    # Create patients
    print("Creating 100 patients...")
    created_ids = []
    for patient_data in patients:
        patient = client.create_patient(patient_data)
        created_ids.append(patient["id"])
    
    # Batch predictions
    print("Running batch predictions...")
    predictions = client.batch_predict(patients)
    
    # Analyze results
    avg_risk = sum(p["risk_score"] for p in predictions) / len(predictions)
    high_risk_count = sum(1 for p in predictions if p["risk_score"] > 0.7)
    
    print(f"Average Risk: {avg_risk:.2%}")
    print(f"High-Risk Patients (>70%): {high_risk_count}")
```

### **Example 3: Async Batch Processing**

```python
import asyncio
from async_api_client import AsyncAPIClient

async def async_batch_example():
    async with AsyncAPIClient() as client:
        # Prepare 500 predictions
        predictions = [
            {
                "age": 30 + i,
                "gender": "M" if i % 2 == 0 else "F",
                "severity": ["mild", "moderate", "severe"][i % 3],
                "mutation": "intron22",
                "dose_intensity": 50.0 + (i % 100),
                "exposure_days": 365,
                "fviii_inhibitor": i % 2 == 0
            }
            for i in range(500)
        ]
        
        # Process with 20 concurrent requests
        print("Processing 500 predictions concurrently...")
        results = await client.batch_predict_async(
            predictions,
            batch_size=20
        )
        
        print(f"✅ Processed {len(results)} predictions")

# Run
if __name__ == "__main__":
    asyncio.run(async_batch_example())
```

### **Example 4: Error Handling**

```python
from api_client import APIClient, APIException

def error_handling_example():
    client = APIClient()
    
    try:
        # Try to get non-existent patient
        patient = client.get_patient(99999)
    
    except APIException as e:
        print(f"Status Code: {e.status_code}")
        print(f"Error Message: {e.message}")
        
        if e.status_code == 404:
            print("Patient not found")
        elif e.status_code == 422:
            print("Validation error")
        elif e.status_code >= 500:
            print("Server error")
```

---

## 📦 Usage in Production

### **requirements.txt**

```
requests>=2.31.0
httpx>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
```

### **Configuration**

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
```

### **Integration with Streamlit**

```python
# streamlit_app.py
import streamlit as st
from api_client import APIClient, APIException

@st.cache_resource
def get_client():
    return APIClient(base_url="http://localhost:8000")

def main():
    st.title("Clinical Decision Support")
    
    client = get_client()
    
    # Patient creation form
    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=150)
        # ... other fields
        
        if st.form_submit_button("Create Patient"):
            try:
                patient = client.create_patient({
                    "name": name,
                    "age": age,
                    # ... other data
                })
                st.success(f"Patient created: {patient['id']}")
            except APIException as e:
                st.error(f"Error: {e.message}")
```

---

## ✅ Summary

✅ Synchronous and async clients  
✅ Complete CRUD operations  
✅ Batch processing  
✅ Error handling  
✅ Real-world examples  
✅ Production-ready code  

**Ready to integrate with frontend!**
