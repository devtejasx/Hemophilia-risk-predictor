"""
FastAPI Client for Streamlit Frontend
Example implementation showing how to call backend APIs from Streamlit
"""

import requests
import streamlit as st
from typing import Dict, Optional, List, Any
from datetime import datetime

# ============= CONFIGURATION =============

API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30  # seconds

# Cache session for connection pooling
session = requests.Session()


# ============= ERROR HANDLING =============

class APIError(Exception):
    """Custom exception for API errors"""
    pass


def handle_api_error(response: requests.Response) -> None:
    """Handle API error responses"""
    try:
        error_data = response.json()
        error_msg = error_data.get("error", error_data.get("detail", "Unknown error"))
    except:
        error_msg = response.text
    
    raise APIError(f"API Error {response.status_code}: {error_msg}")


# ============= PREDICTION CLIENT =============

def predict_risk(
    age: int,
    dose: float,
    exposure: int,
    severity: str,
    mutation: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict inhibitor risk
    
    Args:
        age: Patient age
        dose: Treatment dose in units
        exposure: Days of treatment
        severity: Mild/Moderate/Severe
        mutation: Gene mutation name
        **kwargs: Additional clinical parameters
    
    Returns:
        Dict with risk_score, risk_category, recommendations
    """
    try:
        payload = {
            "age": age,
            "dose": dose,
            "exposure": exposure,
            "severity": severity,
            "mutation": mutation,
            **kwargs
        }
        
        response = session.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


# ============= CHAT CLIENT =============

def chat_query(
    question: str,
    mode: str = "diagnosis_support",
    patient_data: Optional[Dict] = None,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Query clinical AI assistant
    
    Args:
        question: User question
        mode: diagnosis_support, treatment_recommendation, risk_explanation, monitoring_analysis
        patient_data: Optional patient context
        conversation_history: Optional conversation history
    
    Returns:
        Dict with response, mode_used, disclaimer, confidence
    """
    try:
        payload = {
            "question": question,
            "mode": mode,
            "patient_data": patient_data,
            "conversation_history": conversation_history or []
        }
        
        response = session.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def get_chat_modes() -> List[Dict[str, str]]:
    """Get available chat modes"""
    try:
        response = session.get(
            f"{API_BASE_URL}/chat/modes",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        data = response.json()
        return data.get("modes", [])
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def get_definitions(terms: Optional[List[str]] = None) -> Dict[str, str]:
    """Get medical definitions"""
    try:
        params = {"terms": terms} if terms else {}
        
        response = session.get(
            f"{API_BASE_URL}/chat/definitions",
            params=params,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        data = response.json()
        return data.get("definitions", {})
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


# ============= PATIENT CLIENT =============

def get_patients(skip: int = 0, limit: int = 10, **filters) -> List[Dict]:
    """Get patient list"""
    try:
        params = {"skip": skip, "limit": limit}
        params.update(filters)
        
        response = session.get(
            f"{API_BASE_URL}/patients",
            params=params,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def get_patient(patient_id: int) -> Dict:
    """Get specific patient"""
    try:
        response = session.get(
            f"{API_BASE_URL}/patients/{patient_id}",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def create_patient(patient_data: Dict, notes: Optional[str] = None) -> Dict:
    """Create new patient"""
    try:
        payload = {
            "patient_data": patient_data,
            "notes": notes
        }
        
        response = session.post(
            f"{API_BASE_URL}/patients",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def update_patient(patient_id: int, patient_data: Dict) -> Dict:
    """Update patient"""
    try:
        payload = {"patient_data": patient_data}
        
        response = session.put(
            f"{API_BASE_URL}/patients/{patient_id}",
            json=payload,
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def delete_patient(patient_id: int) -> Dict:
    """Delete patient"""
    try:
        response = session.delete(
            f"{API_BASE_URL}/patients/{patient_id}",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


# ============= ANALYTICS CLIENT =============

def get_dashboard_analytics(days: int = 30) -> Dict:
    """Get dashboard analytics"""
    try:
        response = session.get(
            f"{API_BASE_URL}/analytics/dashboard",
            params={"days": days},
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def get_risk_distribution() -> Dict:
    """Get risk score distribution"""
    try:
        response = session.get(
            f"{API_BASE_URL}/analytics/risk-distribution",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


def get_severity_breakdown() -> Dict:
    """Get patient breakdown by severity"""
    try:
        response = session.get(
            f"{API_BASE_URL}/analytics/severity-breakdown",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


# ============= HEALTH & INFO =============

def health_check() -> bool:
    """Check if API is running"""
    try:
        response = session.get(
            f"{API_BASE_URL}/health",
            timeout=5
        )
        return response.status_code == 200
    except:
        return False


def get_api_info() -> Dict:
    """Get API information"""
    try:
        response = session.get(
            f"{API_BASE_URL}/info",
            timeout=API_TIMEOUT
        )
        
        if response.status_code != 200:
            handle_api_error(response)
        
        return response.json()
    
    except requests.RequestException as e:
        raise APIError(f"Connection error: {str(e)}")


# ============= STREAMLIT INTEGRATION HELPERS =============

def check_api_available() -> bool:
    """Check if API is available, show warning if not"""
    if not health_check():
        st.error(
            "⚠️ **API Backend Not Available**\n\n"
            "FastAPI backend is not running. Please start it with:\n"
            "```bash\ncd backend\n"
            "python -m uvicorn main:app --reload\n```\n\n"
            "Documentation: http://localhost:8000/docs"
        )
        return False
    return True


def display_api_status():
    """Display API status in sidebar"""
    with st.sidebar:
        if health_check():
            st.success("✅ API Backend: Connected")
        else:
            st.error("❌ API Backend: Disconnected")


def retry_on_error(max_retries: int = 3):
    """Decorator to retry API calls on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    if attempt == max_retries - 1:
                        raise
                    st.warning(f"Attempt {attempt + 1} failed, retrying...")
        return wrapper
    return decorator


# ============= EXAMPLE USAGE IN STREAMLIT =============

if __name__ == "__main__":
    st.title("FastAPI Client Example")
    
    # Check API availability
    if not check_api_available():
        st.stop()
    
    # Example: Make prediction
    st.subheader("Test Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=12)
        dose = st.number_input("Dose", min_value=0.0, value=2000.0)
    
    with col2:
        exposure = st.number_input("Exposure (days)", min_value=0, value=90)
        severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
    
    mutation = st.text_input("Mutation", value="Intron 22 Inversion")
    
    if st.button("Predict"):
        try:
            result = predict_risk(age, dose, exposure, severity, mutation)
            st.success("Prediction successful!")
            st.json(result)
        except APIError as e:
            st.error(f"Error: {str(e)}")
    
    # Example: Chat query
    st.divider()
    st.subheader("Test Chat")
    
    question = st.text_area("Question", value="What should I monitor?")
    mode = st.selectbox("Mode", ["diagnosis_support", "treatment_recommendation", "risk_explanation", "monitoring_analysis"])
    
    if st.button("Ask AI"):
        try:
            result = chat_query(question, mode)
            st.success("Query successful!")
            st.write(result["response"])
            st.caption(result["disclaimer"])
        except APIError as e:
            st.error(f"Error: {str(e)}")
