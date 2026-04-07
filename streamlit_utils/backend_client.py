"""
Backend Client - FastAPI communication wrapper
"""

import requests
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
import streamlit as st


class BackendClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize backend client
        
        Args:
            base_url: Base URL of FastAPI backend
        """
        self.base_url = base_url
        self.api_prefix = f"{base_url}/api"
        self.timeout = 30
        self.session = requests.Session()
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    
    def health_check(self) -> bool:
        """Check if backend is available"""
        try:
            response = self.session.get(
                f"{self.api_prefix}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend status"""
        try:
            response = self.session.get(
                f"{self.api_prefix}/status",
                timeout=self.timeout
            )
            return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ========================================================================
    # PREDICTIONS
    # ========================================================================
    
    def predict(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get prediction from model
        
        Args:
            patient_data: Patient clinical data
            
        Returns:
            Prediction result with risk score, label, and explanation
        """
        try:
            response = self.session.post(
                f"{self.api_prefix}/predict",
                json=patient_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def predict_batch(self, patients_data: List[Dict[str, Any]]) -> Optional[List[Dict]]:
        """Get predictions for multiple patients
        
        Args:
            patients_data: List of patient clinical data
            
        Returns:
            List of prediction results
        """
        try:
            response = self.session.post(
                f"{self.api_prefix}/predict/batch",
                json={"patients": patients_data},
                timeout=self.timeout * 2  # Longer timeout for batch
            )
            response.raise_for_status()
            return response.json().get("predictions", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Batch prediction error: {str(e)}")
            return None
    
    def get_explainability(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get SHAP explainability for prediction
        
        Args:
            patient_data: Patient clinical data
            
        Returns:
            SHAP explanation data
        """
        try:
            response = self.session.post(
                f"{self.api_prefix}/explain",
                json=patient_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Explainability error: {str(e)}")
            return None
    
    # ========================================================================
    # PATIENT DATA
    # ========================================================================
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient data
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Patient data or None
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/patients/{patient_id}",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching patient: {str(e)}")
            return None
    
    def get_patient_history(self, patient_id: str, limit: int = 50) -> Optional[List[Dict]]:
        """Get patient history
        
        Args:
            patient_id: Patient ID
            limit: Number of records to fetch
            
        Returns:
            List of historical records
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/patients/{patient_id}/history",
                params={"limit": limit},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("history", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching history: {str(e)}")
            return None
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create new patient
        
        Args:
            patient_data: Patient data
            
        Returns:
            Created patient with ID
        """
        try:
            response = self.session.post(
                f"{self.api_prefix}/patients",
                json=patient_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error creating patient: {str(e)}")
            return None
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update patient data
        
        Args:
            patient_id: Patient ID
            patient_data: Updated patient data
            
        Returns:
            Updated patient data
        """
        try:
            response = self.session.put(
                f"{self.api_prefix}/patients/{patient_id}",
                json=patient_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error updating patient: {str(e)}")
            return None
    
    def list_patients(self, limit: int = 100) -> Optional[List[Dict]]:
        """List all patients
        
        Args:
            limit: Maximum number of patients to return
            
        Returns:
            List of patients
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/patients",
                params={"limit": limit},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("patients", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching patients: {str(e)}")
            return None
    
    # ========================================================================
    # CHAT/CHATBOT
    # ========================================================================
    
    def send_chat_message(self, message: str, patient_id: Optional[str] = None) -> Optional[str]:
        """Send message to chatbot and get response
        
        Args:
            message: User message
            patient_id: Optional patient ID for context
            
        Returns:
            Chatbot response
        """
        try:
            payload = {
                "message": message,
                "patient_id": patient_id
            }
            response = self.session.post(
                f"{self.api_prefix}/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            st.error(f"Chat error: {str(e)}")
            return None
    
    def get_chat_history(self, patient_id: Optional[str] = None, limit: int = 50) -> Optional[List[Dict]]:
        """Get chat history
        
        Args:
            patient_id: Optional patient ID
            limit: Number of messages to fetch
            
        Returns:
            List of chat messages
        """
        try:
            params = {"limit": limit}
            if patient_id:
                params["patient_id"] = patient_id
            
            response = self.session.get(
                f"{self.api_prefix}/chat/history",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get("messages", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching chat history: {str(e)}")
            return None
    
    # ========================================================================
    # ANALYTICS/METRICS
    # ========================================================================
    
    def get_dashboard_stats(self) -> Optional[Dict[str, Any]]:
        """Get dashboard statistics
        
        Returns:
            Dashboard metrics (total patients, avg risk, etc.)
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/analytics/dashboard",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching dashboard stats: {str(e)}")
            return None
    
    def get_risk_distribution(self) -> Optional[Dict[str, Any]]:
        """Get risk distribution across patients
        
        Returns:
            Risk distribution data
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/analytics/risk-distribution",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching risk distribution: {str(e)}")
            return None
    
    def get_trends(self, time_period: str = "month") -> Optional[Dict[str, Any]]:
        """Get trending data
        
        Args:
            time_period: "week", "month", "quarter"
            
        Returns:
            Trend data
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/analytics/trends",
                params={"period": time_period},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching trends: {str(e)}")
            return None
    
    # ========================================================================
    # MODEL INFORMATION
    # ========================================================================
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information
        
        Returns:
            Model details (version, accuracy, etc.)
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/model/info",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching model info: {str(e)}")
            return None
    
    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance
        
        Returns:
            Feature importance scores
        """
        try:
            response = self.session.get(
                f"{self.api_prefix}/model/features",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching feature importance: {str(e)}")
            return None
    
    # ========================================================================
    # ERROR HANDLING HELPER
    # ========================================================================
    
    def handle_error(self, error_response: requests.Response) -> str:
        """Handle API error responses
        
        Args:
            error_response: Error response object
            
        Returns:
            Error message string
        """
        try:
            error_data = error_response.json()
            return error_data.get("detail", "An error occurred")
        except:
            return f"Error {error_response.status_code}: {error_response.text}"


# ========================================================================
# SINGLETON INSTANCE
# ========================================================================

@st.cache_resource
def get_backend_client(base_url: str = "http://localhost:8000") -> BackendClient:
    """Get or create backend client (singleton)
    
    Args:
        base_url: Base URL of FastAPI backend
        
    Returns:
        BackendClient instance
    """
    return BackendClient(base_url)
