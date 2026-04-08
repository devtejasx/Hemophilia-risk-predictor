"""
API client for backend communication
"""

import requests
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class APIClient:
    """Client for API communication"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "HemophiliaAI/3.0"
        }
    
    def get_patients(self, limit: int = 100) -> Dict[str, Any]:
        """Get list of patients"""
        try:
            endpoint = f"{self.base_url}/api/patients"
            response = self.session.get(
                endpoint,
                headers=self.headers,
                params={"limit": limit},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching patients: {e}")
            return {"error": str(e), "patients": []}
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new patient"""
        try:
            endpoint = f"{self.base_url}/api/patients"
            response = self.session.post(
                endpoint,
                json=patient_data,
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error creating patient: {e}")
            return {"error": str(e)}
    
    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        """Get patient details"""
        try:
            endpoint = f"{self.base_url}/api/patients/{patient_id}"
            response = self.session.get(
                endpoint,
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching patient: {e}")
            return {"error": str(e)}
    
    def predict_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk prediction for patient"""
        try:
            endpoint = f"{self.base_url}/api/predict"
            response = self.session.post(
                endpoint,
                json=patient_data,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return {"error": str(e), "risk_score": 0}
    
    def get_analytics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get analytics data"""
        try:
            endpoint = f"{self.base_url}/api/analytics"
            response = self.session.get(
                endpoint,
                headers=self.headers,
                params=filters or {},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching analytics: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False


# Singleton instance
_api_client = None


def get_api_client(base_url: str = "http://localhost:8000") -> APIClient:
    """Get API client instance"""
    global _api_client
    if _api_client is None:
        _api_client = APIClient(base_url)
    return _api_client
