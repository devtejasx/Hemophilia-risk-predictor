"""
Machine Learning service for risk prediction.
Consolidated ML logic for hemophilia risk assessment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class MLService:
    """Machine learning prediction service."""
    
    def __init__(self):
        """Initialize ML service."""
        self.model_loaded = False
        self.model = None
        self.feature_names = [
            'age', 'clotting_factor', 'activity_level',
            'compliance', 'bleeds', 'hospitalization'
        ]
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load ML model from file.
        
        Args:
            model_path: Path to model pickle file (optional)
        
        Returns:
            True if loaded successfully
        """
        try:
            if model_path:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_loaded = True
                return True
            else:
                # If no model file, use fallback model
                self.model = self._create_fallback_model()
                self.model_loaded = True
                return True
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            # Fall back to demo model
            self.model = self._create_fallback_model()
            self.model_loaded = True
            return False
    
    def calculate_risk_score(
        self,
        age: int,
        clotting_factor: float,
        activity_level: int,
        compliance: float,
        bleeds: int,
        hospitalization: bool = False,
    ) -> float:
        """Calculate hemophilia risk score.
        
        Args:
            age: Patient age in years
            clotting_factor: Clotting factor percentage (0-100)
            activity_level: Activity level (1-10)
            compliance: Treatment compliance (0-1)
            bleeds: Number of recent bleeds
            hospitalization: Recent hospitalization flag
        
        Returns:
            Risk score from 0 to 1
        """
        
        # Normalize inputs to 0-1 range for consistent scoring
        age_norm = min(age / 80, 1.0)  # Normalize age
        clotting_norm = clotting_factor / 100.0
        activity_norm = activity_level / 10.0
        bleeds_norm = min(bleeds / 10.0, 1.0)  # Cap at 10 bleeds
        hospitalization_norm = 1.0 if hospitalization else 0.0
        
        # Calculate weighted risk score
        risk_score = (
            0.15 * (1 - clotting_norm) +      # Low clotting = high risk (inverse)
            0.20 * (1 - compliance) +          # Low compliance = high risk (inverse)
            0.25 * activity_norm +             # High activity = high risk
            0.20 * bleeds_norm +               # More bleeds = high risk
            0.10 * age_norm +                  # Older age = higher risk
            0.10 * hospitalization_norm        # Hospitalization = risk factor
        )
        
        # Clamp to 0-1 range
        risk_score = max(0.0, min(1.0, risk_score))
        
        return risk_score
    
    def predict_risk(
        self,
        patient_data: Dict[str, float]
    ) -> Dict[str, any]:
        """Predict risk for a patient.
        
        Args:
            patient_data: Dictionary with patient parameters
        
        Returns:
            Dictionary with prediction and confidence
        """
        
        # Extract features
        age = patient_data.get('age', 40)
        clotting_factor = patient_data.get('clotting_factor', 50)
        activity_level = patient_data.get('activity_level', 5)
        compliance = patient_data.get('compliance', 0.8)
        bleeds = patient_data.get('bleeds', 2)
        hospitalization = patient_data.get('hospitalization', False)
        
        # Calculate risk
        risk_score = self.calculate_risk_score(
            age=age,
            clotting_factor=clotting_factor,
            activity_level=activity_level,
            compliance=compliance,
            bleeds=bleeds,
            hospitalization=hospitalization,
        )
        
        # Determine risk category
        if risk_score < 0.4:
            risk_category = "LOW"
        elif risk_score < 0.7:
            risk_category = "MEDIUM"
        else:
            risk_category = "HIGH"
        
        # Calculate confidence (inverse of uncertainty)
        confidence = self._calculate_confidence(patient_data)
        
        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "confidence": confidence,
            "features": {
                "age": age,
                "clotting_factor": clotting_factor,
                "activity_level": activity_level,
                "compliance": compliance,
                "bleeds": bleeds,
                "hospitalization": hospitalization,
            }
        }
    
    def batch_predict(self, patients: List[Dict]) -> List[Dict]:
        """Predict risk for multiple patients.
        
        Args:
            patients: List of patient data dictionaries
        
        Returns:
            List of prediction results
        """
        
        return [self.predict_risk(patient) for patient in patients]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature names to importance values
        """
        
        return {
            "activity_level": 0.25,
            "compliance": 0.20,
            "clotting_factor": 0.15,
            "bleeds": 0.20,
            "age": 0.10,
            "hospitalization": 0.10,
        }
    
    def explain_prediction(self, risk_score: float, patient_data: Dict) -> str:
        """Generate human-readable explanation of prediction.
        
        Args:
            risk_score: The calculated risk score
            patient_data: Patient data used for prediction
        
        Returns:
            Explanation string
        """
        
        explanations = []
        
        # Clotting factor
        cf = patient_data.get('clotting_factor', 50)
        if cf < 30:
            explanations.append("Low clotting factor significantly increases risk")
        elif cf > 80:
            explanations.append("High clotting factor reduces risk")
        
        # Compliance
        compliance = patient_data.get('compliance', 0.8)
        if compliance < 0.6:
            explanations.append("Poor treatment compliance increases risk")
        elif compliance > 0.9:
            explanations.append("Excellent treatment compliance reduces risk")
        
        # Activity level
        activity = patient_data.get('activity_level', 5)
        if activity > 7:
            explanations.append("High activity level increases risk")
        
        # Bleeds
        bleeds = patient_data.get('bleeds', 0)
        if bleeds > 5:
            explanations.append("Frequent recent bleeds indicates elevated risk")
        
        # Age
        age = patient_data.get('age', 40)
        if age > 60:
            explanations.append("Older age is a risk factor")
        
        if not explanations:
            explanations.append("Risk factors are within normal range")
        
        return ". ".join(explanations) + "."
    
    def _create_fallback_model(self):
        """Create a simple fallback model for demo purposes."""
        return None  # Use calculate_risk_score directly
    
    def _calculate_confidence(self, patient_data: Dict) -> float:
        """Calculate confidence score for prediction.
        
        Args:
            patient_data: Patient data
        
        Returns:
            Confidence score 0-1
        """
        
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for missing data
        required_fields = ['age', 'clotting_factor', 'activity_level', 'compliance', 'bleeds']
        missing = sum(1 for field in required_fields if field not in patient_data or patient_data[field] is None)
        
        confidence -= (missing * 0.05)
        
        return max(0.5, min(1.0, confidence))


# Global service instance
_ml_service = None


def get_ml_service() -> MLService:
    """Get or create ML service instance."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
        _ml_service.load_model()
    return _ml_service


def predict_risk(patient_data: Dict) -> Dict:
    """Convenience function to predict risk."""
    return get_ml_service().predict_risk(patient_data)


def get_feature_importance() -> Dict[str, float]:
    """Convenience function to get feature importance."""
    return get_ml_service().get_feature_importance()
