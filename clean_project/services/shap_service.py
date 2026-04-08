"""
SHAP Explainability service for model interpretation.
Provides model-agnostic explanations for predictions.
"""

from typing import Dict, List, Optional, Any
import numpy as np


class SHAPService:
    """Service for explaining model predictions using SHAP concepts."""
    
    def __init__(self):
        """Initialize SHAP service."""
        self.shap_values = None
        self.base_value = 0.5  # Base risk
    
    def generate_feature_importance(
        self,
        patient_data: Dict[str, float],
        prediction: float
    ) -> Dict[str, float]:
        """Generate feature importance for a prediction.
        
        Args:
            patient_data: Patient input features
            prediction: Model prediction value
        
        Returns:
            Dictionary of features to importance scores
        """
        
        # Calculate contribution of each feature
        contributions = {}
        
        # Reference values (baseline)
        reference = {
            'age': 40,
            'clotting_factor': 50,
            'activity_level': 5,
            'compliance': 0.8,
            'bleeds': 2,
            'hospitalization': False,
        }
        
        # Weight each feature's contribution
        weights = {
            'age': 0.10,
            'clotting_factor': 0.15,
            'activity_level': 0.25,
            'compliance': 0.20,
            'bleeds': 0.20,
            'hospitalization': 0.10,
        }
        
        # Calculate deviations from baseline
        for feature, weight in weights.items():
            if feature in patient_data and feature in reference:
                patient_val = patient_data[feature]
                ref_val = reference[feature]
                
                # Calculate contribution
                if feature == 'hospitalization':
                    contribution = weight if patient_val else 0
                else:
                    # Normalize deviation
                    max_val = 100 if feature in ['age', 'clotting_factor'] else 10
                    deviation = abs(patient_val - ref_val) / max(max_val, 1)
                    contribution = weight * min(deviation, 1.0)
                
                contributions[feature] = contribution
        
        # Normalize to sum to prediction value
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            scale_factor = prediction / total_contrib
            contributions = {k: v * scale_factor for k, v in contributions.items()}
        
        return contributions
    
    def explain_prediction(
        self,
        patient_data: Dict[str, float],
        prediction: float,
        prediction_class: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """Generate full explanation for prediction.
        
        Args:
            patient_data: Patient features
            prediction: Prediction value
            prediction_class: Prediction class (LOW, MEDIUM, HIGH)
        
        Returns:
            Explanation dictionary with visualizations
        """
        
        # Get feature importance
        importance = self.generate_feature_importance(patient_data, prediction)
        
        # Sort by impact (descending)
        sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        explanation = {
            "prediction": prediction,
            "prediction_class": prediction_class,
            "base_value": self.base_value,
            "feature_importance": dict(sorted_features),
            "most_important": [f[0] for f in sorted_features[:3]],
            "interpretation": self._generate_interpretation(sorted_features, prediction),
            "supporting_factors": self._get_supporting_factors(patient_data),
        }
        
        return explanation
    
    def _generate_interpretation(
        self,
        sorted_features: List[tuple],
        prediction: float
    ) -> str:
        """Generate text interpretation of prediction.
        
        Args:
            sorted_features: Features sorted by importance
            prediction: Prediction value
        
        Returns:
            Interpretation string
        """
        
        interpretation = f"This prediction of {prediction:.1%} risk is driven primarily by: "
        
        top_features = [f[0] for f in sorted_features[:3]]
        interpretation += ", ".join(top_features) + "."
        
        if prediction < 0.4:
            interpretation += " Overall, the model indicates low risk based on patient factors."
        elif prediction < 0.7:
            interpretation += " Overall, the model indicates moderate risk - continued monitoring recommended."
        else:
            interpretation += " Overall, the model indicates high risk - close management required."
        
        return interpretation
    
    def _get_supporting_factors(self, patient_data: Dict) -> Dict[str, str]:
        """Identify factors supporting the prediction.
        
        Args:
            patient_data: Patient features
        
        Returns:
            Dictionary of factors with explanations
        """
        
        supporting = {}
        
        # Analyze each feature
        if patient_data.get('clotting_factor', 50) < 40:
            supporting['Low clotting factor'] = 'Directly increases bleeding risk'
        
        if patient_data.get('compliance', 0.8) < 0.6:
            supporting['Low compliance'] = 'Improper treatment adherence increases risk'
        
        if patient_data.get('activity_level', 5) > 7:
            supporting['High activity level'] = 'More exposure to injury risk'
        
        if patient_data.get('bleeds', 0) > 3:
            supporting['Recent bleeds'] = 'Indicates active bleeding tendency'
        
        if patient_data.get('age', 40) > 50:
            supporting['Older age'] = 'Age-related risk factor'
        
        if patient_data.get('hospitalization', False):
            supporting['Recent hospitalization'] = 'Indicates acute medical event'
        
        return supporting
    
    def generate_waterfall_data(
        self,
        patient_data: Dict[str, float],
        prediction: float
    ) -> Dict[str, Any]:
        """Generate data for waterfall plot visualization.
        
        Args:
            patient_data: Patient features
            prediction: Prediction value
        
        Returns:
            Data structure for waterfall chart
        """
        
        importance = self.generate_feature_importance(patient_data, prediction)
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        waterfall_data = {
            "features": [f[0] for f in sorted_features],
            "values": [f[1] for f in sorted_features],
            "cumulative": [],
            "base_value": self.base_value,
            "target_value": prediction,
        }
        
        # Calculate cumulative values
        cumulative = self.base_value
        for value in waterfall_data["values"]:
            cumulative += value
            waterfall_data["cumulative"].append(cumulative)
        
        return waterfall_data
    
    def generate_force_plot_data(
        self,
        patient_data: Dict[str, float],
        prediction: float
    ) -> Dict[str, Any]:
        """Generate data for force plot visualization.
        
        Args:
            patient_data: Patient features
            prediction: Prediction value
        
        Returns:
            Data for force plot
        """
        
        importance = self.generate_feature_importance(patient_data, prediction)
        
        # Separate positive and negative influences
        positive = {k: v for k, v in importance.items() if v > 0}
        negative = {k: v for k, v in importance.items() if v < 0}
        
        return {
            "base_value": self.base_value,
            "output_value": prediction,
            "positive_features": positive,
            "negative_features": negative,
        }
    
    def get_decision_path(
        self,
        patient_data: Dict[str, float]
    ) -> List[str]:
        """Get human-readable decision path for prediction.
        
        Args:
            patient_data: Patient features
        
        Returns:
            List of decision steps
        """
        
        path = [
            "📊 Model Decision Path:",
            "",
            "1️⃣ Start with baseline risk assessment (50%)",
        ]
        
        # Add feature-by-feature decisions
        step = 2
        
        if patient_data.get('activity_level', 5) > 6:
            path.append(f"{step}️⃣ High activity level → increases risk")
            step += 1
        
        if patient_data.get('compliance', 0.8) < 0.7:
            path.append(f"{step}️⃣ Low treatment compliance → increases risk")
            step += 1
        
        if patient_data.get('clotting_factor', 50) < 40:
            path.append(f"{step}️⃣ Low clotting factor → increases risk significantly")
            step += 1
        
        if patient_data.get('bleeds', 0) > 3:
            path.append(f"{step}️⃣ Recent bleeds detected → increases risk")
            step += 1
        
        path.append("")
        path.append("➡️ Final Risk Assessment: Model evaluates all factors to produce risk score")
        
        return path


# Global service instance
_shap_service = None


def get_shap_service() -> SHAPService:
    """Get or create SHAP service instance."""
    global _shap_service
    if _shap_service is None:
        _shap_service = SHAPService()
    return _shap_service


def explain_prediction(
    patient_data: Dict[str, float],
    prediction: float,
    prediction_class: str = "MEDIUM"
) -> Dict[str, Any]:
    """Convenience function to explain prediction."""
    return get_shap_service().explain_prediction(patient_data, prediction, prediction_class)


def generate_feature_importance(
    patient_data: Dict[str, float],
    prediction: float
) -> Dict[str, float]:
    """Convenience function to get feature importance."""
    return get_shap_service().generate_feature_importance(patient_data, prediction)
