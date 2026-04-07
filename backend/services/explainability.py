"""
SHAP-based Model Explainability Service
========================================

Provides comprehensive model interpretability using SHAP (SHapley Additive exPlanations).
Generates local explanations for individual predictions and global feature importance.

Features:
- Individual prediction explanations (Shapley values)
- Global feature importance analysis
- SHAP visualization generation (waterfall, force, dependence plots)
- Feature contribution summaries
- Clinical interpretation of predictions
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    SHAP-based model explanation and interpretation service.
    
    Provides methods for generating and visualizing model predictions with
    detailed feature importance and contribution analysis.
    """
    
    def __init__(self, model: Any, background_data: Optional[np.ndarray] = None):
        """
        Initialize explainability service.
        
        Args:
            model: Trained ML model (XGBoost, Random Forest, etc.)
            background_data: Background data for SHAP explainer (samples for reference)
        """
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self.shap_values = None
        self.feature_names = []
        self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize SHAP explainer based on model type."""
        if self.model is None:
            logger.warning("Model is None, cannot initialize explainer")
            return
        
        try:
            # Determine model type and create appropriate explainer
            model_name = type(self.model).__name__.lower()
            
            if 'xgbregressor' in model_name or 'xgbclassifier' in model_name:
                self.explainer = shap.TreeExplainer(self.model)
            elif 'randomforest' in model_name:
                self.explainer = shap.TreeExplainer(self.model)
            elif 'gradientboosting' in model_name:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer for other models
                if self.background_data is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                        shap.sample(self.background_data, 100)
                    )
                else:
                    logger.warning(f"Model type {model_name} not directly supported, using LIME instead")
            
            logger.info(f"SHAP explainer initialized: {type(self.explainer).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            self.explainer = None
    
    def set_feature_names(self, feature_names: List[str]) -> None:
        """
        Set feature names for better interpretability.
        
        Args:
            feature_names: List of feature column names
        """
        self.feature_names = feature_names
        if self.explainer:
            self.explainer.expected_value
    
    def explain_prediction(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            instance: Input features for prediction (1D array or 2D with 1 row)
            feature_names: Optional feature names for interpretation
            
        Returns:
            Dictionary containing:
            - prediction: Model's prediction value
            - base_value: Expected value (model output average)
            - shap_values: Contribution of each feature
            - feature_contributions: List of (feature, contribution) tuples
            - top_positive: Top features increasing prediction
            - top_negative: Top features decreasing prediction
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
        
        # Ensure proper shape
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        try:
            # Get SHAP values
            shap_vals = self.explainer.shap_values(instance)
            
            # Handle multiple classes (get positive class)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            # Get prediction
            prediction = self.model.predict(instance)[0]
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(instance)[0]
            else:
                prediction_proba = None
            
            # Build feature contributions
            feature_names_to_use = feature_names or self.feature_names or [f"Feature_{i}" for i in range(len(shap_vals[0]))]
            
            contributions = [
                {
                    "feature": feature_names_to_use[i],
                    "value": float(instance[0, i]),
                    "contribution": float(shap_vals[0, i]),
                    "abs_contribution": float(abs(shap_vals[0, i]))
                }
                for i in range(len(feature_names_to_use))
            ]
            
            # Sort by absolute contribution
            contributions_sorted = sorted(contributions, key=lambda x: x["abs_contribution"], reverse=True)
            
            # Get top positive and negative contributors
            top_positive = [c for c in contributions_sorted if c["contribution"] > 0][:5]
            top_negative = [c for c in contributions_sorted if c["contribution"] < 0][:5]
            
            return {
                "prediction": float(prediction),
                "prediction_proba": prediction_proba.tolist() if prediction_proba is not None else None,
                "base_value": float(self.explainer.expected_value),
                "shap_values": shap_vals[0].tolist(),
                "feature_contributions": contributions_sorted,
                "top_positive_contributors": top_positive,
                "top_negative_contributors": top_negative,
                "feature_names": feature_names_to_use
            }
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {"error": str(e)}
    
    def explain_batch_predictions(
        self,
        instances: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate SHAP explanations for multiple predictions.
        
        Args:
            instances: Input features (2D array, rows are samples)
            feature_names: Optional feature names
            sample_size: If provided, sample this many explanations (useful for large datasets)
            
        Returns:
            List of explanation dictionaries
        """
        if sample_size and len(instances) > sample_size:
            indices = np.random.choice(len(instances), sample_size, replace=False)
            instances = instances[indices]
        
        explanations = []
        for i in range(len(instances)):
            explanation = self.explain_prediction(instances[i:i+1], feature_names)
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance(
        self,
        instances: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Get global feature importance from SHAP values.
        
        Args:
            instances: Input features for calculating importance (if None, uses background data)
            
        Returns:
            Dictionary containing:
            - mean_abs_shap: Mean absolute SHAP values
            - feature_importance: Sorted feature importance
            - feature_names: Corresponding feature names
        """
        if self.explainer is None:
            return {"error": "Explainer not initialized"}
        
        try:
            if instances is None:
                if self.background_data is None:
                    return {"error": "No instances or background data provided"}
                instances = self.background_data
            
            # Calculate SHAP values for all instances
            shap_vals = self.explainer.shap_values(instances)
            
            # Handle multiple classes
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            
            # Sort by importance
            indices = np.argsort(mean_abs_shap)[::-1]
            
            feature_names_to_use = self.feature_names or [f"Feature_{i}" for i in range(len(mean_abs_shap))]
            
            feature_importance = [
                {
                    "feature": feature_names_to_use[i],
                    "importance": float(mean_abs_shap[i]),
                    "rank": rank + 1
                }
                for rank, i in enumerate(indices)
            ]
            
            return {
                "feature_importance": feature_importance,
                "feature_names": feature_names_to_use,
                "total_features": len(mean_abs_shap)
            }
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {"error": str(e)}
    
    def generate_waterfall_plot(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_features: int = 10
    ) -> Optional[bytes]:
        """
        Generate SHAP waterfall plot for a prediction.
        
        Args:
            instance: Input features (1D or 2D with 1 row)
            feature_names: Optional feature names
            max_features: Maximum features to show in plot
            
        Returns:
            PNG image bytes or None if error
        """
        if self.explainer is None:
            return None
        
        try:
            # Ensure proper shape
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
            
            # Get SHAP values
            shap_vals = self.explainer.shap_values(instance)
            
            # Handle multiple classes
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_names_to_use = feature_names or self.feature_names or [f"Feature_{i}" for i in range(len(shap_vals[0]))]
            
            # Create Explanation object
            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=feature_names_to_use
            )
            
            # Generate waterfall plot
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=max_features, show=False)
            
            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error generating waterfall plot: {str(e)}")
            return None
    
    def generate_force_plot(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate SHAP force plot (HTML).
        
        Args:
            instance: Input features
            feature_names: Optional feature names
            
        Returns:
            HTML string or None if error
        """
        if self.explainer is None:
            return None
        
        try:
            if len(instance.shape) == 1:
                instance = instance.reshape(1, -1)
            
            shap_vals = self.explainer.shap_values(instance)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_names_to_use = feature_names or self.feature_names or [f"Feature_{i}" for i in range(len(shap_vals[0]))]
            
            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=self.explainer.expected_value,
                data=instance[0],
                feature_names=feature_names_to_use
            )
            
            # Generate force plot HTML
            return shap.getjs() + shap.force_plot(
                self.explainer.expected_value,
                shap_vals[0],
                instance[0],
                feature_names=feature_names_to_use,
                show=False
            ).html()
        except Exception as e:
            logger.error(f"Error generating force plot: {str(e)}")
            return None
    
    def generate_dependence_plot(
        self,
        feature_index: int,
        instances: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Optional[bytes]:
        """
        Generate SHAP dependence plot for a feature.
        
        Args:
            feature_index: Index of feature to analyze
            instances: Input features for all predictions
            feature_names: Optional feature names
            
        Returns:
            PNG image bytes or None if error
        """
        if self.explainer is None:
            return None
        
        try:
            shap_vals = self.explainer.shap_values(instances)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_names_to_use = feature_names or self.feature_names or [f"Feature_{i}" for i in range(instances.shape[1])]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_index,
                shap_vals,
                instances,
                feature_names=feature_names_to_use,
                show=False
            )
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error generating dependence plot: {str(e)}")
            return None
    
    def generate_summary_plot(
        self,
        instances: np.ndarray,
        feature_names: Optional[List[str]] = None,
        plot_type: str = "bar"
    ) -> Optional[bytes]:
        """
        Generate SHAP summary plot.
        
        Args:
            instances: Input features
            feature_names: Optional feature names
            plot_type: "bar" or "beeswarm"
            
        Returns:
            PNG image bytes or None if error
        """
        if self.explainer is None:
            return None
        
        try:
            shap_vals = self.explainer.shap_values(instances)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            
            feature_names_to_use = feature_names or self.feature_names or [f"Feature_{i}" for i in range(instances.shape[1])]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_vals,
                instances,
                feature_names=feature_names_to_use,
                plot_type=plot_type,
                show=False
            )
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error generating summary plot: {str(e)}")
            return None
    
    def generate_clinical_explanation(
        self,
        explanation: Dict[str, Any],
        risk_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate clinical interpretation of SHAP explanation.
        
        Args:
            explanation: SHAP explanation dictionary
            risk_threshold: Threshold for risk classification
            
        Returns:
            Dictionary containing clinical interpretation
        """
        try:
            prediction = explanation.get("prediction", 0)
            top_positive = explanation.get("top_positive_contributors", [])
            top_negative = explanation.get("top_negative_contributors", [])
            
            # Determine risk level
            if prediction >= 0.7:
                risk_level = "HIGH"
                risk_description = "High risk of adverse outcome"
            elif prediction >= risk_threshold:
                risk_level = "MODERATE"
                risk_description = "Moderate risk of adverse outcome"
            else:
                risk_level = "LOW"
                risk_description = "Low risk of adverse outcome"
            
            # Build clinical interpretation
            clinical_summary = {
                "risk_level": risk_level,
                "risk_description": risk_description,
                "prediction_score": float(prediction),
                "key_risk_factors": [
                    {
                        "factor": item["feature"],
                        "impact": "increases" if item["contribution"] > 0 else "decreases",
                        "magnitude": abs(item["contribution"]),
                        "current_value": item["value"]
                    }
                    for item in top_positive + top_negative
                ],
                "recommendations": self._generate_recommendations(risk_level, top_positive, top_negative),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return clinical_summary
        except Exception as e:
            logger.error(f"Error generating clinical explanation: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def _generate_recommendations(
        risk_level: str,
        top_positive: List[Dict[str, Any]],
        top_negative: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate clinical recommendations based on risk level.
        
        Args:
            risk_level: Risk classification
            top_positive: Top risk factors
            top_negative: Top protective factors
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append("Immediate clinical review recommended")
            recommendations.append("Consider intensive monitoring protocol")
            recommendations.append("Evaluate need for preventive interventions")
        elif risk_level == "MODERATE":
            recommendations.append("Regular follow-up recommended")
            recommendations.append("Monitor key risk factors closely")
        else:
            recommendations.append("Routine monitoring sufficient")
        
        # Add factor-specific recommendations
        for factor in top_positive:
            feature = factor["feature"].lower()
            if "hemoglobin" in feature or "hgb" in feature:
                recommendations.append("Monitor hemoglobin levels regularly")
            if "inhibitor" in feature:
                recommendations.append("Screen for inhibitor development")
            if "treatment" in feature:
                recommendations.append("Review treatment compliance")
        
        return recommendations
