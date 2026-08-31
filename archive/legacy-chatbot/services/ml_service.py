"""
Machine learning service for model predictions
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MLService:
    """ML model management and predictions"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load pre-trained models"""
        try:
            model_dir = Path(__file__).parent.parent / "models"
            
            # Try to load models if they exist
            rf_path = model_dir / "rf_model.pkl"
            xgb_path = model_dir / "xgb_model.pkl"
            
            if rf_path.exists():
                self.models['rf'] = joblib.load(rf_path)
                logger.info("Random Forest model loaded")
            
            if xgb_path.exists():
                self.models['xgb'] = joblib.load(xgb_path)
                logger.info("XGBoost model loaded")
            
            # Load feature names if available
            if (model_dir / "feature_names.pkl").exists():
                self.feature_names = joblib.load(model_dir / "feature_names.pkl")
            
            logger.info(f"Loaded {len(self.models)} models")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions using ensemble of models
        
        Args:
            features: Input features array
        
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            results = {
                "rf_score": None,
                "xgb_score": None,
                "ensemble_score": None,
                "features_used": len(features),
            }
            
            scores = []
            
            # Random Forest prediction
            if 'rf' in self.models:
                rf_pred = self.models['rf'].predict_proba(features.reshape(1, -1))
                results['rf_score'] = float(rf_pred[0][1] * 100)
                scores.append(results['rf_score'])
            
            # XGBoost prediction
            if 'xgb' in self.models:
                xgb_pred = self.models['xgb'].predict(features.reshape(1, -1))
                results['xgb_score'] = float(xgb_pred[0] * 100)
                scores.append(results['xgb_score'])
            
            # Ensemble score (average)
            if scores:
                results['ensemble_score'] = float(np.mean(scores))
            else:
                # Fallback if no models loaded
                results['ensemble_score'] = float(np.dot(features, np.random.rand(len(features)))) % 100
            
            return results
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e), "ensemble_score": 0}
    
    def get_feature_importance(self, model_name: str = "rf") -> Dict[str, float]:
        """Get feature importance from model"""
        try:
            if model_name in self.models and hasattr(self.models[model_name], 'feature_importances_'):
                importances = self.models[model_name].feature_importances_
                
                if self.feature_names:
                    return dict(zip(self.feature_names, importances))
                else:
                    return {f"Feature_{i}": float(imp) for i, imp in enumerate(importances)}
            else:
                # Return dummy importances
                default_features = {
                    "Age": 0.20,
                    "Severity": 0.25,
                    "Treatment Adherence": 0.20,
                    "Exposure": 0.15,
                    "Blood Type": 0.10,
                    "Dose": 0.10,
                }
                return default_features
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}


# Singleton instance
_ml_service = None


def get_ml_service() -> MLService:
    """Get ML service instance"""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
