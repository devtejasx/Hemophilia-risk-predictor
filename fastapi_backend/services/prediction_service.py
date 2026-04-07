"""
Prediction Service
ML model management and prediction logic
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import sqlite3
from datetime import datetime
from config import settings
from exceptions import ModelLoadException, PredictionException
from models import PredictionInput, FeatureImportance


class PredictionService:
    """Service for ML predictions"""
    
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.columns = None
        self.explainer = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained models and preprocessor"""
        try:
            # Load models with memory optimization
            self.rf_model = joblib.load(settings.RF_MODEL_PATH, mmap_mode='r')
            self.xgb_model = joblib.load(settings.XGB_MODEL_PATH, mmap_mode='r')
            
            # Load columns (feature names)
            try:
                self.columns = joblib.load(settings.COLUMNS_PATH, mmap_mode='r')
            except:
                self.columns = None
            
            # Initialize SHAP explainer for feature importance
            try:
                import shap
                self.explainer = shap.TreeExplainer(self.rf_model)
            except Exception as e:
                print(f"Warning: SHAP initialization failed: {e}")
                self.explainer = None
                
        except Exception as e:
            raise ModelLoadException(f"Failed to load models: {str(e)}")
    
    def predict(self, prediction_input: PredictionInput, patient_id: int = None) -> Dict[str, Any]:
        """
        Generate prediction for patient
        
        Args:
            prediction_input: Patient data for prediction
            patient_id: Optional patient ID for tracking
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Prepare features
            features = self._prepare_features(prediction_input)
            
            # Create DataFrame
            df = pd.DataFrame([features])
            df = pd.get_dummies(df)
            
            # Align columns if available
            if self.columns is not None:
                for col in self.columns:
                    if col not in df:
                        df[col] = 0
                df = df[self.columns]
            
            # Get predictions from both models
            rf_pred = self.rf_model.predict_proba(df)[0][1]
            xgb_pred = self.xgb_model.predict_proba(df)[0][1]
            
            # Ensemble prediction (average)
            risk_score = (rf_pred + xgb_pred) / 2
            
            # Determine risk category
            if risk_score < 0.33:
                risk_category = "Low"
            elif risk_score < 0.67:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            # Get feature importance
            top_features = self._get_feature_importance(df)
            
            result = {
                "risk_score": float(risk_score),
                "risk_category": risk_category,
                "confidence": float(max(rf_pred, xgb_pred)),
                "top_features": top_features,
                "explanation": f"Risk prediction based on {len(top_features)} key factors",
                "timestamp": datetime.utcnow()
            }
            
            return result
            
        except Exception as e:
            raise PredictionException(f"Prediction failed: {str(e)}")
    
    def _prepare_features(self, prediction_input: PredictionInput) -> Dict[str, Any]:
        """Prepare features from input"""
        return {
            "mutation_type": prediction_input.mutation,
            "severity": prediction_input.severity,
            "age_first_treatment": prediction_input.age,
            "dose_intensity": prediction_input.dose_intensity,
            "exposure_days": prediction_input.exposure_days,
            "hemoglobin": prediction_input.hemoglobin,
            "white_blood_cells": prediction_input.white_blood_cells,
            "platelets": prediction_input.platelets
        }
    
    def _get_feature_importance(self, df: pd.DataFrame) -> List[FeatureImportance]:
        """Calculate feature importance using SHAP or permutation"""
        try:
            if self.explainer is not None:
                # Use SHAP for feature importance
                shap_values = self.explainer.shap_values(df)
                if isinstance(shap_values, list):
                    shap_vals = np.array(shap_values[1]).flatten()  # Class 1 for binary
                else:
                    shap_vals = np.array(shap_values).flatten()
                
                # Get top 3 features
                top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]
                
                features = []
                for idx in top_indices:
                    importance = float(abs(shap_vals[idx]))
                    if importance > 0:
                        impact = "High" if importance > 0.1 else "Medium"
                        features.append(FeatureImportance(
                            feature=df.columns[idx],
                            importance_score=importance,
                            impact=impact
                        ))
                return features
            else:
                # Fallback: return top columns
                return [FeatureImportance(
                    feature=col,
                    importance_score=0.5,
                    impact="Medium"
                ) for col in df.columns[:3]]
                
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return []
    
    def batch_predict(self, predictions_input: List[PredictionInput]) -> List[Dict[str, Any]]:
        """Make predictions for multiple patients"""
        results = []
        for pred_input in predictions_input:
            try:
                result = self.predict(pred_input)
                results.append(result)
            except Exception as e:
                print(f"Error in batch prediction: {e}")
                continue
        
        return results
    
    def models_loaded(self) -> bool:
        """Check if models are successfully loaded"""
        return self.rf_model is not None and self.xgb_model is not None


# Global prediction service instance
prediction_service = PredictionService()
