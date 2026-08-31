"""
ML model utilities and prediction logic
Extracted from app.py for use in FastAPI backend
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import os
from pathlib import Path


# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


def load_models() -> Tuple[Optional[Any], Optional[Any], Optional[List[str]]]:
    """Load trained RF and XGBoost models"""
    try:
        rf_path = PROJECT_ROOT / "rf.pkl"
        xgb_path = PROJECT_ROOT / "xgb.pkl"
        
        if not rf_path.exists() or not xgb_path.exists():
            print("⚠️ Model files not found")
            return None, None, None
        
        # Load models with mmap_mode for large files
        rf_model = joblib.load(str(rf_path), mmap_mode='r')
        xgb_model = joblib.load(str(xgb_path), mmap_mode='r')
        
        # Get feature names from XGBoost
        columns = xgb_model.get_booster().feature_names
        
        return rf_model, xgb_model, columns
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def calculate_clinical_adjustment(
    ethnicity: Optional[str] = None,
    blood_type: Optional[str] = None,
    hla_typing: Optional[str] = None,
    product_type: Optional[str] = None,
    treatment_adherence: Optional[float] = None,
    family_history: Optional[str] = None,
    previous_inhibitor: Optional[bool] = None,
    joint_damage_score: Optional[int] = None,
    bleeding_episodes: Optional[int] = None,
    baseline_factor_level: Optional[float] = None,
    immunosuppression: Optional[bool] = None,
    active_infection: Optional[bool] = None,
    vaccination_status: Optional[str] = None,
    physical_activity: Optional[str] = None,
    stress_level: Optional[str] = None,
    comorbidities: Optional[str] = None
) -> float:
    """
    Calculate clinical parameter adjustments to risk score
    """
    adjustment = 0.0
    
    # Ethnicity-based adjustments
    if ethnicity:
        ethnicity_lower = ethnicity.lower()
        if "african" in ethnicity_lower:
            adjustment += 0.05  # Slightly elevated risk
        elif "asian" in ethnicity_lower:
            adjustment -= 0.02  # Slightly reduced risk
    
    # Blood type adjustments
    if blood_type:
        if blood_type in ["AB", "B"]:
            adjustment += 0.03
    
    # Treatment adherence impact
    if treatment_adherence is not None:
        if treatment_adherence < 50:
            adjustment += 0.10
        elif treatment_adherence < 75:
            adjustment += 0.05
        elif treatment_adherence > 95:
            adjustment -= 0.05
    
    # Previous inhibitor history (strong factor)
    if previous_inhibitor:
        adjustment += 0.15
    
    # Bleeding episodes
    if bleeding_episodes and bleeding_episodes > 0:
        adjustment += min(0.15, bleeding_episodes * 0.01)
    
    # Joint damage
    if joint_damage_score and joint_damage_score > 0:
        adjustment += min(0.10, joint_damage_score * 0.02)
    
    # Immunosuppression
    if immunosuppression:
        adjustment += 0.08
    
    # Active infection
    if active_infection:
        adjustment += 0.12
    
    # Physical activity (protective)
    if physical_activity:
        activity_lower = physical_activity.lower()
        if activity_lower in ["high", "very high", "athletic"]:
            adjustment -= 0.05
    
    # Stress level
    if stress_level:
        stress_lower = stress_level.lower()
        if stress_lower in ["high", "very high"]:
            adjustment += 0.05
    
    return adjustment


def predict_inhibitor_risk(
    age: int,
    dose: float,
    exposure: int,
    severity: str,
    mutation: str,
    ethnicity: Optional[str] = None,
    blood_type: Optional[str] = None,
    hla_typing: Optional[str] = None,
    product_type: Optional[str] = None,
    treatment_adherence: Optional[float] = None,
    family_history: Optional[str] = None,
    previous_inhibitor: Optional[bool] = None,
    joint_damage_score: Optional[int] = None,
    bleeding_episodes: Optional[int] = None,
    baseline_factor_level: Optional[float] = None,
    immunosuppression: Optional[bool] = None,
    active_infection: Optional[bool] = None,
    vaccination_status: Optional[str] = None,
    physical_activity: Optional[str] = None,
    stress_level: Optional[str] = None,
    comorbidities: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict inhibitor development risk using ensemble models
    Returns prediction data with risk score, category, and contributing factors
    """
    rf_model, xgb_model, columns = load_models()
    
    if rf_model is None:
        return generate_fallback_prediction(
            age, dose, exposure, severity, mutation, ethnicity,
            blood_type, hla_typing, product_type, treatment_adherence,
            family_history, previous_inhibitor, joint_damage_score,
            bleeding_episodes, baseline_factor_level, immunosuppression,
            active_infection, vaccination_status, physical_activity,
            stress_level, comorbidities
        )
    
    try:
        # Create feature data
        data = {
            "mutation_type": mutation.lower(),
            "exon": {"intron22": 22, "missense": 5, "nonsense": 10}.get(mutation.lower(), 22),
            "severity": severity.lower(),
            "age_first_treatment": age,
            "dose_intensity": dose,
            "exposure_days": exposure
        }
        
        # Convert to DataFrame and encode
        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=['mutation_type', 'severity'])
        
        # Ensure all columns exist
        for col in columns:
            if col not in df:
                df[col] = 0
        
        # Select only required columns
        df = df[columns]
        
        # Get predictions from both models
        rf_proba = rf_model.predict_proba(df)[0][1]
        xgb_proba = xgb_model.predict_proba(df)[0][1]
        
        # Ensemble: average
        risk_score = (rf_proba + xgb_proba) / 2
        
        # Apply clinical adjustments
        risk_adjustment = calculate_clinical_adjustment(
            ethnicity, blood_type, hla_typing, product_type, treatment_adherence,
            family_history, previous_inhibitor, joint_damage_score, bleeding_episodes,
            baseline_factor_level, immunosuppression, active_infection, vaccination_status,
            physical_activity, stress_level, comorbidities
        )
        
        # Apply adjustment and clamp to [0, 1]
        risk_score = np.clip(risk_score + risk_adjustment, 0, 1)
        
        # Determine risk category
        if risk_score < 0.3:
            risk_category = "Low"
        elif risk_score < 0.6:
            risk_category = "Medium"
        elif risk_score < 0.8:
            risk_category = "High"
        else:
            risk_category = "Critical"
        
        # Calculate contributing factors
        contributing_factors = []
        
        if previous_inhibitor:
            contributing_factors.append({"factor": "Previous inhibitor history", "weight": 0.15})
        if active_infection:
            contributing_factors.append({"factor": "Active infection", "weight": 0.12})
        if immunosuppression:
            contributing_factors.append({"factor": "Immunosuppression", "weight": 0.08})
        if treatment_adherence and treatment_adherence < 50:
            contributing_factors.append({"factor": "Low treatment adherence", "weight": 0.10})
        if bleeding_episodes and bleeding_episodes > 0:
            contributing_factors.append({"factor": "Frequent bleeding episodes", "weight": min(0.15, bleeding_episodes * 0.01)})
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_category, contributing_factors)
        
        return {
            "risk_score": float(risk_score),
            "risk_category": risk_category,
            "confidence": float((rf_proba + xgb_proba) / 2),  # Average model confidence
            "contributing_factors": contributing_factors,
            "recommendations": recommendations,
            "model_used": "ensemble"
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return generate_fallback_prediction(
            age, dose, exposure, severity, mutation, ethnicity,
            blood_type, hla_typing, product_type, treatment_adherence,
            family_history, previous_inhibitor, joint_damage_score,
            bleeding_episodes, baseline_factor_level, immunosuppression,
            active_infection, vaccination_status, physical_activity,
            stress_level, comorbidities
        )


def generate_recommendations(risk_category: str, contributing_factors: List[Dict[str, Any]]) -> List[str]:
    """Generate clinical recommendations based on risk category"""
    recommendations = []
    
    base_recommendations = {
        "Low": [
            "Continue regular monitoring",
            "Maintain current treatment regimen",
            "Annual inhibitor screening recommended"
        ],
        "Medium": [
            "Increase monitoring frequency to every 6 months",
            "Review treatment adherence",
            "Consider inhibitor screening every 3-6 months"
        ],
        "High": [
            "Monthly inhibitor screening recommended",
            "Review and optimize treatment strategy",
            "Consult with hemophilia specialist regarding immune tolerance induction",
            "Consider switching to different product type if appropriate"
        ],
        "Critical": [
            "Immediate specialist consultation required",
            "Monthly or more frequent inhibitor screening",
            "Urgent review of treatment strategy",
            "Consider hospitalization for immune tolerance induction evaluation",
            "Monitor for potential immune-mediated complications"
        ]
    }
    
    recommendations.extend(base_recommendations.get(risk_category, []))
    
    # Add factor-specific recommendations
    for factor in contributing_factors:
        if "adherence" in factor.get("factor", "").lower():
            recommendations.append("Implement adherence support strategy (education, reminders, support groups)")
        if "infection" in factor.get("factor", "").lower():
            recommendations.append("Treat active infection and recheck risk after resolution")
        if "inhibitor history" in factor.get("factor", "").lower():
            recommendations.append("Increased vigilance for inhibitor recurrence - frequent screening essential")
    
    return list(set(recommendations))  # Remove duplicates


def generate_fallback_prediction(
    age: int,
    dose: float,
    exposure: int,
    severity: str,
    mutation: str,
    ethnicity: Optional[str] = None,
    blood_type: Optional[str] = None,
    hla_typing: Optional[str] = None,
    product_type: Optional[str] = None,
    treatment_adherence: Optional[float] = None,
    family_history: Optional[str] = None,
    previous_inhibitor: Optional[bool] = None,
    joint_damage_score: Optional[int] = None,
    bleeding_episodes: Optional[int] = None,
    baseline_factor_level: Optional[float] = None,
    immunosuppression: Optional[bool] = None,
    active_infection: Optional[bool] = None,
    vaccination_status: Optional[str] = None,
    physical_activity: Optional[str] = None,
    stress_level: Optional[str] = None,
    comorbidities: Optional[str] = None
) -> Dict[str, Any]:
    """Generate fallback prediction when models unavailable"""
    
    # Base risk calculation using clinical parameters
    risk = 0.3  # Base risk
    
    # Severity adjustment
    severity_risks = {"mild": 0.1, "moderate": 0.35, "severe": 0.5}
    risk = severity_risks.get(severity.lower(), 0.3)
    
    # Exposure adjustment
    if exposure > 129:  # >9 exposures (129 days typical)
        risk += 0.15
    
    # Previous inhibitor (strong factor)
    if previous_inhibitor:
        risk += 0.2
    
    # Age adjustment (young patients at higher risk)
    if age < 5:
        risk += 0.15
    elif age < 12:
        risk += 0.10
    
    # Mutation type adjustment
    mutation_lower = mutation.lower()
    if "intron" in mutation_lower:
        risk += 0.10
    elif "deletions" in mutation_lower:
        risk += 0.08
    
    # Clamp to [0, 1]
    risk = np.clip(risk, 0, 1)
    
    # Determine category
    if risk < 0.3:
        category = "Low"
    elif risk < 0.6:
        category = "Medium"
    elif risk < 0.8:
        category = "High"
    else:
        category = "Critical"
    
    return {
        "risk_score": float(risk),
        "risk_category": category,
        "confidence": 0.5,  # Lower confidence for fallback
        "contributing_factors": [
            {"factor": "Severity", "weight": severity_risks.get(severity.lower(), 0.3)},
            {"factor": "Exposure time", "weight": 0.15 if exposure > 129 else 0}
        ],
        "recommendations": generate_recommendations(category, []),
        "model_used": "fallback"
    }


def get_feature_importance(model_type: str = "xgb") -> Dict[str, float]:
    """Get feature importance from trained model"""
    try:
        if model_type == "xgb":
            _, xgb_model, _ = load_models()
            if xgb_model is None:
                return {}
            importance = xgb_model.get_booster().get_score(importance_type='weight')
            return {k: float(v) for k, v in importance.items()}
        else:
            rf_model, _, _ = load_models()
            if rf_model is None:
                return {}
            importance = rf_model.feature_importances_
            feature_names = rf_model.feature_names_in_
            return {name: float(imp) for name, imp in zip(feature_names, importance)}
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return {}
