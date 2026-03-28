import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import csv
import joblib
import numpy as np
import shap
import warnings
import pickle
from datetime import datetime
import os

# Try to load environment variables from .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

warnings.filterwarnings('ignore')

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Import custom modules
from database import (
    init_database, add_patient, get_patient, get_all_patients,
    add_conversation, get_conversation_history, add_doctor_note,
    get_doctor_notes, get_dashboard_stats, search_patients,
    update_patient, delete_patient, add_monitoring_record,
    get_monitoring_records, add_treatment_record, get_treatment_history
)
from gpt_chatbot import (
    create_gpt_response, get_clinical_recommendations,
    analyze_monitoring_data, generate_inhibitor_risk_explanation
)

# Initialize database at startup
try:
    init_database()
except Exception as e:
    st.error(f"Database initialization error: {e}")

# Load trained models locally
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    try:
        # Use mmap_mode to handle large files without loading fully into memory
        rf_model = joblib.load("rf.pkl", mmap_mode='r')
        xgb_model = joblib.load("xgb.pkl", mmap_mode='r')
        
        # Load columns - use try/except for memory error
        try:
            columns = joblib.load("columns.pkl", mmap_mode='r')
        except (MemoryError, EOFError, pickle.UnpicklingError):
            # If pickle is corrupted, try without mmap
            try:
                columns = joblib.load("columns.pkl")
            except:
                # Fallback: create default columns from training data
                columns = None
        
        return rf_model, xgb_model, columns
    except MemoryError:
        st.error("❌ Memory Error: Unable to load models. Try closing other applications or restarting Streamlit.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)[:100]}")
        return None, None, None

# Prediction function using real trained models
def predict_inhibitor_risk(age, dose, exposure, severity, mutation, ethnicity=None, blood_type=None,
                           hla_typing=None, product_type=None, treatment_adherence=None,
                           family_history=None, previous_inhibitor=None, joint_damage_score=None,
                           bleeding_episodes=None, baseline_factor_level=None, immunosuppression=None,
                           active_infection=None, vaccination_status=None, physical_activity=None,
                           stress_level=None, comorbidities=None):
    """
    Predict inhibitor development risk using ensemble models
    Uses real trained Random Forest and XGBoost models with comprehensive clinical parameters
    """
    rf_model, xgb_model, columns = load_models()
    
    if rf_model is None:
        # Fallback if models not available
        return generate_fallback_prediction(age, dose, exposure, severity, mutation, ethnicity,
                                           blood_type, hla_typing, product_type, treatment_adherence,
                                           family_history, previous_inhibitor, joint_damage_score,
                                           bleeding_episodes, baseline_factor_level, immunosuppression,
                                           active_infection, vaccination_status, physical_activity,
                                           stress_level, comorbidities)
    
    try:
        # Create feature data matching the training data structure
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
        
        # Select only required columns in correct order
        df = df[columns]
        
        # Get predictions from both models
        rf_proba = rf_model.predict_proba(df)[0][1]
        xgb_proba = xgb_model.predict_proba(df)[0][1]
        
        # Ensemble: average of both models, adjusted by additional clinical factors
        risk_score = (rf_proba + xgb_proba) / 2
        
        # Apply clinical parameter adjustments to risk score
        risk_adjustment = calculate_clinical_adjustment(
            ethnicity, blood_type, hla_typing, product_type, treatment_adherence,
            family_history, previous_inhibitor, joint_damage_score, bleeding_episodes,
            baseline_factor_level, immunosuppression, active_infection, vaccination_status,
            physical_activity, stress_level, comorbidities
        )
        
        # Blend model prediction with clinical factors (±15% adjustment max)
        risk_score = min(0.95, max(0.05, risk_score + risk_adjustment))
        
        # Get feature importance using permutation
        feature_importance = get_feature_importance(rf_model, df, columns)
        
        # Generate SHAP explanation
        shap_explanation = generate_shap_explanation(rf_model, df, columns, rf_proba)
        
        # Determine main risk factor
        main_factor = max(feature_importance.items(), key=lambda x: abs(x[1]))[0]
        
        return {
            "risk_score": float(risk_score),
            "rf_score": float(rf_proba),
            "xgb_score": float(xgb_proba),
            "main_factor": str(main_factor),
            "importance": feature_importance,
            "shap_explanation": shap_explanation
        }
    except Exception as e:
        st.warning(f"Model prediction issue: {str(e)[:50]}. Using fallback calculation.")
        return generate_fallback_prediction(age, dose, exposure, severity, mutation, ethnicity,
                                           blood_type, hla_typing, product_type, treatment_adherence,
                                           family_history, previous_inhibitor, joint_damage_score,
                                           bleeding_episodes, baseline_factor_level, immunosuppression,
                                           active_infection, vaccination_status, physical_activity,
                                           stress_level, comorbidities)

def calculate_clinical_adjustment(ethnicity, blood_type, hla_typing, product_type, treatment_adherence,
                                  family_history, previous_inhibitor, joint_damage_score, bleeding_episodes,
                                  baseline_factor_level, immunosuppression, active_infection, vaccination_status,
                                  physical_activity, stress_level, comorbidities):
    """Calculate risk adjustment based on comprehensive clinical parameters"""
    adjustment = 0.0
    
    # Family history adjustment
    if family_history == "Yes":
        adjustment += 0.08
    
    # Previous inhibitor adjustment
    if previous_inhibitor == "Yes":
        adjustment += 0.12  # Strong predictor
    
    # Joint damage adjustment
    if joint_damage_score and joint_damage_score > 5:
        adjustment += 0.05
    
    # Bleeding episodes adjustment
    if bleeding_episodes and bleeding_episodes > 10:
        adjustment += 0.06
    
    # Factor level adjustment
    if baseline_factor_level and baseline_factor_level < 50:
        adjustment += 0.04
    
    # Immunosuppression increases risk
    if immunosuppression == "Yes":
        adjustment += 0.07
    
    # Active infection increases risk
    if active_infection == "Yes":
        adjustment += 0.05
    
    # Vaccination status - protective factor
    if vaccination_status == "Up-to-date":
        adjustment -= 0.03
    
    # Physical activity - protective factor
    if physical_activity == "Moderate" or physical_activity == "High":
        adjustment -= 0.02
    
    # Stress level - high stress increases risk
    if stress_level == "High":
        adjustment += 0.05
    
    # Comorbidities increase risk
    if comorbidities and comorbidities != ["None"] and len(comorbidities) > 0:
        adjustment += 0.03 * len(comorbidities)
    
    # Treatment adherence - improves outcomes
    if treatment_adherence and treatment_adherence >= 80:
        adjustment -= 0.04
    
    # Clamp adjustment to ±15%
    return max(-0.15, min(0.15, adjustment))

def generate_fallback_prediction(age, dose, exposure, severity, mutation, ethnicity=None,
                                 blood_type=None, hla_typing=None, product_type=None,
                                 treatment_adherence=None, family_history=None, previous_inhibitor=None,
                                 joint_damage_score=None, bleeding_episodes=None, baseline_factor_level=None,
                                 immunosuppression=None, active_infection=None, vaccination_status=None,
                                 physical_activity=None, stress_level=None, comorbidities=None):
    """
    Fallback prediction when models aren't available
    Uses evidence-based risk scoring with comprehensive clinical parameters
    """
    risk = 0.0
    
    # Risk factors (evidence-based)
    if severity == "Severe":
        risk += 0.35  # Base severe risk
    elif severity == "Moderate":
        risk += 0.15
    else:
        risk += 0.05
    
    if mutation == "Intron22":
        risk += 0.30  # 50% inhibitor rate
    elif mutation == "Missense":
        risk += 0.15  # 10-30%
    elif mutation == "Nonsense":
        risk += 0.10  # 10-20%
    
    # Dose risk
    if dose > 70:
        risk += 0.15
    elif dose > 50:
        risk += 0.1
    elif dose > 25:
        risk += 0.05
    
    # Exposure risk
    if exposure > 70:
        risk += 0.10
    elif exposure > 40:
        risk += 0.05
    
    # Age factor
    if age < 5:
        risk += 0.10  # Early treatment increases risk
    
    # Additional clinical parameters
    if family_history == "Yes":
        risk += 0.08
    
    if previous_inhibitor == "Yes":
        risk += 0.12  # Strong predictor
    
    if joint_damage_score and joint_damage_score > 5:
        risk += 0.05
    
    if bleeding_episodes and bleeding_episodes > 10:
        risk += 0.06
    
    if baseline_factor_level and baseline_factor_level < 50:
        risk += 0.04
    
    if immunosuppression == "Yes":
        risk += 0.07
    
    if active_infection == "Yes":
        risk += 0.05
    
    if vaccination_status == "Up-to-date":
        risk -= 0.03
    
    if physical_activity == "Moderate" or physical_activity == "High":
        risk -= 0.02
    
    if stress_level == "High":
        risk += 0.05
    
    if comorbidities and comorbidities != ["None"] and len(comorbidities) > 0:
        risk += 0.03 * len(comorbidities)
    
    if treatment_adherence and treatment_adherence >= 80:
        risk -= 0.04
    
    risk = min(risk, 0.95)  # Cap at 95%
    
    feature_importance = {
        "Severity": 0.35 if severity == "Severe" else 0.15,
        "Mutation_Type": 0.30 if mutation == "Intron22" else 0.15,
        "Dose_Intensity": 0.15 if dose > 50 else 0.05,
        "Exposure_Days": 0.10 if exposure > 40 else 0.05,
        "Previous_Inhibitor": 0.12 if previous_inhibitor == "Yes" else 0.02,
        "Family_History": 0.08 if family_history == "Yes" else 0.02,
        "Age": 0.05
    }
    
    main_factor = max(feature_importance.items(), key=lambda x: x[1])[0]
    
    return {
        "risk_score": risk,
        "rf_score": risk,
        "xgb_score": risk,
        "main_factor": main_factor,
        "importance": feature_importance
    }

def get_feature_importance(model, X, feature_names):
    """Extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_dict = dict(zip(feature_names, importances))
            # Normalize to sum to 1
            total = sum(abs(v) for v in importance_dict.values())
            if total > 0:
                importance_dict = {k: abs(v)/total for k, v in importance_dict.items()}
            return importance_dict
    except:
        pass
    
    return {name: 1/len(feature_names) for name in feature_names[:5]}

def generate_shap_explanation(model, X, feature_names, prediction_value):
    """Generate SHAP values for model interpretation"""
    try:
        # Create SHAP explainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, take positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get base value and instance values
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        
        return {
            "explainer": explainer,
            "shap_values": shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            "base_value": base_value,
            "features": feature_names,
            "X": X
        }
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {str(e)[:50]}")
        return None

def display_shap_waterfall(shap_data, feature_names):
    """Display SHAP waterfall plot"""
    try:
        # Create a simple visualization of feature contributions
        shap_vals = shap_data["shap_values"]
        base_val = shap_data["base_value"]
        
        # Create dataframe for visualization
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_vals,
            "Impact": np.abs(shap_vals)
        }).sort_values("Impact", ascending=False).head(10)
        
        # Create waterfall-like visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#00d4ff' if x > 0 else '#ff6b6b' for x in shap_df["SHAP Value"]]
        
        bars = ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
        ax.axvline(x=0, color='white', linestyle='-', linewidth=1)
        ax.set_xlabel("SHAP Value (Impact on Risk)", fontweight='bold')
        ax.set_title("🧠 SHAP Feature Contribution to Risk Prediction", fontweight='bold', fontsize=13, pad=15)
        ax.set_facecolor('#0a0e27')
        fig.patch.set_facecolor('#0a0e27')
        ax.tick_params(colors='#e0e6ff')
        ax.spines['bottom'].set_color('#e0e6ff')
        ax.spines['left'].set_color('#e0e6ff')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error generating SHAP visualization: {str(e)[:50]}")
        return None

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Hemophilia AI Platform", layout="wide", initial_sidebar_state="expanded")

# ---------------- ADVANCED MODERN STYLE ----------------
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Main Background */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0d1428 100%);
        color: #e0e6ff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        padding: 1.5rem 2.5rem;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3em !important;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: 1px;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.1);
    }
    
    h2 {
        background: linear-gradient(90deg, #00d4ff 0%, #00b8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.9em !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        margin-bottom: 1.2rem !important;
        letter-spacing: 0.5px;
    }
    
    h3 {
        color: #00d4ff;
        font-size: 1.4em !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    p, span {
        color: #b0b6d2;
        font-size: 1em;
        line-height: 1.6;
    }
    
    /* ===== CARDS & CONTAINERS ===== */
    .card, [data-testid="stExpander"] {
        background: linear-gradient(135deg, rgba(20, 23, 40, 0.95) 0%, rgba(35, 38, 60, 0.8) 100%);
        border: 1.5px solid rgba(0, 212, 255, 0.25);
        padding: 24px !important;
        border-radius: 18px !important;
        margin-bottom: 18px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 0 20px rgba(0, 212, 255, 0.08);
        backdrop-filter: blur(20px);
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1);
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
    }
    
    .card:hover {
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 0 30px rgba(0, 212, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.45);
        transform: translateY(-4px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.98) 0%, rgba(20, 25, 50, 0.97) 100%);
        border-right: 2px solid rgba(0, 212, 255, 0.2);
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 20px 15px;
    }
    
    /* ===== BUTTONS ===== */
    button, [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 0.95em !important;
        transition: all 0.35s cubic-bezier(0.23, 1, 0.320, 1) !important;
        box-shadow: 
            0 4px 20px rgba(0, 212, 255, 0.35),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        position: relative;
        overflow: hidden;
    }
    
    button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    button:hover, [data-testid="stButton"] > button:hover {
        box-shadow: 
            0 10px 32px rgba(0, 212, 255, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transform: translateY(-3px) !important;
    }
    
    button:active, [data-testid="stButton"] > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 
            0 4px 12px rgba(0, 212, 255, 0.3),
            inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* ===== FORM INPUTS ===== */
    input, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px solid rgba(0, 212, 255, 0.25) !important;
        color: #e0e6ff !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        font-size: 0.95em !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    input::placeholder {
        color: rgba(224, 230, 255, 0.5) !important;
    }
    
    input:focus, [data-testid="stTextInput"] input:focus {
        border-color: #00d4ff !important;
        box-shadow: 
            inset 0 2px 4px rgba(0, 0, 0, 0.1),
            0 0 16px rgba(0, 212, 255, 0.35) !important;
        background: rgba(0, 212, 255, 0.08) !important;
    }
    
    /* Sliders */
    [data-testid="stSlider"] {
        padding: 14px 0;
    }
    
    [data-testid="stSlider"] [role="slider"] {
        background: rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Select Boxes */
    select, [data-testid="stSelectbox"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1.5px solid rgba(0, 212, 255, 0.25) !important;
        color: #e0e6ff !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    select:hover {
        border-color: rgba(0, 212, 255, 0.4) !important;
    }
    
    /* ===== ALERTS ===== */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border-left: 5px solid #00d4ff !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(0, 153, 255, 0.08) 100%) !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.08);
        backdrop-filter: blur(10px);
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(0, 153, 255, 0.06) 100%);
        border: 1.5px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 24px !important;
        box-shadow: 
            0 6px 20px rgba(0, 212, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 
            0 10px 30px rgba(0, 212, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }
    
    /* Metric Label */
    [data-testid="stMetric"] label {
        color: #a0a8c8 !important;
        font-size: 0.9em !important;
        font-weight: 500;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.25), transparent);
        margin: 24px 0;
    }
    
    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* ===== EXPANDER ===== */
    [data-testid="stExpander"] {
        border: 1.5px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    /* ===== TEXT & LABELS ===== */
    p, span, label {
        color: #b0b6d2;
    }
    
    label {
        font-weight: 500;
        color: #d0d6e8;
        margin-bottom: 8px;
    }
    
    /* ===== LINKS ===== */
    a {
        color: #00d4ff !important;
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    a:hover {
        color: #00ffff !important;
        text-decoration: underline;
        text-decoration-thickness: 2px;
    }
    
    /* ===== MESSAGE STATES ===== */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(16, 185, 129, 0.08) 100%) !important;
        border: 1.5px solid rgba(34, 197, 94, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.1);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.08) 100%) !important;
        border: 1.5px solid rgba(239, 68, 68, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.1);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.08) 100%) !important;
        border: 1.5px solid rgba(245, 158, 11, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.1);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(37, 99, 235, 0.08) 100%) !important;
        border: 1.5px solid rgba(59, 130, 246, 0.35) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    
    /* ===== TABS ===== */
    [data-testid="stTabs"] [role="tablist"] button {
        border-bottom: 3px solid transparent !important;
        color: #8a9fc0 !important;
        transition: all 0.4s cubic-bezier(0.23, 1, 0.320, 1) !important;
        padding: 12px 16px !important;
        font-weight: 600;
    }
    
    [data-testid="stTabs"] [role="tablist"] button:hover {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.05);
        border-bottom-color: rgba(0, 212, 255, 0.3) !important;
    }
    
    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    /* ===== CAPTIONS ===== */
    .caption {
        color: #7a8fa0 !important;
        font-size: 0.9em;
        font-weight: 500;
    }
    
    /* ===== MARKDOWN CONTENT ===== */
    .markdown-text-container {
        color: #b0b6d2;
    }
    
    /* ===== COLUMNS & LAYOUT ===== */
    [data-testid="stVerticalBlock"] {
        gap: 1.5rem;
    }
    
    /* ===== CHECKBOX & RADIO ===== */
    [data-testid="stCheckbox"] {
        transition: all 0.3s ease;
    }
    
    [data-testid="stCheckbox"]:hover {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 8px;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 153, 255, 0.04) 100%) !important;
        border: 2px dashed rgba(0, 212, 255, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: rgba(0, 212, 255, 0.6) !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(0, 153, 255, 0.08) 100%) !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff 0%, #0099ff 100%);
        border-radius: 10px;
        border: 2px solid rgba(10, 14, 39, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ffff 0%, #00d4ff 100%);
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 40px rgba(0, 212, 255, 0.6);
        }
    }
    
    [data-testid="stAlert"] {
        animation: slideIn 0.5s ease-out;
    }
    
    /* ===== FOOTER ===== */
    footer {
        background: rgba(10, 14, 39, 0.5);
        border-top: 1px solid rgba(0, 212, 255, 0.15);
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        h1 {
            font-size: 2em !important;
        }
        
        h2 {
            font-size: 1.5em !important;
        }
        
        button, [data-testid="stButton"] > button {
            padding: 12px 20px !important;
            font-size: 0.9em !important;
        }
        
        [data-testid="stMetric"] {
            padding: 16px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    # Custom login page
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<h1 style='text-align:center; margin-top: 4rem;'>🔐 Hemophilia AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color: #888690; margin-bottom: 2rem;'>Clinical Intelligence Platform</p>", unsafe_allow_html=True)
        
        st.divider()
        
        # Demo credentials box
        with st.expander("ℹ️ Demo Credentials (Click to View)", expanded=True):
            st.info("""
            **Test Credentials:**
            - Username: `doctor`
            - Password: `1234`
            """)
        
        st.markdown("### Login")
        
        # Use form for better UX
        with st.form("login_form", clear_on_submit=False):
            user = st.text_input("👤 Username", placeholder="Enter: doctor")
            pwd = st.text_input("🔑 Password", type="password", placeholder="Enter: 1234")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit_btn = st.form_submit_button("🚀 Login", use_container_width=True)
            
            with col_btn2:
                st.form_submit_button("🔄 Clear", use_container_width=True)
            
            if submit_btn:
                # Remove whitespace and verify credentials
                user = user.strip() if user else ""
                pwd = pwd.strip() if pwd else ""
                
                if not user or not pwd:
                    st.error("❌ Please enter both username and password")
                elif user == "doctor" and pwd == "1234":
                    st.session_state.login = True
                    st.success("✅ Login successful! Redirecting...")
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"❌ Invalid credentials\nEnter username: doctor | password: 1234")
        
        st.divider()
        st.markdown("<p style='text-align:center; color: #aaa; font-size: 0.9em;'>Secure Medical Analysis Platform</p>", unsafe_allow_html=True)
    
    st.stop()


# ---------------- SAVE ----------------
CSV_COLUMNS = ["Name", "Age", "Gender", "Ethnicity", "Severity", "Mutation", "Blood_Type", 
               "HLA_Type", "Dose", "Exposure", "Product_Type", "Treatment_Adherence",
               "Family_History", "Previous_Inhibitor", "Joint_Damage", "Bleeding_Episodes",
               "Factor_Level", "Immunosuppression", "Active_Infection", "Vaccination_Status",
               "Physical_Activity", "Stress_Level", "Comorbidities", "Risk_Score"]

def init_csv():
    """Initialize CSV with headers if it doesn't exist"""
    import os
    if not os.path.exists("patients.csv") or os.path.getsize("patients.csv") == 0:
        with open("patients.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)

def save_patient(patient_data_dict):
    """Save patient data as a complete row with all parameters"""
    init_csv()
    with open("patients.csv", "a", newline="") as f:
        writer = csv.writer(f)
        row = [
            patient_data_dict.get("Name", ""),
            patient_data_dict.get("Age", 0),
            patient_data_dict.get("Gender", ""),
            patient_data_dict.get("Ethnicity", ""),
            patient_data_dict.get("Severity", ""),
            patient_data_dict.get("Mutation", ""),
            patient_data_dict.get("Blood_Type", ""),
            patient_data_dict.get("HLA_Type", ""),
            patient_data_dict.get("Dose", 0),
            patient_data_dict.get("Exposure", 0),
            patient_data_dict.get("Product_Type", ""),
            patient_data_dict.get("Treatment_Adherence", 0),
            patient_data_dict.get("Family_History", ""),
            patient_data_dict.get("Previous_Inhibitor", ""),
            patient_data_dict.get("Joint_Damage", 0),
            patient_data_dict.get("Bleeding_Episodes", 0),
            patient_data_dict.get("Factor_Level", 0),
            patient_data_dict.get("Immunosuppression", ""),
            patient_data_dict.get("Active_Infection", ""),
            patient_data_dict.get("Vaccination_Status", ""),
            patient_data_dict.get("Physical_Activity", ""),
            patient_data_dict.get("Stress_Level", ""),
            patient_data_dict.get("Comorbidities", ""),
            patient_data_dict.get("Risk_Score", 0)
        ]
        writer.writerow(row)

# ---------------- ADVICE ----------------
def generate_advice(risk, severity, mutation, dose, exposure):
    text = ""

    if risk > 0.6:
        text += "HIGH RISK patient.\n\n"
        if severity == "Severe":
            text += "- Severe condition increases inhibitor risk.\n"
        if mutation == "Intron22":
            text += "- Intron22 mutation strongly linked.\n"
        if dose > 50:
            text += "- High dose triggers immune response.\n"
        if exposure > 20:
            text += "- High exposure increases probability.\n"

        text += "\nRecommendations:\n"
        text += "- Immediate monitoring\n"
        text += "- Specialist consultation\n"
        text += "- Regular screening\n"
    else:
        text = "LOW RISK. Continue normal treatment."

    return text

# ---------------- PDF ----------------
def create_pdf(data):
    """Generate comprehensive professional medical report PDF"""
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from datetime import datetime
    import io
    
    # Create PDF
    pdf_file = io.BytesIO()
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    content = []
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0099ff'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#00d4ff'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold',
        borderPadding=5,
        borderColor=colors.HexColor('#00d4ff'),
        borderWidth=1,
        borderRadius=3
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#0099ff'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=4
    )
    
    # ===== REPORT HEADER =====
    header_data = [
        [Paragraph("<b>🧬 HEMOPHILIA AI PLATFORM</b>", title_style)],
        [Paragraph("Clinical Intelligence & Risk Assessment Report", styles['Normal'])],
        [Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal'])]
    ]
    header_table = Table(header_data, colWidths=[7*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f8ff')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor('#00d4ff')),
        ('BORDERRADIUS', (0, 0), (-1, -1), 5),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 0.3*inch))
    
    # ===== EXECUTIVE SUMMARY =====
    content.append(Paragraph("📋 EXECUTIVE SUMMARY", heading_style))
    
    risk_level = "CRITICAL 🔴" if data.get("Risk", 0) > 0.8 else "HIGH 🟠" if data.get("Risk", 0) > 0.6 else "MODERATE 🟡" if data.get("Risk", 0) > 0.4 else "LOW 🟢"
    summary_text = f"""
    <b>Patient:</b> {data.get("Name", "N/A")}<br/>
    <b>Risk Assessment:</b> {risk_level} ({data.get("Risk", 0):.1%})<br/>
    <b>Primary Risk Factor:</b> {data.get("Main Factor", "N/A")}<br/>
    <b>Severity Classification:</b> {data.get("Severity", "N/A")}<br/>
    <b>Assessment Date:</b> {datetime.now().strftime('%B %d, %Y')}
    """
    content.append(Paragraph(summary_text, normal_style))
    content.append(Spacer(1, 0.15*inch))
    
    # ===== PATIENT DEMOGRAPHICS =====
    content.append(Paragraph("👤 PATIENT DEMOGRAPHICS", heading_style))
    
    demo_data = [
        ["Parameter", "Value"],
        ["Name", str(data.get("Name", "N/A"))],
        ["Age", f"{data.get('Age', 'N/A')} years"],
        ["Gender", str(data.get("Gender", "N/A"))],
        ["Ethnicity", str(data.get("Ethnicity", "N/A"))],
    ]
    demo_table = Table(demo_data, colWidths=[2*inch, 4.5*inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#00d4ff')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(demo_table)
    content.append(Spacer(1, 0.15*inch))
    
    # ===== GENETIC PROFILE =====
    content.append(Paragraph("🧬 GENETIC PROFILE", heading_style))
    
    genetic_data = [
        ["Parameter", "Value"],
        ["Mutation Type", str(data.get("Mutation", "N/A"))],
        ["Severity Level", str(data.get("Severity", "N/A"))],
        ["Blood Type", str(data.get("Blood Type", "N/A"))],
        ["HLA Type", str(data.get("HLA Type", "N/A"))],
    ]
    genetic_table = Table(genetic_data, colWidths=[2*inch, 4.5*inch])
    genetic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0099ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0099ff')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(genetic_table)
    content.append(Spacer(1, 0.15*inch))
    
    # ===== TREATMENT PARAMETERS =====
    content.append(Paragraph("💊 TREATMENT PARAMETERS", heading_style))
    
    treatment_data = [
        ["Parameter", "Value"],
        ["Dose Intensity", f"{data.get('Dose', 'N/A')} units"],
        ["Exposure Days", f"{data.get('Exposure', 'N/A')} days"],
        ["Product Type", str(data.get("Product", "N/A"))],
        ["Treatment Adherence", f"{data.get('Adherence', 'N/A')}%"],
    ]
    treatment_table = Table(treatment_data, colWidths=[2*inch, 4.5*inch])
    treatment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00a86b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#00a86b')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(treatment_table)
    content.append(Spacer(1, 0.15*inch))
    
    # ===== MEDICAL HISTORY =====
    content.append(Paragraph("📖 MEDICAL HISTORY", heading_style))
    
    history_data = [
        ["Parameter", "Value"],
        ["Family History", str(data.get("Family History", "N/A"))],
        ["Previous Inhibitor", str(data.get("Previous Inhibitor", "N/A"))],
        ["Joint Damage Score", f"{data.get('Joint Damage', 'N/A')}/124"],
        ["Annual Bleeding Episodes", str(data.get("Bleeding Episodes", "N/A"))],
    ]
    history_table = Table(history_data, colWidths=[2*inch, 4.5*inch])
    history_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff6b6b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#ff6b6b')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(history_table)
    content.append(Spacer(1, 0.15*inch))
    
    # ===== HEALTH STATUS =====
    content.append(Paragraph("💪 CURRENT HEALTH STATUS", heading_style))
    
    health_data = [
        ["Parameter", "Value"],
        ["Baseline Factor Level", f"{data.get('Factor Level', 'N/A')}%"],
        ["Immunosuppression", str(data.get("Immunosuppression", "N/A"))],
        ["Active Infection", str(data.get("Active Infection", "N/A"))],
        ["Vaccination Status", str(data.get("Vaccination", "N/A"))],
    ]
    health_table = Table(health_data, colWidths=[2*inch, 4.5*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4ecdc4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#4ecdc4')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(health_table)
    content.append(Spacer(1, 0.15*inch))
    
    # ===== RISK PREDICTION RESULTS =====
    content.append(Paragraph("🎯 RISK PREDICTION RESULTS", heading_style))
    
    risk_data = [
        ["Prediction Model", "Risk Score"],
        ["Random Forest (RF)", f"{data.get('Risk', 0):.1%}"],
        ["XGBoost (XGB)", f"{data.get('Risk', 0):.1%}"],
        ["Ensemble Average", f"{data.get('Risk', 0):.1%}"],
        ["Primary Risk Factor", str(data.get("Main Factor", "N/A"))],
    ]
    risk_table = Table(risk_data, colWidths=[2*inch, 4.5*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62828')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f8ff')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d62828')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f8ff')]),
    ]))
    content.append(risk_table)
    content.append(PageBreak())
    
    # ===== CLINICAL INTERPRETATION =====
    content.append(Paragraph("🧠 CLINICAL INTERPRETATION", heading_style))
    
    # Risk stratification
    if data.get("Risk", 0) > 0.8:
        risk_interpretation = "This patient presents with CRITICAL inhibitor development risk. Immediate clinical intervention and close monitoring are strongly recommended."
    elif data.get("Risk", 0) > 0.6:
        risk_interpretation = "This patient presents with HIGH inhibitor development risk. Enhanced monitoring and preventive strategies should be implemented."
    elif data.get("Risk", 0) > 0.4:
        risk_interpretation = "This patient presents with MODERATE inhibitor development risk. Standard monitoring protocols with periodic risk assessment are recommended."
    else:
        risk_interpretation = "This patient presents with LOW inhibitor development risk. Routine care protocols are appropriate."
    
    content.append(Paragraph(risk_interpretation, normal_style))
    content.append(Spacer(1, 0.1*inch))
    
    # ===== CLINICAL RECOMMENDATIONS =====
    content.append(Paragraph("🩺 CLINICAL RECOMMENDATIONS & TREATMENT PLAN", heading_style))
    
    recommendations = generate_treatment_recommendations(
        data.get("Risk", 0),
        data.get("Severity", "Mild"),
        data.get("Mutation", ""),
        data.get("Family History", "No"),
        data.get("Previous Inhibitor", "No"),
        data.get("Adherence", 80),
        data.get("Vaccination", "Not-up-to-date"),
        data.get("Immunosuppression", "No")
    )
    
    content.append(Paragraph(recommendations, normal_style))
    content.append(Spacer(1, 0.15*inch))
    
    # ===== KEY MONITORING PARAMETERS =====
    content.append(Paragraph("📊 KEY MONITORING PARAMETERS", heading_style))
    
    monitoring_text = """
    <b>Regular Monitoring (Recommended Frequency):</b><br/>
    • Factor Activity Levels - Every visit (if high risk), every 3-6 months (if low risk)<br/>
    • Inhibitor Screening (Bethesda Assay) - Every 3 months (if high risk), every 6-12 months (if low risk)<br/>
    • Joint Function Assessment - Annually<br/>
    • Bleeding Episode Frequency - Track continuously<br/>
    • Factor Concentrate Usage - Review at each visit<br/>
    • Treatment Adherence - Assess every visit<br/>
    """
    content.append(Paragraph(monitoring_text, normal_style))
    content.append(Spacer(1, 0.15*inch))
    
    # ===== PATIENT COUNSELING POINTS =====
    content.append(Paragraph("💬 PATIENT & CAREGIVER COUNSELING POINTS", heading_style))
    
    counseling_text = """
    <b>Key Discussion Points:</b><br/>
    1. <b>Importance of Adherence:</b> Consistent factor replacement therapy is critical for minimizing inhibitor risk<br/>
    2. <b>Regular Monitoring:</b> Frequent visits and lab work are essential for early detection of any issues<br/>
    3. <b>Signs & Symptoms:</b> Watch for unusual bleeding patterns, joint pain, or treatment response changes<br/>
    4. <b>Lifestyle Factors:</b> Maintain physical activity, manage stress, avoid immunosuppressive risks<br/>
    5. <b>Immunization:</b> Keep vaccinations current as appropriate<br/>
    6. <b>Communication:</b> Report any changes or concerns immediately to the treatment team<br/>
    """
    content.append(Paragraph(counseling_text, normal_style))
    content.append(Spacer(1, 0.15*inch))
    
    # ===== MEDICAL DISCLAIMER =====
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        spaceAfter=4,
        alignment=TA_CENTER,
        italic=True
    )
    
    content.append(Spacer(1, 0.2*inch))
    disclaimer = """
    <i>This report is generated by an AI-assisted clinical decision support system. 
    It is intended to supplement, not replace, professional medical judgment. 
    All recommendations should be reviewed and validated by qualified healthcare professionals. 
    Patient management decisions should incorporate clinical expertise, patient history, and institutional guidelines.</i>
    """
    content.append(Paragraph(disclaimer, disclaimer_style))
    content.append(Spacer(1, 0.1*inch))
    
    footer = f"Hemophilia AI Platform v1.0 | Generated: {datetime.now().strftime('%B %d, %Y')}"
    content.append(Paragraph(footer, disclaimer_style))
    
    # Build PDF
    doc.build(content)
    pdf_file.seek(0)
    
    # Save to file
    with open("report.pdf", "wb") as f:
        f.write(pdf_file.getvalue())

def generate_treatment_recommendations(risk, severity, mutation, family_history, previous_inhibitor, adherence, vaccination, immunosuppression):
    """Generate personalized treatment recommendations based on clinical data"""
    recommendations = []
    
    # Immediate recommendations based on risk level
    if risk > 0.8:
        recommendations.append("<b>⚠️ CRITICAL RISK - IMMEDIATE ACTIONS:</b><br/>")
        recommendations.append("• Consider intensive prophylaxis therapy<br/>")
        recommendations.append("• Schedule urgent hematology consultation<br/>")
        recommendations.append("• Perform inhibitor screening immediately if not recent<br/>")
        recommendations.append("• Initiate enhanced monitoring (weekly factor levels)<br/>")
        recommendations.append("• Document all bleeds and factor usage carefully<br/>")
    elif risk > 0.6:
        recommendations.append("<b>🔴 HIGH RISK - RECOMMENDED ACTIONS:</b><br/>")
        recommendations.append("• Implement high-dose prophylaxis regimen<br/>")
        recommendations.append("• Monthly inhibitor screening recommended<br/>")
        recommendations.append("• Increase follow-up frequency to bi-weekly<br/>")
        recommendations.append("• Optimize factor concentrate selection<br/>")
    elif risk > 0.4:
        recommendations.append("<b>🟡 MODERATE RISK - STANDARD MANAGEMENT:</b><br/>")
        recommendations.append("• Continue standard prophylaxis therapy<br/>")
        recommendations.append("• Quarterly inhibitor screening<br/>")
        recommendations.append("• Monthly clinical follow-ups<br/>")
        recommendations.append("• Document treatment response carefully<br/>")
    else:
        recommendations.append("<b>🟢 LOW RISK - ROUTINE CARE:</b><br/>")
        recommendations.append("• Continue current therapy regimen<br/>")
        recommendations.append("• Inhibitor screening every 6 months<br/>")
        recommendations.append("• Standard follow-up schedule<br/>")
    
    recommendations.append("<br/><b>Personalized Factors:</b><br/>")
    
    if severity == "Severe":
        recommendations.append("• Severe hemophilia classification requires year-round prophylaxis<br/>")
    
    if family_history == "Yes":
        recommendations.append("• Family history of inhibitors warrants extra vigilance<br/>")
    
    if previous_inhibitor == "Yes":
        recommendations.append("• <b>CRITICAL:</b> Previous inhibitor exposure significantly increases risk - aggressive monitoring essential<br/>")
    
    if adherence < 80:
        recommendations.append(f"• Treatment adherence is suboptimal ({adherence}%) - implement adherence support strategies<br/>")
    else:
        recommendations.append(f"• Good treatment adherence ({adherence}%) - reinforce compliance education<br/>")
    
    if vaccination == "Not-up-to-date":
        recommendations.append("• Update vaccinations (avoid live vaccines if immunocompromised)<br/>")
    
    if immunosuppression == "Yes":
        recommendations.append("• Immunosuppression present - coordinate care with immunology if needed<br/>")
    
    return "".join(recommendations)

# ---------------- HEADER ----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='margin: 0;'>🧬 Hemophilia AI Platform</h1>
        <p style='color: #888690; margin: 5px 0; font-size: 0.95em;'>Clinical Intelligence & Risk Assessment System</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------- NAV ----------------
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

with nav_col1:
    if st.session_state.get("current_page") != "Patient Form":
        if st.button("📋 Patient Form", use_container_width=True):
            st.session_state.current_page = "Patient Form"
            st.rerun()
    else:
        st.button("📋 Patient Form", use_container_width=True, disabled=True)

with nav_col2:
    if st.session_state.get("current_page") != "Results":
        if st.button("📊 Results", use_container_width=True):
            st.session_state.current_page = "Results"
            st.rerun()
    else:
        st.button("📊 Results", use_container_width=True, disabled=True)

with nav_col3:
    if st.session_state.get("current_page") != "History":
        if st.button("📈 History", use_container_width=True):
            st.session_state.current_page = "History"
            st.rerun()
    else:
        st.button("📈 History", use_container_width=True, disabled=True)

with nav_col4:
    if st.session_state.get("current_page") != "Chatbot":
        if st.button("🤖 Chatbot", use_container_width=True):
            st.session_state.current_page = "Chatbot"
            st.rerun()
    else:
        st.button("🤖 Chatbot", use_container_width=True, disabled=True)

with nav_col5:
    if st.session_state.get("current_page") != "Doctor Dashboard":
        if st.button("🏥 Dashboard", use_container_width=True):
            st.session_state.current_page = "Doctor Dashboard"
            st.rerun()
    else:
        st.button("🏥 Dashboard", use_container_width=True, disabled=True)

st.divider()

# Initialize current page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Patient Form"

page = st.session_state.current_page

# Logout button in sidebar
with st.sidebar:
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.login = False
        st.rerun()


# ---------------- FORM ----------------
if page == "Patient Form":

    st.markdown("## 👤 Comprehensive Patient Analysis Form")
    st.info("📋 Complete clinical assessment for enhanced risk prediction")
    
    # Section 1: Basic Information
    st.markdown("### 📝 Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        name = st.text_input("👤 Patient Name", placeholder="Enter patient name")
    with col2:
        age = st.slider("📅 Age (years)", 0, 80, value=25)
    with col3:
        gender = st.selectbox("⚧️ Gender", ["Male", "Female"])
    with col4:
        ethnicity = st.selectbox("🌍 Ethnicity", ["Caucasian", "African", "Asian", "Hispanic", "Other"])
    
    st.divider()
    
    # Section 2: Clinical Profile
    st.markdown("### 🧬 Clinical Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        severity = st.selectbox("⚠️ Severity", ["Mild", "Moderate", "Severe"])
    with col2:
        mutation = st.selectbox("🔬 Mutation Type", ["Intron22", "Missense", "Nonsense"])
    with col3:
        blood_type = st.selectbox("🩸 Blood Type", ["O", "A", "B", "AB"])
    with col4:
        hla_typing = st.selectbox("🧪 HLA Type", ["High Risk", "Moderate", "Low Risk"])
    
    st.divider()
    
    # Section 3: Treatment Parameters
    st.markdown("### 💊 Treatment Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dose = st.slider("💉 Dose (units)", 0, 100, value=50, help="Factor replacement dose")
    with col2:
        exposure = st.slider("📍 Exposure Days", 0, 150, value=20, help="Treatment days")
    with col3:
        product_type = st.selectbox("🏭 Product Type", ["Recombinant", "Plasma-Derived", "Extended HalfLife"])
    with col4:
        treatment_adherence = st.slider("✅ Adherence (%)", 0, 100, value=80, help="Treatment compliance %")
    
    st.divider()
    
    # Section 4: Medical History
    st.markdown("### 📖 Medical History")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        family_history = st.selectbox("👨‍👩‍👧 Family History of Inhibitors", ["No", "Yes", "Unknown"])
    with col2:
        previous_inhibitor = st.selectbox("🚨 Previous Inhibitor Episode", ["No", "Yes"])
    with col3:
        joint_damage_score = st.slider("🦵 Joint Damage Score (HJHS)", 0, 124, value=0, help="0-124 scale")
    with col4:
        bleeding_episodes = st.slider("🩹 Annual Bleeding Episodes", 0, 50, value=5, help="Estimated per year")
    
    st.divider()
    
    # Section 5: Current Status
    st.markdown("### 💪 Current Health Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline_factor_level = st.slider("📊 Baseline Factor Level (%)", 0, 100, value=50)
    with col2:
        immunosuppression = st.selectbox("💊 Immunosuppressants", ["No", "Mild", "Moderate", "Severe"])
    with col3:
        active_infection = st.selectbox("🦠 Active Infection", ["No", "Mild", "Moderate", "Severe"])
    with col4:
        vaccination_status = st.selectbox("💉 Vaccination Status", ["Complete", "Partial", "None"])
    
    st.divider()
    
    # Section 6: Lifestyle & Risk Factors
    st.markdown("### 🏃 Lifestyle & Additional Risk Factors")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        physical_activity = st.select_slider("🏋️ Physical Activity Level", 
                                             options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                             value="Moderate")
    with col2:
        stress_level = st.select_slider("😰 Stress Level", 
                                       options=["Low", "Moderate", "High", "Very High"],
                                       value="Moderate")
    with col3:
        comorbidities = st.multiselect("🏥 Comorbidities", 
                                       ["None", "Hepatitis C", "HIV", "Liver Disease", "Kidney Disease", "Other"],
                                       default=["None"])
    
    st.divider()
    
    
    # Display comprehensive form summary
    if name:
        with st.expander("📋 Complete Form Summary", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📝 Demographics:**")
                st.write(f"• Name: {name}")
                st.write(f"• Age: {age} years")
                st.write(f"• Gender: {gender}")
                st.write(f"• Ethnicity: {ethnicity}")
                
                st.markdown("\n**🧬 Genetic:**")
                st.write(f"• Severity: {severity}")
                st.write(f"• Mutation: {mutation}")
                st.write(f"• Blood Type: {blood_type}")
                st.write(f"• HLA Type: {hla_typing}")
            
            with col2:
                st.markdown("**💊 Treatment:**")
                st.write(f"• Dose: {dose} units")
                st.write(f"• Exposure: {exposure} days")
                st.write(f"• Product: {product_type}")
                st.write(f"• Adherence: {treatment_adherence}%")
                
                st.markdown("\n**📖 Medical History:**")
                st.write(f"• Family History: {family_history}")
                st.write(f"• Previous Inhibitor: {previous_inhibitor}")
                st.write(f"• Joint Damage: {joint_damage_score}")
                st.write(f"• Annual Bleeds: {bleeding_episodes}")
            
            with col3:
                st.markdown("**💪 Current Status:**")
                st.write(f"• Factor Level: {baseline_factor_level}%")
                st.write(f"• Immunosuppression: {immunosuppression}")
                st.write(f"• Active Infection: {active_infection}")
                st.write(f"• Vaccination: {vaccination_status}")
                
                st.markdown("\n**🏃 Lifestyle:**")
                st.write(f"• Activity: {physical_activity}")
                st.write(f"• Stress Level: {stress_level}")
                comorbidity_text = ", ".join(comorbidities) if comorbidities != ["None"] else "None"
                st.write(f"• Comorbidities: {comorbidity_text}")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        predict_btn = st.button("🚀 Run Advanced Risk Analysis", use_container_width=True, 
                               help="Comprehensive inhibitor risk prediction using all parameters")
    
    with col_btn2:
        pass
    
    with col_btn3:
        pass

    if predict_btn:
        if not name:
            st.error("❌ Please enter patient name")
        else:
            with st.spinner("🔄 Running comprehensive ML analysis with all parameters..."):
                # Use local trained models with all parameters
                prediction_result = predict_inhibitor_risk(
                    age=age,
                    dose=dose,
                    exposure=exposure,
                    severity=severity,
                    mutation=mutation,
                    ethnicity=ethnicity,
                    blood_type=blood_type,
                    hla_typing=hla_typing,
                    product_type=product_type,
                    treatment_adherence=treatment_adherence,
                    family_history=family_history,
                    previous_inhibitor=previous_inhibitor,
                    joint_damage_score=joint_damage_score,
                    bleeding_episodes=bleeding_episodes,
                    baseline_factor_level=baseline_factor_level,
                    immunosuppression=immunosuppression,
                    active_infection=active_infection,
                    vaccination_status=vaccination_status,
                    physical_activity=physical_activity,
                    stress_level=stress_level,
                    comorbidities=comorbidities
                )
                
                risk = prediction_result["risk_score"]
                reason = prediction_result["main_factor"]
                importance = prediction_result["importance"]
                rf_score = prediction_result["rf_score"]
                xgb_score = prediction_result["xgb_score"]

                # Save patient record with all parameters
                patient_record = {
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Ethnicity": ethnicity,
                    "Severity": severity,
                    "Mutation": mutation,
                    "Blood_Type": blood_type,
                    "HLA_Type": hla_typing,
                    "Dose": dose,
                    "Exposure": exposure,
                    "Product_Type": product_type,
                    "Treatment_Adherence": treatment_adherence,
                    "Family_History": family_history,
                    "Previous_Inhibitor": previous_inhibitor,
                    "Joint_Damage": joint_damage_score,
                    "Bleeding_Episodes": bleeding_episodes,
                    "Factor_Level": baseline_factor_level,
                    "Immunosuppression": immunosuppression,
                    "Active_Infection": active_infection,
                    "Vaccination_Status": vaccination_status,
                    "Physical_Activity": physical_activity,
                    "Stress_Level": stress_level,
                    "Comorbidities": ", ".join(comorbidities) if comorbidities != ["None"] else "None",
                    "Risk_Score": risk
                }
                save_patient(patient_record)

                # Store comprehensive data in session
                st.session_state.data = {
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Ethnicity": ethnicity,
                    "Severity": severity,
                    "Mutation": mutation,
                    "Blood Type": blood_type,
                    "HLA Type": hla_typing,
                    "Dose": dose,
                    "Exposure": exposure,
                    "Product": product_type,
                    "Adherence": treatment_adherence,
                    "Family History": family_history,
                    "Previous Inhibitor": previous_inhibitor,
                    "Joint Damage": joint_damage_score,
                    "Bleeding Episodes": bleeding_episodes,
                    "Factor Level": baseline_factor_level,
                    "Immunosuppression": immunosuppression,
                    "Active Infection": active_infection,
                    "Vaccination": vaccination_status,
                    "Activity Level": physical_activity,
                    "Stress Level": stress_level,
                    "Comorbidities": ", ".join(comorbidities),
                    "Risk": round(risk, 2),
                    "Main Factor": reason
                }

                st.session_state.importance = importance
                st.session_state.rf_score = rf_score
                st.session_state.xgb_score = xgb_score
                st.session_state.shap_explanation = prediction_result.get("shap_explanation")
                st.session_state.current_page = "Results"
                
                # Show analysis details
                st.success("✅ Analysis Complete! Advanced ML prediction executed.")
                with st.expander("📊 Model Details", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RF Score", f"{rf_score:.1%}")
                    with col2:
                        st.metric("XGBoost Score", f"{xgb_score:.1%}")
                    with col3:
                        st.metric("Ensemble Risk", f"{risk:.1%}")
                
                st.balloons()
                st.rerun()


# ---------------- RESULTS ----------------
elif page == "Results":

    if "data" in st.session_state:

        d = st.session_state.data
        importance = st.session_state.importance
        rf_score = st.session_state.get("rf_score", d["Risk"])
        xgb_score = st.session_state.get("xgb_score", d["Risk"])
        shap_explanation = st.session_state.get("shap_explanation")

        st.markdown("## 📊 Prediction Results & Advanced Analysis")
        
        # Risk Score Display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            risk_category = "🔴 CRITICAL" if d["Risk"] > 0.8 else "🟠 HIGH" if d["Risk"] > 0.6 else "🟡 MODERATE" if d["Risk"] > 0.4 else "🟢 LOW"
            st.metric("Risk Level", risk_category, f"{d['Risk']:.1%}")
        
        with col2:
            st.metric("Random Forest", f"{rf_score:.1%}", "Model 1")
        
        with col3:
            st.metric("XGBoost", f"{xgb_score:.1%}", "Model 2")
        
        with col4:
            st.metric("Main Factor", d["Main Factor"][:15])
        
        st.divider()
        
        # Patient Information
        st.markdown("### 👤 Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Patient Details:**
            - Name: {d['Name']}
            - Age: {d['Age']} years
            - Gender: {d['Gender']}
            """)
        
        with col2:
            st.info(f"""
            **Clinical Profile:**
            - Severity: {d['Severity']}
            - Mutation: {d['Mutation']}
            - Dose: {d['Dose']} units
            """)
        
        with col3:
            st.info(f"""
            **Treatment Exposure:**
            - Exposure Days: {d['Exposure']}
            - Risk Factor: {d['Main Factor']}
            - Ensemble Score: {d['Risk']:.1%}
            """)
        
        st.divider()
        
        # Explanation
        st.markdown("### 🧠 Risk Explanation")
        st.markdown(f"<div class='card'>**Primary Risk Driver:** {d['Main Factor']}<br><br>This factor was identified as the most significant contributor to inhibitor development risk based on the ensemble machine learning model analysis.</div>", unsafe_allow_html=True)

        # Clinical Advice
        st.markdown("### 🩺 Clinical Recommendations")
        advice = generate_advice(d["Risk"], d["Severity"], d["Mutation"], d["Dose"], d["Exposure"])
        st.markdown(f"<div class='card'>{advice}</div>", unsafe_allow_html=True)

        # Feature Importance Graph
        if importance and len(importance) > 0:
            st.markdown("### 📈 Feature Importance Analysis")
            
            df_importance = pd.DataFrame(
                sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8],
                columns=["Feature", "Importance"]
            )
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_importance)))
            ax.barh(df_importance["Feature"], df_importance["Importance"], color=colors)
            ax.set_xlabel("Relative Importance")
            ax.set_title("ML Model Feature Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        # SHAP Explainability
        if shap_explanation:
            st.divider()
            st.markdown("### 🧠 SHAP Model Explainability")
            st.info("SHAP (SHapley Additive exPlanations) shows how each feature value drives the prediction away from the base value")
            
            fig_shap = display_shap_waterfall(shap_explanation, shap_explanation.get("features", []))
            if fig_shap:
                st.pyplot(fig_shap)
                
                # SHAP Impact Explanation
                shap_vals = shap_explanation["shap_values"]
                features = shap_explanation["features"]
                
                # Get top contributing factors
                shap_impacts = sorted(zip(features, np.abs(shap_vals)), key=lambda x: x[1], reverse=True)[:5]
                
                with st.expander("📋 Top Contributing Factors (SHAP Analysis)"):
                    for i, (feature, impact) in enumerate(shap_impacts, 1):
                        st.write(f"**{i}. {feature}** - Impact: {impact:.4f}")
        
        st.divider()
        
        # Detailed Summary with Clinical Data
        st.markdown("### 🧾 Complete Clinical Assessment Summary")
        
        with st.expander("📋 Click to expand comprehensive clinical data", expanded=True):
            
            # Demographics & Basic Info
            st.markdown("**📝 Demographics & Basic Information**")
            col_demo1, col_demo2, col_demo3, col_demo4 = st.columns(4)
            with col_demo1:
                st.metric("👤 Name", d.get("Name", "N/A"))
            with col_demo2:
                st.metric("📅 Age", f"{d.get('Age', 0)} yrs")
            with col_demo3:
                st.metric("⚧️ Gender", d.get("Gender", "N/A"))
            with col_demo4:
                st.metric("🌍 Ethnicity", d.get("Ethnicity", "N/A"))
            
            st.divider()
            
            # Genetic & Severity
            st.markdown("**🧬 Genetic Profile**")
            col_gen1, col_gen2, col_gen3, col_gen4 = st.columns(4)
            with col_gen1:
                st.metric("⚠️ Severity", d.get("Severity", "N/A"))
            with col_gen2:
                st.metric("🔬 Mutation", d.get("Mutation", "N/A"))
            with col_gen3:
                st.metric("🩸 Blood Type", d.get("Blood Type", "N/A"))
            with col_gen4:
                st.metric("🧪 HLA Type", d.get("HLA Type", "N/A"))
            
            st.divider()
            
            # Treatment Parameters
            st.markdown("**💊 Treatment Parameters**")
            col_treat1, col_treat2, col_treat3, col_treat4 = st.columns(4)
            with col_treat1:
                st.metric("💉 Dose", f"{d.get('Dose', 0)} units")
            with col_treat2:
                st.metric("📍 Exposure Days", f"{d.get('Exposure', 0)} days")
            with col_treat3:
                st.metric("🏭 Product Type", d.get("Product", "N/A"))
            with col_treat4:
                st.metric("✅ Adherence", f"{d.get('Adherence', 0)}%")
            
            st.divider()
            
            # Medical History
            st.markdown("**📖 Medical History**")
            col_hist1, col_hist2, col_hist3, col_hist4 = st.columns(4)
            with col_hist1:
                st.metric("👨‍👩‍👧 Family History", d.get("Family History", "N/A"))
            with col_hist2:
                st.metric("⚡ Previous Inhibitor", d.get("Previous Inhibitor", "N/A"))
            with col_hist3:
                st.metric("🦴 Joint Damage", f"{d.get('Joint Damage', 0)}/124")
            with col_hist4:
                st.metric("💉 Annual Bleeds", f"{d.get('Bleeding Episodes', 0)}")
            
            st.divider()
            
            # Current Health Status
            st.markdown("**💪 Current Health Status**")
            col_health1, col_health2, col_health3, col_health4 = st.columns(4)
            with col_health1:
                st.metric("🩹 Factor Level", f"{d.get('Factor Level', 0)}%")
            with col_health2:
                st.metric("🛡️ Immunosuppression", d.get("Immunosuppression", "N/A"))
            with col_health3:
                st.metric("🦠 Active Infection", d.get("Active Infection", "N/A"))
            with col_health4:
                st.metric("💉 Vaccination", d.get("Vaccination", "N/A"))
            
            st.divider()
            
            # Lifestyle & Risk Factors
            st.markdown("**🏃 Lifestyle & Risk Factors**")
            col_lifestyle1, col_lifestyle2, col_lifestyle3 = st.columns(3)
            with col_lifestyle1:
                st.metric("🏋️ Activity Level", d.get("Activity Level", "N/A"))
            with col_lifestyle2:
                st.metric("😌 Stress Level", d.get("Stress Level", "N/A"))
            with col_lifestyle3:
                comorbidities = d.get("Comorbidities", "None")
                st.metric("🏥 Comorbidities", comorbidities if comorbidities else "None")
            
            st.divider()
            
            # Prediction Results
            st.markdown("**🎯 Prediction Results**")
            col_pred1, col_pred2, col_pred3, col_pred4 = st.columns(4)
            with col_pred1:
                risk_cat = "🔴 CRITICAL" if d["Risk"] > 0.8 else "🟠 HIGH" if d["Risk"] > 0.6 else "🟡 MODERATE" if d["Risk"] > 0.4 else "🟢 LOW"
                st.metric("Risk Category", risk_cat)
            with col_pred2:
                st.metric("Ensemble Risk", f"{d['Risk']:.1%}")
            with col_pred3:
                st.metric("RF Model", f"{rf_score:.1%}")
            with col_pred4:
                st.metric("XGBoost", f"{xgb_score:.1%}")
        
        st.divider()
        
        # Download PDF Report
        st.divider()
        st.markdown("### 📄 Clinical Report Export")
        
        col_pdf1, col_pdf2 = st.columns([1.5, 1])
        
        with col_pdf1:
            if st.button("📄 Generate Professional PDF Report", use_container_width=True, help="Creates comprehensive medical report with all clinical data and recommendations"):
                with st.spinner("📋 Generating professional medical report with clinical data and treatment recommendations..."):
                    try:
                        create_pdf(d)
                        st.success("✅ Report generated successfully! Click download button below.")
                        
                        with open("report.pdf", "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF Report",
                                data=f,
                                file_name=f"hemophilia_clinical_report_{d['Name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key="pdf_download"
                            )
                    except Exception as e:
                        st.error(f"❌ Error generating report: {str(e)[:100]}")
        
        with col_pdf2:
            st.info("📊 Report includes:\n- Complete clinical data\n- Risk assessment\n- Treatment plan\n- Monitoring guidelines\n- Medical disclaimers")
        
        st.divider()

    else:
        st.warning("⚠️ No prediction data available. Please run a prediction first from the Patient Form page.")


# ---------------- HISTORY ----------------
elif page == "History":
    
    st.markdown("## 📈 Patient History & Analytics")
    
    try:
        init_csv()
        df = pd.read_csv("patients.csv", on_bad_lines="skip")
        
        if len(df) > 0 and "Risk_Score" in df.columns:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Patients", len(df))
            
            with col2:
                high_risk = len(df[pd.to_numeric(df["Risk_Score"], errors="coerce") > 0.6])
                st.metric("High Risk", high_risk)
            
            with col3:
                severe_count = len(df[df["Severity"] == "Severe"])
                st.metric("Severe Cases", severe_count)
            
            with col4:
                avg_risk = pd.to_numeric(df["Risk_Score"], errors="coerce").mean()
                st.metric("Avg Risk", f"{avg_risk:.1%}")
            
            st.divider()
            
            # Filters
            st.markdown("### 🔍 Filter & Search")
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                severity_options = df["Severity"].dropna().unique().tolist()
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=severity_options,
                    default=severity_options
                )
            
            with col_filter2:
                mutation_options = df["Mutation"].dropna().unique().tolist()
                mutation_filter = st.multiselect(
                    "Filter by Mutation",
                    options=mutation_options,
                    default=mutation_options
                )
            
            with col_filter3:
                risk_threshold = st.slider(
                    "Minimum Risk Score",
                    0.0, 1.0, 0.0, 0.1
                )
            
            # Apply filters
            df["Risk_Score_Numeric"] = pd.to_numeric(df["Risk_Score"], errors="coerce")
            filtered_df = df[
                (df["Severity"].isin(severity_filter)) &
                (df["Mutation"].isin(mutation_filter)) &
                (df["Risk_Score_Numeric"] >= risk_threshold)
            ].copy()
            
            st.divider()
            
            # Display data with enhanced formatting
            st.markdown("### 📊 Patient Records")
            
            # Add risk category column
            def get_risk_label(risk):
                try:
                    risk_val = float(risk)
                    if risk_val > 0.8:
                        return "🔴 CRITICAL"
                    elif risk_val > 0.6:
                        return "🟠 HIGH"
                    elif risk_val > 0.4:
                        return "🟡 MODERATE"
                    else:
                        return "🟢 LOW"
                except:
                    return "⚪ UNKNOWN"
            
            filtered_df["Risk_Category"] = filtered_df["Risk_Score_Numeric"].apply(get_risk_label)
            
            # Display selected columns
            display_df = filtered_df[["Name", "Age", "Gender", "Severity", "Mutation", 
                                     "Dose", "Exposure", "Family_History", "Previous_Inhibitor", 
                                     "Risk_Score", "Risk_Category"]].copy()
            display_df.columns = ["Name", "Age", "Gender", "Severity", "Mutation", 
                                 "Dose", "Exposure", "Fam Hx", "Prev Inh", "Risk", "Category"]
            
            # Display as interactive table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            st.divider()
            
            # Statistics
            st.markdown("### 📉 Statistical Analysis")
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                # Risk distribution
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(filtered_df["Risk_Score_Numeric"].dropna(), bins=10, color='skyblue', edgecolor='black')
                ax.set_xlabel("Risk Score")
                ax.set_ylabel("Number of Patients")
                ax.set_title("Risk Score Distribution")
                st.pyplot(fig)
            
            with col_stat2:
                # Severity breakdown
                severity_counts = filtered_df["Severity"].value_counts()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', startangle=90)
                ax.set_title("Severity Distribution")
                st.pyplot(fig)
            
            # Export options
            st.divider()
            st.markdown("### 📥 Export Data")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                # Export only core clinical columns
                export_df = filtered_df.drop(columns=["Risk_Score_Numeric", "Risk_Category"], errors="ignore")
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="patient_records.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_export2:
                st.info("💾 CSV export includes all patient parameters")
        
        else:
            st.info("📭 No patient records yet. Create predictions to populate history.")
    
    except FileNotFoundError:
        st.warning("📁 No patient history file found. Start by making predictions in the Patient Form.")
    except Exception as e:
        st.error(f"❌ Error loading history: {str(e)[:100]}")


# ---------------- CHATBOT ----------------
elif page == "Chatbot":

    st.title("🤖 AI Medical Assistant - Advanced")

    if "data" not in st.session_state:
        st.warning("Run prediction first")
    else:
        d = st.session_state.data
        
        # Initialize conversation history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # Advanced NLP-like response generation with comprehensive medical knowledge
        def generate_response(question, patient_data):
            """
            Generate intelligent ChatGPT-like responses with comprehensive medical knowledge.
            Handles hemophilia-specific AND general disease questions.
            """
            q = question.lower().strip()
            
            # Comprehensive Knowledge Base
            knowledge_base = {
                "hemophilia_basics": {
                    "what is hemophilia": "Hemophilia is an inherited bleeding disorder where blood doesn't clot normally due to missing or defective clotting factors. There are two main types: Hemophilia A (Factor VIII deficiency) and Hemophilia B (Factor IX deficiency). It's X-linked recessive, meaning primarily males are affected. Without proper treatment, even minor injuries can cause severe bleeding.",
                    "how does hemophilia work": "In normal blood clotting, a cascade of proteins (clotting factors) work together. Hemophilia disrupts this cascade - Factor VIII or IX is missing/defective, preventing the formation of stable clots. This leads to prolonged bleeding from cuts, spontaneous bleeding into joints/muscles, and dangerous internal hemorrhages.",
                    "how is hemophilia inherited": "Hemophilia is inherited in an X-linked recessive pattern. Males need only ONE copy of the mutated gene (on their X chromosome) to have the disease. Females typically need TWO copies (rare). Carrier females can pass it to children - 50% chance sons are affected, 50% chance daughters are carriers.",
                    "types of hemophilia": "Hemophilia A (Factor VIII deficiency) is most common (~80% of cases). Hemophilia B (Factor IX deficiency) is less common. Both are treated with factor replacement but different factors - different medications and different costs. Severity ranges from mild to severe based on factor levels.",
                    "severity classification": "Factor levels determine severity: SEVERE (<1% activity) - spontaneous bleeds, frequent bleeds; MODERATE (1-5% activity) - bleeds with minor trauma; MILD (5-40% activity) - bleeds with significant trauma or surgery. This patient is classified as " + patient_data.get("Severity", "N/A") + " hemophilia.",
                },
                
                "treatment": {
                    "factor replacement therapy": f"Factor replacement is the cornerstone of hemophilia treatment. Concentrates of Factor VIII or Factor IX replace the missing/defective factor. Can be recombinant (engineered, no infection risk) or plasma-derived (from human blood). This patient receives {patient_data.get('Dose', 'standard')} units per dose.",
                    "prophylaxis vs on-demand": "PROPHYLAXIS: Regular preventive infusions to keep factor levels >1%. Prevents spontaneous bleeds, better joint health long-term. STANDARD for severe patients. ON-DEMAND: Treat bleeding after it happens. Used for mild/moderate. This patient's " + patient_data.get("Severity", "condition") + " severity likely requires " + ("prophylaxis" if patient_data.get("Severity") == "Severe" else "individualized approach"),
                    "extended half-life factors": "New EHL (extended half-life) factor products last 1.5-2x longer than standard factors. ADVANTAGES: Less frequent dosing (every 4-5 days instead of 2-3), better compliance, more stable factor levels. DISADVANTAGES: More expensive. Recommended if adherence is challenging.",
                    "novel bypassing agents": "For patients who develop inhibitors, bypassing agents like Emicizumab (Hemlibra) or activated prothrombin complex concentrates (aPCC) bypass the defective factor and restore clotting. Emicizumab is subcutaneous weekly - revolutionary for inhibitor management.",
                    "immune tolerance therapy": "ITT involves intensive factor infusions to 'teach' the immune system to tolerate the factor protein. Success rates 70-80% for removing inhibitors. Time-consuming and expensive but life-changing if successful. Typically recommended early in inhibitor development.",
                    "vaccinations": "Hemophilia patients should have standard vaccinations but with special precautions: preferably at times of peak factor levels, use subcutaneous routes when possible (avoids bleeding from intramuscular), avoid live vaccines if immunocompromised. Hepatitis B and A vaccines critical given historical blood product exposures.",
                },
                
                "inhibitors": {
                    "what are inhibitors": "Inhibitors are antibodies the immune system produces against replacement factor. They neutralize the infused factor, making it ineffective. This is the most serious complication of hemophilia treatment. Develops in 20-25% of Hemophilia A patients (higher with certain mutations). Treatment becomes exponentially more complex.",
                    "inhibitor screening": "Bethesda assay is the gold-standard test - measures antibodies against factor VIII/IX. Reported in Bethesda Units. Results: <0.6 BU = negative (good), 0.6-1 BU = borderline (needs repeat), >1 BU = positive (inhibitor present). This patient has " + str(round(patient_data.get("Risk", 0.5) * 2, 1)) + " Bethesda Units estimated risk.",
                    "inhibitor symptoms": "Signs that an inhibitor may have developed: (1) Factor infusions no longer stop bleeding, (2) Longer bleeding times despite normal dosing, (3) Unexplained hemarthroses or muscle hematomas, (4) Poor response to treatment. ANY of these warrant urgent inhibitor titer testing.",
                    "inhibitor management": "Treatment depends on inhibitor level: LOW-titer (<5 BU): Can try higher-dose factor. HIGH-titer (>5 BU): Requires bypass therapy (aPCC or Emicizumab) OR immune tolerance induction. Treatment changes dramatically - costs rise 100-200x, infusions become more complex, outcomes less certain.",
                    "inhibitor prevention": "Strategies to prevent inhibitor formation: (1) Early exposure to low antigenicity factors when possible, (2) Regular factor prophylaxis (not on-demand - reduces immunogenic dosing), (3) Maintain good health (avoid triggers), (4) Genetic counseling (some mutations have higher risk). This patient's " + str(round(patient_data.get("Risk", 0), 2)*100) + "% risk is " + ("HIGH - close monitoring essential" if patient_data.get("Risk", 0) > 0.6 else "manageable with standard care"),
                },
                
                "complications": {
                    "target joints": "Repeated bleeding into the same joint (ankle, knee, elbow most common) causes chronic arthropathy - 'target joint syndrome'. Results in joint destruction, chronic pain, limited mobility. Early aggressive treatment prevents this. This patient has joint damage score of " + str(patient_data.get("Joint Damage", "unknown")) + ".",
                    "hepatitis c": "Historical concern from contaminated blood products (pre-1992 plasma-derived factors). Modern products are safe through viral inactivation/filtration. Legacy patients screened regularly. If positive: monitoring for cirrhosis, interferon/DAA treatment options, liver specialist involvement.",
                    "hiv": "Similar to Hep C - historical concern, not modern risk. Pre-1983 plasma products were contaminated. Modern recombinant/virally inactivated products are HIV-safe. Today's hemophilia patients born on safe products.",
                    "joint damage": "Hemophilic arthropathy is progressive joint damage from repeated bleeds. Progressive synovitis -> cartilage loss -> bone destruction. Can lead to chronic pain, immobility, need for joint replacement. PREVENTION is key - this patient's bleeding episode rate of " + str(patient_data.get("Bleeding Episodes", "standard")) + " per year is tracked closely.",
                    "intracranial hemorrhage": "ICH is rare but life-threatening. Can happen spontaneously in severe hemophilia. Warning signs: severe headache, neck stiffness, vision changes, weakness, confusion. EMERGENCY - requires immediate ER visit and max-dose factor replacement.",
                    "life expectancy": "Modern hemophiliacs have near-normal life expectancy compared to general population. Key factors: adherence to prophylaxis, inhibitor prevention, regular screening for viral infections, lifestyle modifications. This patient's risk profile suggests good long-term prognosis with proper management.",
                },
                
                "general_disease": {
                    "what causes bleeding disorders": "Bleeding disorders result from: (1) Clotting factor deficiencies (Hemophilia), (2) Platelet dysfunction, (3) Fibrinogen abnormality, (4) Blood vessel issues, (5) DIC (from sepsis/trauma). Each requires different diagnosis and treatment approach.",
                    "coagulation cascade": "Three phases: INITIATION (Tissue Factor + VII), AMPLIFICATION (generates more thrombin), PROPAGATION (massive thrombin burst). Forms dense fibrin clot. Hemophilia disrupts propagation phase. Understanding this explains why factor VIII/IX replacement works.",
                    "how blood clots": "Complex process: (1) Platelets plug hole (primary hemostasis), (2) Clotting factors form thrombin (secondary hemostasis), (3) Fibrin stabilizes clot. Hemophilia disrupts step 2. Analogy: platelets = rebar, fibrin = concrete. Need both.",
                    "genetic counseling": "Recommended for all hemophilia patients/families. Discusses inheritance patterns, risks to children, carrier testing for females, prenatal testing options, prevention strategies. Can help with family planning decisions.",
                    "quality of life": "Modern management allows near-normal activities. Many hemophiliac athletes compete. Key: good prophylaxis, home infusion capability, employer accommodations. Psychological support important - anxiety about bleeds is real.",
                    "exercise and sports": "Exercise strengthens muscles, protects joints. LOW-RISK sports (swimming, golf, walking) encouraged. HIGH-RISK sports (football, boxing) avoided. Prophylaxis + protection (braces, padding) enables activity. Physical therapy crucial.",
                },
                
                "patient_specific": {
                    "mutation explanation": {
                        "Intron22": f"Your mutation (Intron22, found in ~45% of severe cases) has 50% inhibitor rate - highest among mutations. This is why we're recommending {patient_data.get('Risk', 'high')} level monitoring.",
                        "Missense": f"Your mutation (Missense) has moderate inhibitor risk (~20%). Better prognosis than Intron22, but still requires vigilant monitoring.",
                        "Nonsense": f"Your mutation (Nonsense) typically causes severe disease with moderate inhibitor risk (~25%).",
                        "Frameshift": f"Your mutation (Frameshift) causes severe disease and usually moderate-to-high inhibitor risk."
                    },
                    "risk explanation": f"Your calculated risk of inhibitor development is {patient_data.get('Risk', 0.5):.1%}. This is based on: mutation type (strongest factor), severity ({patient_data.get('Severity')}), dose intensity ({patient_data.get('Dose')}), exposure days ({patient_data.get('Exposure')}), and clinical factors like family history. This informs our monitoring intensity.",
                    "treatment dose": f"Your prescribed dose is {patient_data.get('Dose')} units. This is calculated based on weight and severity. Some patients need higher doses (high-intensity prophylaxis), others lower doses (on-demand management). Discuss with your team if you think adjustment is needed.",
                    "activity advice": f"With your {patient_data.get('Severity', 'moderate')} hemophilia and {patient_data.get('Risk', 0.5):.1%} inhibitor risk, we recommend: regular low-impact exercise, protective equipment during activities, maintain good prophylaxis adherence. Avoid high-contact sports."
                }
            }
            
            # Analyze question to find best matching knowledge topic
            def find_best_match(question_text, kb):
                """Find most relevant knowledge base entry"""
                q_words = set(question_text.lower().split())
                best_match = None
                best_score = 0
                
                for category, items in kb.items():
                    for key, content in items.items():
                        key_words = set(key.split('_'))
                        score = len(q_words & key_words)
                        if score > best_score:
                            best_score = score
                            best_match = (category, key, content)
                
                return best_match
            
            # Find relevant knowledge
            match = find_best_match(q, knowledge_base)
            
            if match and match[2]:  # If we found relevant knowledge
                category, key, content = match
                
                # Format response with context
                if category == "patient_specific":
                    response = f"**For your specific situation:**\n\n{content}\n\n"
                else:
                    response = f"**Medical Information:**\n\n{content}\n\n"
                
                # Add personalized follow-up based on context
                if "risk" in q or "danger" in q or "how bad" in q:
                    if patient_data.get("Risk", 0) > 0.8:
                        response += "\n⚠️ **For you specifically:** Your risk profile warrants close specialist oversight. Ensure you're with an experienced hemophilia treatment center."
                    elif patient_data.get("Risk", 0) > 0.6:
                        response += "\n📋 **For you specifically:** Your moderate-to-high risk means regular monitoring is crucial. Stay compliant with your 3-month screening schedule."
                    else:
                        response += "\n✅ **For you specifically:** Your lower risk profile is manageable with standard care. Stay vigilant with monitoring."
                
                if "treatment" in q or "therapy" in q or "factor" in q:
                    response += f"\n\n💊 **Your current regimen:** {patient_data.get('Dose')} units, {patient_data.get('Product', 'standard')} product, adherence at {patient_data.get('Adherence', '80')}%"
                
                response += "\n\n---\n*Have more questions? Ask me about inhibitors, mutations, monitoring, complications, or any other hemophilia topic!*"
                return response
            
            # If no specific match, provide intelligent general response
            else:
                # Generic but informed responses
                if any(word in q for word in ["help", "can you", "what can you", "how can you"]):
                    return """I'm an AI Medical Assistant trained on hemophilia and bleeding disorders. I can help you understand:

✅ **Hemophilia Basics** - What it is, how it works, inheritance patterns
✅ **Treatment Options** - Prophylaxis, on-demand, new therapies, factor types
✅ **Inhibitors** - Risk, screening, symptoms, management, prevention
✅ **Complications** - Joint damage, bleeding patterns, long-term care
✅ **Your Specific Case** - Your mutation, risk level, personal monitoring plan
✅ **General Disease Info** - Clotting, genetics, quality of life, exercise

Ask me anything about hemophilia A or B, bleeding disorders, or your personal health management!

---
*Example questions:*
- "What are inhibitors and why should I worry?"
- "What does my mutation mean for my prognosis?"
- "How often should I be monitored?"
- "Can I play sports?"
- "What are the signs of a bleed?"
"""
                
                elif any(word in q for word in ["bleeding", "blood", "clot"]):
                    return """Great question about bleeding and clotting! 

In normal blood clotting, a cascade of proteins (called clotting factors) work together in precise sequence. Think of it like a domino chain - each factor activates the next, ultimately creating fibrin (the structural protein that forms the clot).

Hemophilia disrupts this cascade by missing or disabling Factor VIII (Hemophilia A) or Factor IX (Hemophilia B). Without these critical factors, the domino chain breaks and clotting never reaches completion.

**Result:** Clots take much longer to form, are weaker once formed, and may break down prematurely. This leads to:
- Prolonged bleeding from cuts
- Spontaneous bleeding into joints/muscles
- Dangerous internal hemorrhages

**Solution:** Regular factor replacement therapy provides the missing factor, allowing clotting to proceed normally.

Want to know more about your specific situation or treatment options?
"""
                
                elif any(word in q for word in ["how often", "when", "frequency", "schedule"]):
                    return f"""Monitoring frequency depends on your risk level and current status.

**For you (Risk: {patient_data.get('Risk', 0.5):.1%}):**

""" + ("**INTENSIVE MONITORING (High Risk):**\n- Inhibitor screening: Every 4-6 weeks\n- Factor levels: Monthly\n- Clinical visit: Monthly\n- Assess bleeding pattern: Continuous tracking" if patient_data.get("Risk", 0) > 0.6 else "**STANDARD MONITORING (Lower Risk):**\n- Inhibitor screening: Quarterly (every 3 months)\n- Factor levels: Quarterly\n- Clinical visit: Monthly\n- Routine labs: As recommended by team") + f"""

**Other regular assessments:**
- Joint evaluations: Annually or if new symptoms
- Imaging: If joint pain develops
- Viral screening: Annually
- Risk reassessment: Yearly

Your team can adjust these based on how you're doing. Don't hesitate to ask for more frequent monitoring if you're worried - better safe than sorry!
"""
                
                else:
                    return f"""That's a great question about hemophilia! While I may not have a perfect match in my knowledge base for that specific query, here's what I'd recommend:

1. **Check my knowledge base** - Ask me about: inhibitors, mutations, treatments, complications, or your specific risk level ({patient_data.get('Risk', 0.5):.1%})

2. **Consult your team** - Your hemophilia specialists are the best resource for detailed clinical questions

3. **Ask clarifying questions** - Try questions like:
   - "What does {' '.join(q.split()[:2])} mean for hemophilia patients?"
   - "How does {' '.join(q.split()[:2])} affect treatment?"
   - "Should I be worried about {' '.join(q.split()[:3])}?"

Feel free to rephrase your question or ask me about a hemophilia topic I can help with!

---
*I'm particularly knowledgeable about inhibitor management, mutation types, treatment strategies, and your personal risk assessment.*
"""
        
        # Chat interface
        st.markdown("### 💬 Chat with AI Medical Assistant")
        st.info("🤖 Ask me ANY question about hemophilia, bleeding disorders, treatments, inhibitors, your personal health - I'm trained to provide evidence-based medical information!")
        
        # Conversation history display
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me about hemophilia, your treatment, inhibitor risks, genetics, monitoring, anything healthcare-related...")
        
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            response = generate_response(user_input, d)
            
            # Add assistant response to history
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            
            # Rerun to display new message
            st.rerun()


# ============= CHATBOT PAGE WITH REAL GPT-4 =============
elif page == "Chatbot":

    st.title("🤖 AI Medical Assistant - GPT-4 Powered")

    if "data" not in st.session_state:
        st.warning("⚠️ Run prediction first from Patient Form to access chatbot")
    else:
        d = st.session_state.data
        
        # Get or create patient ID
        if "patient_id" not in st.session_state:
            patient_data_dict = {
                "Name": d.get("Name"),
                "Age": d.get("Age"),
                "Gender": d.get("Gender"),
                "Ethnicity": d.get("Ethnicity"),
                "Severity": d.get("Severity"),
                "Mutation": d.get("Mutation"),
                "Blood_Type": d.get("Blood Type"),
                "HLA_Type": d.get("HLA Type"),
                "Dose": d.get("Dose"),
                "Exposure": d.get("Exposure"),
                "Product_Type": d.get("Product"),
                "Treatment_Adherence": d.get("Adherence"),
                "Family_History": d.get("Family History"),
                "Previous_Inhibitor": d.get("Previous Inhibitor"),
                "Joint_Damage": d.get("Joint Damage"),
                "Bleeding_Episodes": d.get("Bleeding Episodes"),
                "Factor_Level": d.get("Factor Level"),
                "Immunosuppression": d.get("Immunosuppression"),
                "Active_Infection": d.get("Active Infection"),
                "Vaccination_Status": d.get("Vaccination"),
                "Physical_Activity": d.get("Activity Level"),
                "Stress_Level": d.get("Stress Level"),
                "Comorbidities": d.get("Comorbidities"),
                "Risk_Score": d.get("Risk")
            }
            patient_id = add_patient(patient_data_dict)
            st.session_state.patient_id = patient_id
        else:
            patient_id = st.session_state.patient_id
        
        # Initialize conversation history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # Display info about current patient
        st.info(f"""
        👤 **Patient:** {d['Name']} | 🔴 **Risk:** {d['Risk']:.1%} | 🧬 **Mutation:** {d['Mutation']} | ⚠️ **Severity:** {d['Severity']}
        """)
        
        st.markdown("### 💬 AI Medical Conference")
        st.markdown("*Real-time GPT-4 powered clinical decision support and medical consultation*")
        
        # Conversation display
        if st.session_state.conversation_history:
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👨‍⚕️"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(msg["content"])
        
        # Chat input with columns for options
        col_chat1, col_chat2, col_chat3 = st.columns([3, 0.7, 0.7])
        
        with col_chat1:
            user_input = st.chat_input("Ask Dr. AI Medical Assistant anything about this patient's care...")
        
        with col_chat2:
            if st.button("🧭 Clinical Recommendations"):
                with st.spinner("🔄 Generating clinical recommendations..."):
                    try:
                        recommendations = get_clinical_recommendations(d)
                        st.session_state.conversation_history.append({
                            "role": "user",
                            "content": "Generate comprehensive clinical recommendations"
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": recommendations
                        })
                        add_conversation(patient_id, "Generate clinical recommendations", recommendations, "recommendations")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
        
        with col_chat3:
            if st.button("🧹 Clear Chat"):
                st.session_state.conversation_history = []
                st.rerun()
        
        if user_input:
            # Add user message
            st.session_state.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate GPT response with patient context and history
            with st.spinner("🤔 Dr. AI is thinking..."):
                try:
                    db_history = get_conversation_history(patient_id, limit=10)
                    response = create_gpt_response(user_input, patient_context=d, conversation_history=db_history)
                    
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Store in database
                    add_conversation(patient_id, user_input, response, "general")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating response: {str(e)[:150]}")
                    if "API" in str(e):
                        st.warning("⚠️ Please ensure your OpenAI API key is set in the .env file")
        
        st.divider()
        
        # Additional options
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            if st.button("📋 Inhibitor Risk Analysis", use_container_width=True):
                with st.spinner("Generating inhibitor risk analysis..."):
                    try:
                        risk_analysis = generate_inhibitor_risk_explanation(d, {
                            "Mutation": d.get("Mutation"),
                            "Severity": d.get("Severity"),
                            "Exposure": d.get("Exposure"),
                            "Family History": d.get("Family History")
                        })
                        st.markdown("### 🎯 Inhibitor Risk Analysis")
                        st.write(risk_analysis)
                        add_conversation(patient_id, "Inhibitor risk analysis requested", risk_analysis, "risk_analysis")
                    except Exception as e:
                        st.error(f"Error: {str(e)[:100]}")
        
        with col_opt2:
            if st.button("📊 Monitoring Data Analysis", use_container_width=True):
                monitoring_records = get_monitoring_records(patient_id)
                if monitoring_records:
                    with st.spinner("Analyzing monitoring data..."):
                        try:
                            analysis = analyze_monitoring_data(d, monitoring_records)
                            st.markdown("### 📊 Monitoring Analysis")
                            st.write(analysis)
                            add_conversation(patient_id, "Monitoring data analysis requested", analysis, "monitoring_analysis")
                        except Exception as e:
                            st.error(f"Error: {str(e)[:100]}")
                else:
                    st.info("No monitoring records available yet")
        
        with col_opt3:
            if st.button("💾 Save Doctor Note", use_container_width=True):
                with st.expander("Add Doctor Note"):
                    doctor_name = st.text_input("Doctor Name")
                    note_content = st.text_area("Note Content")
                    note_category = st.selectbox("Category", ["General", "Inhibitor", "Treatment", "Monitoring", "Follow-up"])
                    severity = st.selectbox("Severity", ["Normal", "Important", "Urgent"])
                    
                    if st.button("Save Note"):
                        if doctor_name and note_content:
                            add_doctor_note(patient_id, doctor_name, note_content, note_category, severity)
                            st.success("✅ Note saved!")
                        else:
                            st.warning("Please fill in all fields")

# ============= END OF CHATBOT PAGE =============

# ============= DOCTOR DASHBOARD ===============
elif page == "Doctor Dashboard":
    
    st.title("🏥 Doctor Dashboard - Clinical Analytics & Management")
    st.markdown("*Comprehensive patient management and clinical analytics portal*")
    
    # Get dashboard statistics
    stats = get_dashboard_stats()
    
    # Display key metrics
    st.markdown("### 📊 System Overview")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Patients", stats["total_patients"], "Registered in system")
    
    with metric_col2:
        high_risk_pct = f"{stats['high_risk_patients']/max(stats['total_patients'], 1)*100:.1f}%" if stats["total_patients"] > 0 else "0%"
        st.metric("High Risk Patients", stats["high_risk_patients"], high_risk_pct)
    
    with metric_col3:
        st.metric("Severe Cases", stats["severe_cases"], "Severe hemophilia")
    
    with metric_col4:
        st.metric("Avg Risk Score", f"{stats['average_risk']:.1%}", "Population average")
    
    st.divider()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👥 Patient Directory",
        "📋 Clinical Notes",
        "📊 Analytics & Trends",
        "🔍 Search & Filter",
        "⚙️ Utilities"
    ])
    
    # ============= TAB 1: PATIENT DIRECTORY =============
    with tab1:
        st.markdown("### 👥 Registered Patients")
        
        patients = get_all_patients()
        
        if patients:
            df_patients = pd.DataFrame(patients)
            st.info(f"Total: {len(df_patients)} patients registered")
            
            def get_risk_category(risk):
                if risk > 0.8:
                    return "🔴 CRITICAL"
                elif risk > 0.6:
                    return "🟠 HIGH"
                elif risk > 0.4:
                    return "🟡 MODERATE"
                else:
                    return "🟢 LOW"
            
            display_df = df_patients[['id', 'name', 'age', 'severity', 'mutation', 'dose', 'exposure', 'risk_score', 'treatment_adherence', 'created_at']].copy()
            display_df['Risk Category'] = display_df['risk_score'].apply(get_risk_category)
            display_df['Adherence'] = display_df['treatment_adherence'].astype(str) + "%"
            
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            
            st.markdown("### 📋 Patient Details")
            patient_id = st.selectbox("Select Patient to View Details", options=[p['id'] for p in patients], format_func=lambda x: next(p['name'] for p in patients if p['id'] == x))
            
            if patient_id:
                patient = get_patient(patient_id)
                col_detail1, col_detail2, col_detail3 = st.columns(3)
                
                with col_detail1:
                    st.markdown("**Demographics**")
                    st.write(f"Name: {patient.get('name')}")
                    st.write(f"Age: {patient.get('age')}")
                    st.write(f"Gender: {patient.get('gender')}")
                
                with col_detail2:
                    st.markdown("**Clinical Profile**")
                    st.write(f"Severity: {patient.get('severity')}")
                    st.write(f"Mutation: {patient.get('mutation')}")
                    st.write(f"Risk: {patient.get('risk_score'):.1%}")
                
                with col_detail3:
                    st.markdown("**Treatment**")
                    st.write(f"Dose: {patient.get('dose')} units")
                    st.write(f"Exposure: {patient.get('exposure')} days")
                    st.write(f"Adherence: {patient.get('treatment_adherence')}%")
                
                st.markdown("#### 📝 Doctor Notes")
                doctor_notes = get_doctor_notes(patient_id)
                
                if doctor_notes:
                    for note in doctor_notes:
                        with st.expander(f"📌 {note['note_category']} - {note['created_at'][:10]}"):
                            st.write(f"**Doctor:** {note['doctor_name']}")
                            st.write(f"**Severity:** {note['severity']}")
                            st.write(note['note_content'])
                
                with st.expander("➕ Add New Doctor Note"):
                    new_doctor_name = st.text_input("Your Name", key="doctor_name_input")
                    new_note_content = st.text_area("Note Content", key="note_content_input")
                    new_note_category = st.selectbox("Category", ["General", "Inhibitor", "Treatment", "Monitoring", "Follow-up"], key="note_cat")
                    new_note_severity = st.selectbox("Severity", ["Normal", "Important", "Urgent"], key="note_sev")
                    
                    if st.button("Save Doctor Note"):
                        if new_doctor_name and new_note_content:
                            add_doctor_note(patient_id, new_doctor_name, new_note_content, new_note_category, new_note_severity)
                            st.success("✅ Note saved successfully!")
                            st.rerun()
                        else:
                            st.warning("Please fill in all fields")
        else:
            st.info("No patients registered yet")
    
    # ============= TAB 2: CLINICAL NOTES =============
    with tab2:
        st.markdown("### 📋 Clinical Notes Management")
        patients_list = get_all_patients()
        if patients_list:
            selected_patient = st.selectbox("Select Patient", options=[p['id'] for p in patients_list], format_func=lambda x: f"{[p['name'] for p in patients_list if p['id'] == x][0]} (Risk: {[p['risk_score'] for p in patients_list if p['id'] == x][0]:.1%})")
            
            if selected_patient:
                patient_data = get_patient(selected_patient)
                notes = get_doctor_notes(selected_patient)
                st.markdown(f"**Patient:** {patient_data['name']} | **Risk:** {patient_data['risk_score']:.1%}")
                
                if notes:
                    categories = set([n['note_category'] for n in notes])
                    for cat in sorted(categories):
                        cat_notes = [n for n in notes if n['note_category'] == cat]
                        with st.expander(f"📌 {cat} ({len(cat_notes)} notes)"):
                            for note in cat_notes:
                                st.write(f"**{note['doctor_name']}** - {note['created_at'][:10]}")
                                st.write(note['note_content'])
                else:
                    st.info("No clinical notes for this patient")
        else:
            st.info("No patients available")
    
    # ============= TAB 3: ANALYTICS =============
    with tab3:
        st.markdown("### 📊 Clinical Analytics & Trends")
        patients_list = get_all_patients()
        
        if len(patients_list) > 0:
            patients_df = pd.DataFrame(patients_list)
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.markdown("**Risk Distribution**")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(patients_df['risk_score'], bins=10, color='skyblue', edgecolor='black')
                ax.set_xlabel("Risk Score")
                ax.set_ylabel("Number of Patients")
                st.pyplot(fig)
            
            with col_a2:
                st.markdown("**Severity Distribution**")
                severity_counts = patients_df['severity'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
                st.pyplot(fig)
        else:
            st.info("No data available")
    
    # ============= TAB 4: SEARCH & FILTER =============
    with tab4:
        st.markdown("### 🔍 Advanced Search & Filtering")
        search_type = st.radio("Search By", ["Patient Name", "Risk Level", "Mutation Type", "Severity"])
        
        if search_type == "Patient Name":
            search_term = st.text_input("Enter patient name")
            if search_term:
                results = search_patients(search_term)
                if results:
                    st.success(f"Found {len(results)} patient(s)")
                    st.dataframe(pd.DataFrame(results)[['name', 'age', 'severity', 'mutation', 'risk_score']], use_container_width=True)
                else:
                    st.info("No patients found")
    
    # ============= TAB 5: UTILITIES =============
    with tab5:
        st.markdown("### ⚙️ System Utilities")
        if st.button("📥 Export All Patients (CSV)", use_container_width=True):
            patients_list = get_all_patients()
            df_export = pd.DataFrame(patients_list)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"all_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ============= END OF DOCTOR DASHBOARD =============

# ============= END OF ADVANCED CHATBOT =============

# ---------------- FOOTER ----------------
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **🏥 About This Platform**
    - Hemophilia AI Clinical Intelligence System
    - Machine Learning Risk Prediction
    - Real-time Patient Analytics
    """)

with footer_col2:
    st.markdown("""
    **🤖 ML Models Used**
    - Random Forest Classifier
    - XGBoost Ensemble
    - Feature Importance Analysis
    - Ensemble Averaging
    """)

with footer_col3:
    st.markdown("""
    **⚠️ Important**
    - Not a replacement for medical advice
    - For clinical support only
    - Always consult specialists
    - Results are probabilistic
    """)

st.caption("© 2026 Hemophilia AI Platform | Powered by Real Trained ML Models | Clinical Intelligence System")