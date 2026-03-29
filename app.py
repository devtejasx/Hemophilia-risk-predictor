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
    get_monitoring_records, add_treatment_record, get_treatment_history,
    register_user, authenticate_user, get_all_users, get_doctors,
    assign_patient_to_doctor, get_assigned_doctor, get_doctor_patients,
    get_patient_audit_trail
)
from gpt_chatbot import (
    create_gpt_response, get_clinical_recommendations,
    analyze_monitoring_data, generate_inhibitor_risk_explanation,
    analyze_case_complexity, generate_treatment_plan, 
    compare_treatment_options, explain_test_results, 
    identify_clinical_alerts, provide_patient_education,
    multi_turn_consultation, generate_progress_summary
)
from user_auth import UserManager
from dashboard_persistence import DashboardPersistence

# Initialize database at startup
try:
    init_database()
    # Initialize demo users for first-time use
    UserManager.initialize_demo_users()
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
        
        # Ensure shap_vals is 1D
        if isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) > 1:
                shap_vals = shap_vals.flatten()
        
        # Ensure we have the correct number of features
        if len(shap_vals) != len(feature_names):
            shap_vals = shap_vals[:len(feature_names)]
        
        # Create dataframe for visualization
        shap_df = pd.DataFrame({
            "Feature": feature_names[:len(shap_vals)],
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
        background: linear-gradient(135deg, #0d1428 0%, #1a1f3a 50%, #0a0e27 100%);
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        padding: 1.5rem 2.5rem;
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1 {
        color: #00d4ff;
        font-size: 2.8em !important;
        font-weight: 800 !important;
        margin-bottom: 0.8rem !important;
        letter-spacing: 1px;
    }
    
    h2 {
        color: #00d4ff;
        font-size: 2em !important;
        font-weight: 700 !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: 0.5px;
    }
    
    h3 {
        color: #00d4ff;
        font-size: 1.5em !important;
        font-weight: 600 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    p, span {
        color: #ffffff;
        font-size: 1.05em;
        line-height: 1.7;
    }
    
    /* ===== CARDS & CONTAINERS ===== */
    .card, [data-testid="stExpander"] {
        background: rgba(25, 30, 50, 0.8);
        border: 2px solid #00d4ff;
        padding: 24px !important;
        border-radius: 16px !important;
        margin-bottom: 18px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.25);
        border-color: #00ffff;
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 20, 40, 0.95);
        border-right: 2px solid #00d4ff;
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 20px 15px;
    }
    
    /* ===== BUTTONS ===== */
    button, [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        font-weight: 700 !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.35) !important;
        position: relative;
        overflow: hidden;
    }
    
    button:hover, [data-testid="stButton"] > button:hover {
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    button:active, [data-testid="stButton"] > button:active {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* ===== FORM INPUTS ===== */
    input, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid #00d4ff !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        font-size: 1em !important;
        transition: all 0.3s ease !important;
    }
    
    input::placeholder {
        color: #a0a8c0 !important;
    }
    
    input:focus, [data-testid="stTextInput"] input:focus {
        border-color: #00ffff !important;
        box-shadow: 0 0 16px rgba(0, 212, 255, 0.5) !important;
        background: rgba(0, 212, 255, 0.1) !important;
    }
    
    /* Sliders */
    [data-testid="stSlider"] {
        padding: 14px 0;
    }
    
    [data-testid="stSlider"] [role="slider"] {
        background: #00d4ff !important;
    }
    
    /* Select Boxes */
    select, [data-testid="stSelectbox"] {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 2px solid #00d4ff !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    select:hover {
        border-color: #00ffff !important;
    }
    
    /* ===== ALERTS ===== */
    [data-testid="stAlert"] {
        border-radius: 14px !important;
        border-left: 5px solid #00d4ff !important;
        background: rgba(0, 212, 255, 0.15) !important;
        padding: 18px 20px !important;
        box-shadow: 0 4px 16px rgba(0, 212, 255, 0.15);
        color: #ffffff !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: rgba(0, 212, 255, 0.1);
        border: 2px solid #00d4ff;
        border-radius: 14px;
        padding: 24px !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.15);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.25);
        transform: translateY(-2px);
    }
    
    /* Metric Label */
    [data-testid="stMetric"] label {
        color: #ffffff !important;
        font-size: 1em !important;
        font-weight: 600;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 2px;
        background: #00d4ff;
        margin: 24px 0;
    }
    
    /* ===== DATAFRAME ===== */
    [data-testid="stDataFrame"] {
        border-radius: 14px !important;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        border: 2px solid #00d4ff !important;
    }
    
    /* ===== EXPANDER ===== */
    [data-testid="stExpander"] {
        border: 2px solid #00d4ff !important;
    }
    
    /* ===== TEXT & LABELS ===== */
    p, span, label {
        color: #ffffff;
    }
    
    label {
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 10px;
        font-size: 1.05em;
    }
    
    /* ===== LINKS ===== */
    a {
        color: #00d4ff !important;
        text-decoration: none;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    a:hover {
        color: #00ffff !important;
        text-decoration: underline;
    }
    
    /* ===== MESSAGE STATES ===== */
    .stSuccess {
        background: rgba(34, 197, 94, 0.2) !important;
        border: 2px solid #22c55e !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(34, 197, 94, 0.2);
        color: #ffffff !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 2px solid #ef4444 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.2);
        color: #ffffff !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.2) !important;
        border: 2px solid #f59e0b !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.2);
        color: #ffffff !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.2) !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2);
        color: #ffffff !important;
    }
    
    /* ===== TABS ===== */
    [data-testid="stTabs"] [role="tablist"] button {
        border-bottom: 3px solid transparent !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
        padding: 12px 16px !important;
        font-weight: 600;
        font-size: 1em;
    }
    
    [data-testid="stTabs"] [role="tablist"] button:hover {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.1);
        border-bottom-color: #00d4ff !important;
    }
    
    [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    
    /* ===== CAPTIONS ===== */
    .caption {
        color: #a0a8c0 !important;
        font-size: 0.95em;
        font-weight: 500;
    }
    
    /* ===== MARKDOWN CONTENT ===== */
    .markdown-text-container {
        color: #ffffff;
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
        background: rgba(0, 212, 255, 0.1);
        border-radius: 8px;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 2px dashed #00d4ff !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #00ffff !important;
        background: rgba(0, 212, 255, 0.15) !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d4ff 0%, #0099ff 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00ffff 0%, #00d4ff 100%);
    }
    
    /* ===== FOOTER ===== */
    footer {
        background: rgba(10, 14, 39, 0.5);
        border-top: 2px solid #00d4ff;
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
            font-size: 0.95em !important;
        }
        
        [data-testid="stMetric"] {
            padding: 16px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN & AUTHENTICATION ----------------
# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.consultation_history = []

# Display login page if not authenticated
if not st.session_state.get("authenticated"):
    UserManager.login_page()
    st.stop()

# User is authenticated - Display main application header with user profile
st.set_page_config(page_title="🏥 Hemophilia AI Platform", layout="wide", initial_sidebar_state="expanded")


# ---------- SIDEBAR WITH USER PROFILE ------
with st.sidebar:
    st.divider()
    # Display current user
    curr_user = st.session_state.get("user", {})
    col_prof1, col_prof2 = st.columns([2, 1])
    with col_prof1:
        st.markdown(f"### 👤 {curr_user.get('full_name', 'User')}")
        st.caption(f"Role: **{curr_user.get('role', 'User').upper()}**")
    with col_prof2:
        if st.button("🚪 Logout", key="logout_button_sidebar_2"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    st.divider()


# ------------------- SAVE ----------------
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
    try:
        with open("patients.csv", "a", newline="") as f:
            writer = csv.writer(f)
            # Match the exact order and names from CSV_COLUMNS
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
                patient_data_dict.get("Vaccination_Status", patient_data_dict.get("Vaccination", "")),  # Handle both key names
                patient_data_dict.get("Physical_Activity", patient_data_dict.get("Activity Level", "")),  # Handle both key names
                patient_data_dict.get("Stress_Level", patient_data_dict.get("Stress Level", "")),  # Handle both key names
                patient_data_dict.get("Comorbidities", ""),
                patient_data_dict.get("Risk_Score", 0)
            ]
            writer.writerow(row)
        
        # Also save to database for backup
        try:
            add_patient(
                name=patient_data_dict.get("Name", ""),
                age=patient_data_dict.get("Age", 0),
                severity=patient_data_dict.get("Severity", ""),
                mutation=patient_data_dict.get("Mutation", ""),
                dose=patient_data_dict.get("Dose", 0),
                exposure=patient_data_dict.get("Exposure", 0),
                risk_score=patient_data_dict.get("Risk_Score", 0),
                treatment_adherence=patient_data_dict.get("Treatment_Adherence", 0)
            )
        except:
            pass  # Silently fail if database save fails
    except Exception as e:
        pass  # Silently fail to prevent app crashes

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

# Logout button is already in the sidebar above


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
                    "Vaccination": vaccination_status,
                    "Activity Level": physical_activity,
                    "Stress Level": stress_level,
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
                
                # Ensure shap_vals is 1D numpy array
                if isinstance(shap_vals, np.ndarray):
                    if len(shap_vals.shape) > 1:
                        shap_vals = shap_vals.flatten()
                
                # Convert to list to ensure it's compatible with sorting
                shap_vals = np.asarray(shap_vals).flatten().tolist()
                
                # Get top contributing factors
                shap_impacts = sorted(zip(features, np.abs(shap_vals)), key=lambda x: x[1], reverse=True)[:5]
                
                with st.expander("📋 Top Contributing Factors (SHAP Analysis)"):
                    for i, (feature, impact) in enumerate(shap_impacts, 1):
                        st.write(f"**{i}. {feature}** - Impact: {impact:.4f}")
        
        st.divider()
        
        # Detailed Summary with Clinical Data
        st.markdown("### 🧾 Complete Clinical Assessment Summary")
        
        with st.expander("Click to expand comprehensive clinical data", expanded=True):
            
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
                                     "Risk_Score_Numeric", "Risk_Category"]].copy()
            display_df.columns = ["Name", "Age", "Gender", "Severity", "Mutation", 
                                 "Dose", "Exposure", "Fam Hx", "Prev Inh", "Risk Score", "Risk Category"]
            
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


# ---------------- CHATBOT WITH GPT ----------------
elif page == "Chatbot":

    st.title("🤖 Smart Medical Assistant - Powered by GPT")
    st.markdown("*Intelligent AI-powered clinical consultation with full chat history and patient context*")

    if "data" not in st.session_state:
        st.warning("⚠️ Run a prediction first from Patient Form to use the chatbot")
    else:
        d = st.session_state.data
        
        # Initialize conversation history
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        # Advanced response generation using GPT API
        def generate_gpt_response(question, patient_data, conversation_history):
            """
            Generate responses using GPT API with patient context and chat history
            """
            try:
                # Use GPT to generate personalized response
                response = create_gpt_response(
                    question,
                    patient_context=patient_data,
                    conversation_history=conversation_history
                )
                return response
                
            except Exception as e:
                # Fallback response if GPT API fails
                error_msg = str(e)
                if "API" in error_msg or "key" in error_msg.lower():
                    return """⚠️ **API Connection Issue**
                    
The AI couldn't connect to GPT. Ensure your OpenAI API key is set in `.env` file."""
                else:
                    return f"""⚠️ **Error**: {error_msg[:80]}"""
        
        
        # Chat interface - Advanced ChatGPT style
        st.markdown("### 💬 Smart Medical Assistant - Advanced AI")
        
        # Display patient info in a professional card
        with st.container():
            col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
            with col_p1:
                st.markdown(f"**👤 {d.get('Name', 'Patient')}**")
            with col_p2:
                st.markdown(f"**🧬 {d.get('Mutation', 'N/A')}**")
            with col_p3:
                st.markdown(f"**⚠️ {d.get('Severity', 'N/A')}**")
            with col_p4:
                risk_val = d.get('Risk', 0)
                risk_emoji = '🔴' if risk_val > 0.8 else '🟠' if risk_val > 0.6 else '🟡' if risk_val > 0.4 else '🟢'
                st.markdown(f"**{risk_emoji} {risk_val:.1%}**")
            with col_p5:
                st.markdown(f"**📋 {d.get('Dose', 'N/A')} units**")
        
        # System learning indicator
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 12px; border-radius: 8px; margin: 10px 0;">
        <p style="color: white; margin: 0; font-size: 13px;">
        🧠 <b>AI Learning Mode Active:</b> This chatbot learns from all patients in the system. The more patients we track, 
        the smarter recommendations become based on similar cases and real outcomes.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Create tabs for different chat features
        tab_chat, tab_quick, tab_history = st.tabs(["💬 Chat", "⚡ Quick Questions", "📝 History"])
        
        # ============= MAIN CHAT TAB =============
        with tab_chat:
            # Welcome message or conversation
            if len(st.session_state.conversation_history) == 0:
                st.markdown("""
                ### 🎯 Welcome to Your Personal AI Medical Assistant!
                
                **I'm here to help with:**
                - 🧬 Your hemophilia mutation and what it means
                - ⚠️ Inhibitor risk assessment and management
                - 💊 Treatment options and dosing strategies
                - 📊 Monitoring schedules personalized for you
                - 🎯 Lifestyle advice and physical activity
                - ❓ Any medical questions about your condition
                - 📚 Educational information on bleeding disorders
                
                **Start by asking a question or picking a quick topic →**
                """)
            
            # Display conversation with enhanced styling
            if len(st.session_state.conversation_history) > 0:
                for msg_idx, msg in enumerate(st.session_state.conversation_history):
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(msg['content'])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(msg["content"])
                            # Add response actions
                            col_action1, col_action2, col_action3 = st.columns([1, 1, 1])
                            with col_action1:
                                if st.button("👍 Helpful", key=f"helpful_{msg_idx}", use_container_width=True):
                                    from gpt_chatbot import record_patient_feedback
                                    record_patient_feedback(d.get('Name', 'Unknown'), "helpful", f"Q: {msg.get('content', '')[:50]}")
                                    st.toast("✅ Thanks! This helps us learn.", icon="😊")
                            with col_action2:
                                if st.button("📋 Copy", key=f"copy_{msg_idx}", use_container_width=True):
                                    st.toast("📋 Copied to clipboard!", icon="✨")
                            with col_action3:
                                if st.button("❌ Not helpful", key=f"unhelpful_{msg_idx}", use_container_width=True):
                                    from gpt_chatbot import record_patient_feedback
                                    record_patient_feedback(d.get('Name', 'Unknown'), "not_helpful", f"Q: {msg.get('content', '')[:50]}")
                                    st.toast("💡 We'll improve this!", icon="🔧")
            
            st.divider()
            
            # Advanced input section
            st.markdown("**Ask me anything:**")
            col_chat_input, col_chat_btn = st.columns([5, 1])
            
            with col_chat_input:
                user_input = st.text_area(
                    "Your question:",
                    placeholder="E.g., 'Should I be concerned about my inhibitor risk?' or 'What activities can I do safely?'",
                    label_visibility="collapsed",
                    height=80,
                    key="chat_input_advanced"
                )
            
            with col_chat_btn:
                st.markdown("")
                st.markdown("")
                send_btn = st.button("🚀 Send", use_container_width=True, key="send_advanced")
            
            if send_btn and user_input:
                # Add user message
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                
                # Show loading state
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.info("🤔 AI is analyzing your question with your patient data...")
                
                # Generate response
                try:
                    response = generate_gpt_response(user_input, d, st.session_state.conversation_history)
                    
                    # Remove loading state
                    progress_placeholder.empty()
                    
                    # Add response
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    progress_placeholder.empty()
                    st.error(f"Error: {str(e)[:100]}")
        
        # ============= QUICK QUESTIONS TAB =============
        with tab_quick:
            st.markdown("### ⚡ Quick Questions")
            st.markdown("*Click any question to get instant answers tailored to your profile*")
            
            quick_questions = [
                ("🧬 What does my mutation mean?", f"Explain my {d.get('Mutation', 'N/A')} mutation and its implications."),
                ("⚠️ How high is my inhibitor risk?", f"Based on my {d.get('Risk', 0):.1%} risk score, what does this mean?"),
                ("💊 What's my treatment plan?", f"I'm on {d.get('Dose', 'N/A')} units. Is this appropriate?"),
                ("📊 How often should I be monitored?", f"Given my {d.get('Severity', 'N/A')} severity, what's my monitoring schedule?"),
                ("🎯 Can I exercise?", f"What physical activities are safe for me?"),
                ("🚨 What are warning signs?", "What symptoms should I watch for?"),
                ("👨‍👩‍👧 Can I have children?", "What are my options regarding family planning?"),
                ("💉 What about my joint health?", f"I have {d.get('Joint_Damage', '0')}/10 joint damage. What should I do?"),
            ]
            
            cols = st.columns(2)
            for idx, (title, full_q) in enumerate(quick_questions):
                with cols[idx % 2]:
                    if st.button(title, use_container_width=True, key=f"quick_q_{idx}"):
                        st.session_state.conversation_history.append({"role": "user", "content": full_q})
                        
                        with st.spinner("🤔 Getting answer..."):
                            response = generate_gpt_response(full_q, d, st.session_state.conversation_history)
                        
                        st.session_state.conversation_history.append({"role": "assistant", "content": response})
                        st.rerun()
        
        # ============= HISTORY TAB =============
        with tab_history:
            st.markdown("### 📝 Conversation History")
            
            if len(st.session_state.conversation_history) == 0:
                st.info("No conversation history yet. Start chatting to see your history here!")
            else:
                col_export, col_clear = st.columns(2)
                
                with col_export:
                    # Export conversation
                    history_text = "\n\n".join([
                        f"**You:** {msg['content']}" if msg['role'] == 'user' else f"**AI Assistant:** {msg['content']}"
                        for msg in st.session_state.conversation_history
                    ])
                    st.download_button(
                        label="📥 Export History",
                        data=history_text,
                        file_name="chat_history.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col_clear:
                    if st.button("🗑️ Clear History", use_container_width=True, key="clear_history_btn"):
                        st.session_state.conversation_history = []
                        st.success("✅ Chat history cleared!")
                        st.rerun()
                
                st.divider()
                
                # Summary view
                st.markdown("**Summary of your questions:**")
                for idx, msg in enumerate(st.session_state.conversation_history):
                    if msg['role'] == 'user':
                        # Show question summary
                        q_preview = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
                        with st.expander(f"❓ Q{(idx//2)+1}: {q_preview}", expanded=False):
                            st.write(f"**Question:** {msg['content']}")
                            # Show next response if exists
                            if idx + 1 < len(st.session_state.conversation_history):
                                st.write(f"**Response:** {st.session_state.conversation_history[idx+1]['content'][:500]}...")
        
        # ============= SIDEBAR ADVANCED OPTIONS =============
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⚙️ Assistant Settings")
            
            # Conversation style
            conv_style = st.radio(
                "Conversation Style:",
                ["Detailed & Scientific", "Simple & Friendly", "Quick & Direct"],
                key="conv_style"
            )
            
            if st.button("🎯 Clinical Summary", use_container_width=True, key="clinical_summary_btn"):
                summary_q = "Generate a comprehensive clinical summary of my case with recommendations."
                st.session_state.conversation_history.append({"role": "user", "content": summary_q})
                
                with st.spinner("Generating summary..."):
                    response = generate_gpt_response(summary_q, d, st.session_state.conversation_history)
                
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("📄 Treatment Plan", use_container_width=True, key="treatment_plan_btn"):
                plan_q = "Create a detailed treatment and monitoring plan for me."
                st.session_state.conversation_history.append({"role": "user", "content": plan_q})
                
                with st.spinner("Creating plan..."):
                    response = generate_gpt_response(plan_q, d, st.session_state.conversation_history)
                
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("❓ FAQ for My Condition", use_container_width=True, key="faq_btn"):
                faq_q = "What are the most common questions patients like me ask? Provide FAQs."
                st.session_state.conversation_history.append({"role": "user", "content": faq_q})
                
                with st.spinner("Curating FAQs..."):
                    response = generate_gpt_response(faq_q, d, st.session_state.conversation_history)
                
                st.session_state.conversation_history.append({"role": "assistant", "content": response})
                st.rerun()
            
            st.markdown("---")
            st.markdown("### 🧠 System Learning Status")
            
            # Get learning statistics
            try:
                import sqlite3
                conn = sqlite3.connect("hemophilia_clinic.db")
                cursor = conn.cursor()
                
                # Count patients in system
                cursor.execute("SELECT COUNT(*) FROM patients")
                patient_count = cursor.fetchone()[0]
                
                # Count feedback records
                cursor.execute("SELECT COUNT(*) FROM chatbot_feedback" if True else "SELECT 0")  # Safe query
                try:
                    cursor.execute("SELECT COUNT(*) FROM chatbot_feedback")
                    feedback_count = cursor.fetchone()[0]
                except:
                    feedback_count = 0
                
                conn.close()
                
                st.info(f"""
                **📚 AI Knowledge Base:**
                - {patient_count} patients tracked
                - {feedback_count} interactions logged
                - System improving with each patient
                """)
            except:
                pass
            
            st.markdown("---")
            st.markdown("### 💡 Tips")
            st.caption("💬 Ask follow-up questions for clarification")
            st.caption("❌ Let me know if responses aren't helpful")
            st.caption("📊 I always consider your specific patient data")
            st.caption("🔒 Your data stays private and secure")


# ============= END OF ENHANCED CHATBOT PAGE =============

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
    
    # ============= COMPREHENSIVE SYSTEM OVERVIEW =============
    st.markdown("### 📊 Comprehensive System Overview")
    
    # Get all patients from database
    all_db_patients = get_all_patients()
    
    if all_db_patients:
        db_df = pd.DataFrame(all_db_patients)
        
        # Create comprehensive overview
        overview_col1, overview_col2, overview_col3 = st.columns(3)
        
        with overview_col1:
            st.metric("Total Registered Patients", len(db_df))
            avg_age = db_df['age'].mean() if 'age' in db_df.columns else 0
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        with overview_col2:
            male_count = len(db_df[db_df['gender'] == 'Male']) if 'gender' in db_df.columns else 0
            female_count = len(db_df[db_df['gender'] == 'Female']) if 'gender' in db_df.columns else 0
            st.metric("Male Patients", male_count)
            st.metric("Female Patients", female_count)
        
        with overview_col3:
            avg_risk = db_df['risk_score'].mean() if 'risk_score' in db_df.columns else 0
            st.metric("Average Risk Score", f"{avg_risk:.1%}")
            max_risk = db_df['risk_score'].max() if 'risk_score' in db_df.columns else 0
            st.metric("Highest Risk Score", f"{max_risk:.1%}")
        
        # Severity breakdown
        severity_breakdown_col1, severity_breakdown_col2, severity_breakdown_col3 = st.columns(3)
        
        if 'severity' in db_df.columns:
            mild_count = len(db_df[db_df['severity'] == 'Mild'])
            moderate_count = len(db_df[db_df['severity'] == 'Moderate'])
            severe_count = len(db_df[db_df['severity'] == 'Severe'])
            
            with severity_breakdown_col1:
                st.metric("Mild Cases", mild_count)
            with severity_breakdown_col2:
                st.metric("Moderate Cases", moderate_count)
            with severity_breakdown_col3:
                st.metric("Severe Cases", severe_count)
        
        # Save System Overview button
        if st.button("💾 Save Complete System Overview", use_container_width=True, key="save_overview"):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create comprehensive overview report
                overview_report = {
                    'Generated At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Total Patients': len(db_df),
                    'Average Age': db_df['age'].mean() if 'age' in db_df.columns else 0,
                    'Male Patients': len(db_df[db_df['gender'] == 'Male']) if 'gender' in db_df.columns else 0,
                    'Female Patients': len(db_df[db_df['gender'] == 'Female']) if 'gender' in db_df.columns else 0,
                    'Average Risk Score': db_df['risk_score'].mean() if 'risk_score' in db_df.columns else 0,
                    'Highest Risk Score': db_df['risk_score'].max() if 'risk_score' in db_df.columns else 0,
                    'Lowest Risk Score': db_df['risk_score'].min() if 'risk_score' in db_df.columns else 0,
                }
                
                if 'severity' in db_df.columns:
                    overview_report['Mild Cases'] = len(db_df[db_df['severity'] == 'Mild'])
                    overview_report['Moderate Cases'] = len(db_df[db_df['severity'] == 'Moderate'])
                    overview_report['Severe Cases'] = len(db_df[db_df['severity'] == 'Severe'])
                
                # Save full patient data
                db_df.to_csv(f"system_overview_patients_{timestamp}.csv", index=False)
                
                # Save overview statistics
                overview_df = pd.DataFrame([overview_report])
                overview_df.to_csv(f"system_overview_stats_{timestamp}.csv", index=False)
                
                st.success(f"✅ System overview saved successfully!")
                st.info(f"📁 Files saved:\n- system_overview_patients_{timestamp}.csv\n- system_overview_stats_{timestamp}.csv")
            except Exception as e:
                st.error(f"❌ Error saving overview: {str(e)}")
        
        st.divider()
    
    # ============= ALL PREDICTED PATIENTS OVERVIEW =============
    st.markdown("### 🎯 All Predicted Patients Data")
    
    # Load all predicted patients from CSV
    try:
        patients_df = pd.read_csv("patients.csv")
        
        if len(patients_df) > 0:
            st.info(f"📊 Total predictions made: {len(patients_df)}")
            
            # Add risk category
            def get_risk_label(risk_str):
                try:
                    risk_val = float(str(risk_str).strip().rstrip('%')) / 100 if '%' in str(risk_str) else float(risk_str)
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
            
            patients_df["Risk_Category"] = patients_df["Risk_Score"].apply(get_risk_label)
            
            # Display metrics by risk category
            col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
            critical = len(patients_df[patients_df["Risk_Category"] == "🔴 CRITICAL"])
            high = len(patients_df[patients_df["Risk_Category"] == "🟠 HIGH"])
            moderate = len(patients_df[patients_df["Risk_Category"] == "🟡 MODERATE"])
            low = len(patients_df[patients_df["Risk_Category"] == "🟢 LOW"])
            
            with col_risk1:
                st.metric("🔴 Critical", critical)
            with col_risk2:
                st.metric("🟠 High", high)
            with col_risk3:
                st.metric("🟡 Moderate", moderate)
            with col_risk4:
                st.metric("🟢 Low", low)
            
            # Display all predicted patients table
            st.markdown("#### 📋 Predicted Patient Records")
            
            # Select columns to display
            display_cols = ["Name", "Age", "Gender", "Severity", "Mutation", "Dose", "Exposure", 
                          "Family_History", "Previous_Inhibitor", "Risk_Score", "Risk_Category"]
            
            available_cols = [col for col in display_cols if col in patients_df.columns]
            patients_display = patients_df[available_cols].copy()
            
            # Add filtering options
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=patients_df["Severity"].unique() if "Severity" in patients_df.columns else [],
                    default=patients_df["Severity"].unique() if "Severity" in patients_df.columns else []
                )
            
            with col_filter2:
                risk_filter = st.multiselect(
                    "Filter by Risk Category",
                    options=["🔴 CRITICAL", "🟠 HIGH", "🟡 MODERATE", "🟢 LOW"],
                    default=["🔴 CRITICAL", "🟠 HIGH", "🟡 MODERATE", "🟢 LOW"]
                )
            
            with col_filter3:
                age_range = st.slider("Age Range", 0, 100, (0, 100))
            
            # Apply filters
            filtered_patients = patients_display.copy()
            
            if "Severity" in filtered_patients.columns and severity_filter:
                filtered_patients = filtered_patients[filtered_patients["Severity"].isin(severity_filter)]
            
            if "Risk_Category" in filtered_patients.columns and risk_filter:
                filtered_patients = filtered_patients[filtered_patients["Risk_Category"].isin(risk_filter)]
            
            if "Age" in filtered_patients.columns:
                filtered_patients = filtered_patients[
                    (filtered_patients["Age"] >= age_range[0]) & 
                    (filtered_patients["Age"] <= age_range[1])
                ]
            
            st.dataframe(
                filtered_patients,
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Download option
            col_download1, col_download2 = st.columns(2)
            with col_download1:
                csv_data = filtered_patients.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_data,
                    file_name="predicted_patients.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("📭 No predicted patients yet. Create predictions from Patient Form to populate this section.")
    except FileNotFoundError:
        st.info("📁 No predicted patient data available yet.")
    except Exception as e:
        st.warning(f"⚠️ Error loading patient data: {str(e)[:50]}")
    
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
        
        # Export options
        st.markdown("#### 📤 Export Options")
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            if st.button("📥 Export All Patients (CSV)", use_container_width=True):
                patients_list = get_all_patients()
                df_export = pd.DataFrame(patients_list)
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Patients CSV",
                    data=csv,
                    file_name=f"all_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with export_col2:
            if st.button("📊 Export System Statistics", use_container_width=True):
                patients_list = get_all_patients()
                if patients_list:
                    df = pd.DataFrame(patients_list)
                    stats_data = {
                        'Metric': [
                            'Total Patients',
                            'Average Age',
                            'Male Patients',
                            'Female Patients',
                            'Average Risk Score',
                            'Highest Risk Score',
                            'Lowest Risk Score',
                            'Mild Cases',
                            'Moderate Cases',
                            'Severe Cases',
                            'Average Treatment Adherence',
                            'Report Generated'
                        ],
                        'Value': [
                            len(df),
                            f"{df['age'].mean():.1f}" if 'age' in df.columns else 'N/A',
                            len(df[df['gender'] == 'Male']) if 'gender' in df.columns else 0,
                            len(df[df['gender'] == 'Female']) if 'gender' in df.columns else 0,
                            f"{df['risk_score'].mean():.1%}" if 'risk_score' in df.columns else 'N/A',
                            f"{df['risk_score'].max():.1%}" if 'risk_score' in df.columns else 'N/A',
                            f"{df['risk_score'].min():.1%}" if 'risk_score' in df.columns else 'N/A',
                            len(df[df['severity'] == 'Mild']) if 'severity' in df.columns else 0,
                            len(df[df['severity'] == 'Moderate']) if 'severity' in df.columns else 0,
                            len(df[df['severity'] == 'Severe']) if 'severity' in df.columns else 0,
                            f"{df['treatment_adherence'].mean():.1f}%" if 'treatment_adherence' in df.columns else 'N/A',
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ]
                    }
                    stats_df = pd.DataFrame(stats_data)
                    csv = stats_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Statistics",
                        data=csv,
                        file_name=f"system_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        st.divider()
        st.markdown("#### 🔍 Patient Summary View")
        
        # Quick summary
        patients_list = get_all_patients()
        if patients_list:
            df_summary = pd.DataFrame(patients_list)
            
            summary_text = f"""
            **System Summary:**
            - Total Patients: {len(df_summary)}
            - Average Age: {df_summary['age'].mean():.1f} years
            - Male/Female: {len(df_summary[df_summary['gender'] == 'Male'])} / {len(df_summary[df_summary['gender'] == 'Female'])}
            - Average Risk: {df_summary['risk_score'].mean():.1%}
            - Risk Range: {df_summary['risk_score'].min():.1%} - {df_summary['risk_score'].max():.1%}
            """
            st.markdown(summary_text)

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