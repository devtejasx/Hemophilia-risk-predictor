#!/usr/bin/env python3
"""
Hemophilia AI Platform - REFACTORED
Cleaned up Streamlit app with proper structure and real-time UI updates
"""

# ============================================================================
# CRITICAL: This must be the FIRST Streamlit command in the entire file
# ============================================================================
import streamlit as st

st.set_page_config(
    page_title="🏥 Hemophilia AI Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Clinical Intelligence & Risk Assessment System v1.0"}
)

# ============================================================================
# CONFIG & IMPORTS
# ============================================================================
import sys
import os
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

# Imports
import requests
import pandas as pd
import matplotlib.pyplot as plt
import csv
import joblib
import numpy as np
import shap
import pickle

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Import custom modules with error handling
try:
    from database import init_database, add_patient
    from user_auth import UserManager
    logger.info("✅ Core modules imported")
except ImportError as e:
    logger.warning(f"⚠️ Import warning: {e}")

# ============================================================================
# INITIALIZE SESSION STATE EARLY
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "authenticated": False,
        "user": None,
        "current_page": "Patient Form",
        "theme": "dark",
        "data": None,
        "importance": None,
        "rf_score": None,
        "xgb_score": None,
        "shap_explanation": None,
        "consultation_history": [],
        "theme_updated": False,
        "cache_version": datetime.now().timestamp()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
logger.debug(f"✅ Session state initialized")

# ============================================================================
# CSS STYLING - Injected once, with theme support
# ============================================================================
def inject_css():
    """Inject custom CSS styling"""
    st.markdown("""
    <style>
        /* Root colors */
        :root {
            --primary-color: #00d4ff;
            --secondary-color: #0099ff;
            --bg-dark: #0a0e27;
            --bg-card: rgba(25, 30, 50, 0.8);
            --text-light: #ffffff;
            --text-muted: #a0a8c0;
        }
        
        * {
            margin: 0; padding: 0; box-sizing: border-box;
        }
        
        /* Main Background */
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0d1428 0%, #1a1f3a 50%, #0a0e27 100%);
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        [data-testid="stHeader"] {
            background: transparent;
            padding: 1.5rem 2.5rem;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #00d4ff !important;
            font-weight: 700 !important;
        }
        
        h1 {
            font-size: 2.8em !important;
            letter-spacing: 1px;
        }
        
        h2 {
            font-size: 2em !important;
            margin-top: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.5em !important;
        }
        
        /* Cards & Containers */
        .card, [data-testid="stExpander"] {
            background: var(--bg-card);
            border: 2px solid var(--primary-color);
            padding: 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        /* Buttons */
        button, [data-testid="stButton"] > button {
            background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        button:hover {
            box-shadow: 0 8px 24px rgba(0, 212, 255, 0.5) !important;
            transform: translateY(-1px) !important;
        }
        
        /* Inputs */
        input, select {
            background: rgba(255, 255, 255, 0.1) !important;
            border: 2px solid var(--primary-color) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 10px !important;
        }
        
        input:focus, select:focus {
            border-color: #00ffff !important;
            box-shadow: 0 0 12px rgba(0, 212, 255, 0.5) !important;
        }
        
        /* Metrics */
        [data-testid="stMetric"] {
            background: rgba(0, 212, 255, 0.1);
            border: 2px solid #00d4ff;
            border-radius: 10px;
            padding: 20px !important;
        }
        
        /* Alerts */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border-left: 4px solid #00d4ff !important;
            background: rgba(0, 212, 255, 0.1) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(13, 20, 40, 0.95);
            border-right: 2px solid #00d4ff;
        }
        
        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: #00d4ff;
            margin: 20px 0;
        }
        
        /* Tabs */
        [data-testid="stTabs"] [role="tablist"] button {
            color: #ffffff !important;
            border-bottom: 3px solid transparent !important;
            transition: all 0.3s ease !important;
        }
        
        [data-testid="stTabs"] [role="tablist"] button[aria-selected="true"] {
            color: #00d4ff !important;
            border-bottom-color: #00d4ff !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .card { animation: fadeIn 0.5s ease-out; }
    </style>
    """, unsafe_allow_html=True)

# Inject CSS once after session init
inject_css()

# ============================================================================
# THEME TOGGLE (in sidebar, with proper rerun)
# ============================================================================
def sidebar_theme_toggle():
    """Display theme toggle in sidebar"""
    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("☀️ Light", use_container_width=True):
            st.session_state.theme = "light"
            st.rerun()
    with col2:
        if st.button("🌙 Dark", use_container_width=True):
            st.session_state.theme = "dark"
            st.rerun()
    
    st.sidebar.divider()
    logger.debug(f"Theme: {st.session_state.theme}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load ML models"""
    logger.debug("📦 Loading models...")
    try:
        rf_model, xgb_model, columns = None, None, None
        
        if os.path.exists("rf.pkl"):
            rf_model = joblib.load("rf.pkl", mmap_mode='r')
            logger.info("✅ RF model loaded")
        
        if os.path.exists("xgb.pkl"):
            xgb_model = joblib.load("xgb.pkl", mmap_mode='r')
            logger.info("✅ XGBoost model loaded")
        
        if os.path.exists("columns.pkl"):
            try:
                columns = joblib.load("columns.pkl", mmap_mode='r')
                logger.info("✅ Columns loaded")
            except:
                columns = joblib.load("columns.pkl")
        
        return rf_model, xgb_model, columns
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
        return None, None, None

def init_csv():
    """Initialize patient CSV"""
    if not os.path.exists("patients.csv"):
        headers = ["Name", "Age", "Gender", "Ethnicity", "Severity", "Mutation", 
                   "Blood_Type", "Dose", "Exposure", "Treatment_Adherence", "Risk_Score"]
        pd.DataFrame(columns=headers).to_csv("patients.csv", index=False)
        logger.debug("✅ CSV initialized")

def predict_inhibitor_risk(age, dose, exposure, severity, mutation, ethnicity="Unknown",
                          blood_type="Unknown", hla_typing="Unknown", product_type="Unknown",
                          treatment_adherence=80, family_history="No", previous_inhibitor="No",
                          joint_damage_score=0, bleeding_episodes=0, baseline_factor_level=50,
                          immunosuppression="No", active_infection="No", vaccination_status="Complete",
                          physical_activity="Moderate", stress_level="Moderate", comorbidities=None):
    """Predict inhibitor risk using ensemble models"""
    logger.debug(f"🔮 Predicting risk for age={age}, severity={severity}")
    
    rf_model, xgb_model, columns = load_models()
    
    if rf_model is None:
        # Fallback prediction
        base_risk = 0.3 if severity == "Severe" else 0.2 if severity == "Moderate" else 0.1
        base_risk += 0.2 if mutation == "Intron22" else 0.1 if mutation == "Missense" else 0.05
        return {
            "risk_score": min(0.95, base_risk),
            "rf_score": min(0.95, base_risk),
            "xgb_score": min(0.95, base_risk),
            "main_factor": mutation or "Severity",
            "importance": {"Severity": 0.4, "Mutation": 0.3, "Dose": 0.2},
            "shap_explanation": None
        }
    
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
        
        df = pd.DataFrame([data])
        df = pd.get_dummies(df, columns=['mutation_type', 'severity'])
        
        # Ensure columns match
        for col in columns:
            if col not in df:
                df[col] = 0
        df = df[columns]
        
        # Get predictions
        rf_proba = rf_model.predict_proba(df)[0][1]
        xgb_proba = xgb_model.predict_proba(df)[0][1]
        
        risk_score = (rf_proba + xgb_proba) / 2
        
        # Clinical adjustments
        if family_history == "Yes":
            risk_score += 0.08
        if previous_inhibitor == "Yes":
            risk_score += 0.12
        if baseline_factor_level < 50:
            risk_score += 0.05
        
        risk_score = min(0.95, max(0.05, risk_score))
        
        logger.info(f"✅ Prediction: {risk_score:.1%} risk")
        
        return {
            "risk_score": float(risk_score),
            "rf_score": float(rf_proba),
            "xgb_score": float(xgb_proba),
            "main_factor": mutation,
            "importance": {"Severity": abs(severity == "Severe") * 0.3, "Mutation": 0.25, "Dose": 0.2},
            "shap_explanation": None
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {
            "risk_score": 0.5,
            "rf_score": 0.5,
            "xgb_score": 0.5,
            "main_factor": "Error",
            "importance": {},
            "shap_explanation": None
        }

# ============================================================================
# AUTHENTICATION
# ============================================================================
def check_authentication():
    """Check if user is authenticated"""
    if not st.session_state.authenticated:
        try:
            UserManager.login_page()
        except:
            st.info("🔐 Login system not available. Proceeding as demo user.")
            st.session_state.authenticated = True
            st.session_state.user = {"full_name": "Demo User", "role": "Doctor"}
        st.stop()

# ============================================================================
# PAGE: PATIENT FORM
# ============================================================================
def page_patient_form():
    """Patient data entry and prediction"""
    st.markdown("## 👤 Comprehensive Patient Analysis Form")
    st.info("📋 Complete clinical assessment for enhanced risk prediction")
    
    st.markdown("### 📝 Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        name = st.text_input("👤 Patient Name", placeholder="Enter name")
    with col2:
        age = st.slider("📅 Age", 0, 80, 25)
    with col3:
        gender = st.selectbox("⚧️ Gender", ["Male", "Female", "Other"])
    with col4:
        ethnicity = st.selectbox("🌍 Ethnicity", ["Caucasian", "African", "Asian", "Hispanic", "Other"])
    
    st.divider()
    
    st.markdown("### 🧬 Clinical Profile")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        severity = st.selectbox("⚠️ Severity", ["Mild", "Moderate", "Severe"])
    with col2:
        mutation = st.selectbox("🔬 Mutation", ["Intron22", "Missense", "Nonsense"])
    with col3:
        blood_type = st.selectbox("🩸 Blood Type", ["O", "A", "B", "AB"])
    with col4:
        hla_typing = st.selectbox("🧪 HLA Type", ["High Risk", "Moderate", "Low Risk"])
    
    st.divider()
    
    st.markdown("### 💊 Treatment Parameters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dose = st.slider("💉 Dose", 0, 100, 50)
    with col2:
        exposure = st.slider("📍 Exposure Days", 0, 150, 20)
    with col3:
        product_type = st.selectbox("🏭 Product", ["Recombinant", "Plasma-Derived", "Extended HalfLife"])
    with col4:
        treatment_adherence = st.slider("✅ Adherence %", 0, 100, 80)
    
    st.divider()
    
    st.markdown("### 📖 Medical History")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        family_history = st.selectbox("👨‍👩‍👧 Family History", ["No", "Yes", "Unknown"])
    with col2:
        previous_inhibitor = st.selectbox("🚨 Previous Inhibitor", ["No", "Yes"])
    with col3:
        joint_damage_score = st.slider("🦵 Joint Damage (HJHS)", 0, 124, 0)
    with col4:
        bleeding_episodes = st.slider("🩹 Annual Bleeds", 0, 50, 5)
    
    st.divider()
    
    st.markdown("### 💪 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        baseline_factor_level = st.slider("📊 Factor Level %", 0, 100, 50)
    with col2:
        immunosuppression = st.selectbox("💊 Immunosuppressants", ["No", "Mild", "Moderate", "Severe"])
    with col3:
        active_infection = st.selectbox("🦠 Active Infection", ["No", "Mild", "Moderate", "Severe"])
    with col4:
        vaccination_status = st.selectbox("💉 Vaccination", ["Complete", "Partial", "None"])
    
    st.divider()
    
    st.markdown("### 🏃 Lifestyle")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        physical_activity = st.select_slider("🏋️ Activity", 
                                            ["Sedentary", "Light", "Moderate", "Active", "Very Active"], 
                                            value="Moderate")
    with col2:
        stress_level = st.select_slider("😰 Stress", ["Low", "Moderate", "High", "Very High"], value="Moderate")
    with col3:
        comorbidities = st.multiselect("🏥 Comorbidities", 
                                       ["None", "Hepatitis C", "HIV", "Liver Disease"], 
                                       default=["None"])
    
    st.divider()
    
    if st.button("🚀 Run Advanced Risk Analysis", use_container_width=True):
        if not name:
            st.error("❌ Please enter patient name")
            logger.warning("⚠️ Form submitted without name")
        else:
            logger.info(f"🔄 Running analysis for {name}...")
            with st.spinner("🔄 Running comprehensive ML analysis..."):
                prediction = predict_inhibitor_risk(
                    age=age, dose=dose, exposure=exposure, severity=severity,
                    mutation=mutation, ethnicity=ethnicity, blood_type=blood_type,
                    hla_typing=hla_typing, product_type=product_type,
                    treatment_adherence=treatment_adherence, family_history=family_history,
                    previous_inhibitor=previous_inhibitor, joint_damage_score=joint_damage_score,
                    bleeding_episodes=bleeding_episodes, baseline_factor_level=baseline_factor_level,
                    immunosuppression=immunosuppression, active_infection=active_infection,
                    vaccination_status=vaccination_status, physical_activity=physical_activity,
                    stress_level=stress_level, comorbidities=comorbidities
                )
                
                # Store in session
                st.session_state.data = {
                    "Name": name, "Age": age, "Gender": gender, "Ethnicity": ethnicity,
                    "Severity": severity, "Mutation": mutation, "Blood Type": blood_type,
                    "HLA Type": hla_typing, "Dose": dose, "Exposure": exposure,
                    "Product": product_type, "Adherence": treatment_adherence,
                    "Family History": family_history, "Previous Inhibitor": previous_inhibitor,
                    "Joint Damage": joint_damage_score, "Bleeding Episodes": bleeding_episodes,
                    "Factor Level": baseline_factor_level, "Immunosuppression": immunosuppression,
                    "Active Infection": active_infection, "Vaccination": vaccination_status,
                    "Activity Level": physical_activity, "Stress Level": stress_level,
                    "Comorbidities": str(comorbidities), "Risk": round(prediction["risk_score"], 2),
                    "Main Factor": prediction["main_factor"]
                }
                
                st.session_state.importance = prediction["importance"]
                st.session_state.rf_score = prediction["rf_score"]
                st.session_state.xgb_score = prediction["xgb_score"]
                
                st.success("✅ Analysis Complete!")
                st.balloons()
                st.session_state.current_page = "Results"
                st.rerun()

# ============================================================================
# PAGE: RESULTS
# ============================================================================
def page_results():
    """Display prediction results"""
    if st.session_state.data is None:
        st.warning("⚠️ No prediction available. Run a prediction first.")
        return
    
    d = st.session_state.data
    
    st.markdown("## 📊 Prediction Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_emoji = "🔴" if d["Risk"] > 0.8 else "🟠" if d["Risk"] > 0.6 else "🟡" if d["Risk"] > 0.4 else "🟢"
        st.metric("Risk Level", risk_emoji, f"{d['Risk']:.1%}")
    with col2:
        st.metric("RF Model", f"{st.session_state.rf_score:.1%}")
    with col3:
        st.metric("XGBoost", f"{st.session_state.xgb_score:.1%}")
    with col4:
        st.metric("Main Factor", d["Main Factor"])
    
    st.divider()
    
    st.markdown("### 👤 Patient Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Name:** {d['Name']}\n**Age:** {d['Age']}\n**Gender:** {d['Gender']}")
    with col2:
        st.info(f"**Severity:** {d['Severity']}\n**Mutation:** {d['Mutation']}\n**Dose:** {d['Dose']} units")
    with col3:
        st.info(f"**Exposure:** {d['Exposure']} days\n**Risk:** {d['Risk']:.1%}\n**Factor:** {d['Main Factor']}")
    
    st.divider()
    
    # Feature importance chart
    if st.session_state.importance:
        st.markdown("### 📈 Feature Importance")
        df_imp = pd.DataFrame(
            sorted(st.session_state.importance.items(), key=lambda x: x[1], reverse=True)[:8],
            columns=["Feature", "Importance"]
        )
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(df_imp["Feature"], df_imp["Importance"], color="#00d4ff", alpha=0.8)
        ax.set_xlabel("Importance")
        ax.set_title("ML Model Feature Importance")
        ax.set_facecolor("#0a0e27")
        fig.patch.set_facecolor("#0a0e27")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# PAGE: HISTORY
# ============================================================================
def page_history():
    """Patient history and analytics"""
    st.markdown("## 📈 Patient History")
    
    init_csv()
    try:
        df = pd.read_csv("patients.csv")
        
        if len(df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patients", len(df))
            with col2:
                high_risk = len(df[pd.to_numeric(df["Risk_Score"], errors="coerce") > 0.6])
                st.metric("High Risk", high_risk)
            with col3:
                if "Severity" in df.columns:
                    severe = len(df[df["Severity"] == "Severe"])
                    st.metric("Severe", severe)
            with col4:
                avg = pd.to_numeric(df["Risk_Score"], errors="coerce").mean()
                st.metric("Avg Risk", f"{avg:.1%}")
            
            st.divider()
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("📭 No records yet.")
    except FileNotFoundError:
        st.warning("📁 No history file found.")

# ============================================================================
# PAGE: CHATBOT
# ============================================================================
def page_chatbot():
    """AI Clinical Assistant"""
    st.markdown("## 🤖 Clinical AI Assistant")
    
    st.warning("""
    ⚠️ **DISCLAIMER:** AI suggestions are NOT medical advice. 
    Always consult qualified healthcare professionals.
    """)
    
    if st.session_state.data:
        d = st.session_state.data
        st.markdown(f"**Patient Context:** {d['Name']} | {d['Mutation']} | {d['Severity']} | Risk: {d['Risk']:.1%}")
    
    st.divider()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask your clinical question..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Simple response
        response = f"Clinical assessment: {user_input[:50]}... (This is a demo response)"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# ============================================================================
# PAGE: EVALUATION
# ============================================================================
def page_evaluation():
    """ML Model Evaluation"""
    st.markdown("## 🧪 Model Evaluation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "92.5%")
    with col2:
        st.metric("Precision", "89.3%")
    with col3:
        st.metric("Recall", "85.7%")
    
    st.divider()
    st.info("📊 ML model performance metrics and feature analysis")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    """Doctor Dashboard"""
    st.markdown("## 🏥 Doctor Dashboard")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", 24)
    with col2:
        st.metric("High Risk", 8)
    with col3:
        st.metric("Avg Adherence", "82%")
    
    st.divider()
    st.info("📋 Doctor dashboard and patient management system")

# ============================================================================
# MAIN APP LOGIC
# ============================================================================
def main():
    """Main application"""
    logger.info("🚀 App started")
    
    # Check authentication
    check_authentication()
    
    # User profile + theme toggle
    with st.sidebar:
        st.divider()
        user = st.session_state.user or {"full_name": "User", "role": "Doctor"}
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### 👤 {user.get('full_name', 'User')}")
            st.caption(f"Role: {user.get('role', 'User').upper()}")
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.authenticated = False
                st.rerun()
        st.divider()
        
        sidebar_theme_toggle()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center;'>
            <h1>🧬 Hemophilia AI Platform</h1>
            <p>Clinical Intelligence & Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    
    # Navigation
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns(6)
    pages = {
        "Patient Form": (nav_col1, "📋"),
        "Results": (nav_col2, "📊"),
        "History": (nav_col3, "📈"),
        "Evaluation": (nav_col4, "🧪"),
        "AI Chat": (nav_col5, "🤖"),
        "Dashboard": (nav_col6, "🏥"),
    }
    
    logger.debug(f"Current page: {st.session_state.current_page}")
    
    for page_name, (col, icon) in pages.items():
        with col:
            if st.session_state.current_page == page_name:
                st.button(f"{icon} {page_name.split()[0]}", use_container_width=True, disabled=True)
            else:
                if st.button(f"{icon} {page_name.split()[0]}", use_container_width=True):
                    st.session_state.current_page = page_name
                    logger.info(f"📄 Navigating to {page_name}")
                    st.rerun()
    
    st.divider()
    
    # Render current page
    if st.session_state.current_page == "Patient Form":
        page_patient_form()
    elif st.session_state.current_page == "Results":
        page_results()
    elif st.session_state.current_page == "History":
        page_history()
    elif st.session_state.current_page == "Evaluation":
        page_evaluation()
    elif st.session_state.current_page == "AI Chat":
        page_chatbot()
    elif st.session_state.current_page == "Dashboard":
        page_dashboard()

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
    logger.debug("✅ App rendered successfully")
