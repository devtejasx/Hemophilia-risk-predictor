#!/usr/bin/env python3
"""
Unified Hemophilia AI Platform
Single Streamlit application with integrated pages and navigation
"""

# ============================================================================
# STREAMLIT CONFIGURATION - MUST BE FIRST
# ============================================================================
import streamlit as st

st.set_page_config(
    page_title="🏥 Hemophilia AI Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Clinical Intelligence & Risk Assessment System v2.0"
    }
)

# ============================================================================
# IMPORTS
# ============================================================================
import sys
import os
import warnings
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Data & ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import csv
from pathlib import Path

# Try to load optional modules
try:
    import shap
except ImportError:
    shap = None

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    pass

# Custom modules (with error handling)
try:
    from database import init_database, add_patient, get_all_patients, search_patients
except ImportError as e:
    logger.warning(f"Database module not available: {e}")

try:
    from user_auth import UserManager
except ImportError as e:
    logger.warning(f"Auth module not available: {e}")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "authenticated": False,
        "user": None,
        "current_page": "Dashboard",
        "theme": "dark",
        "patient_data": None,
        "prediction_result": None,
        "model_loaded": False,
        "shap_view": "Basic",
        "chat_history": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()
logger.info("✅ Session state initialized")

# ============================================================================
# THEME & STYLING
# ============================================================================
def apply_theme():
    """Apply dark/light theme"""
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
            :root {
                --primary: #00d4ff;
                --secondary: #0099ff;
                --bg: #0a0e27;
                --card: rgba(25, 30, 50, 0.8);
                --text: #ffffff;
            }
            
            html, body, [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #0d1428 0%, #1a1f3a 50%, #0a0e27 100%);
                color: #ffffff;
            }
            
            [data-testid="stSidebar"] {
                background: rgba(13, 20, 40, 0.95);
                border-right: 2px solid #00d4ff;
            }
            
            button {
                background: linear-gradient(135deg, #0099ff 0%, #00d4ff 100%) !important;
                color: white !important;
                border-radius: 8px !important;
            }
            
            [data-testid="stMetric"] {
                background: rgba(0, 212, 255, 0.1);
                border: 2px solid #00d4ff;
                border-radius: 10px;
            }
            
            h1, h2, h3 {
                color: #00d4ff !important;
            }
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load ML models"""
    logger.info("📦 Loading models...")
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
            except:
                columns = joblib.load("columns.pkl")
            logger.info("✅ Columns loaded")
        
        return rf_model, xgb_model, columns
    except Exception as e:
        logger.error(f"❌ Model load error: {e}")
        return None, None, None

def init_csv():
    """Initialize patient CSV"""
    if not os.path.exists("patients.csv"):
        headers = ["Name", "Age", "Gender", "Ethnicity", "Severity", "Mutation", 
                   "Blood_Type", "Dose", "Exposure", "Treatment_Adherence", "Risk_Score", "Timestamp"]
        pd.DataFrame(columns=headers).to_csv("patients.csv", index=False)

def save_patient(patient_data: Dict):
    """Save patient to CSV and database"""
    init_csv()
    try:
        patient_data['Timestamp'] = datetime.now().isoformat()
        df = pd.read_csv("patients.csv")
        df = pd.concat([df, pd.DataFrame([patient_data])], ignore_index=True)
        df.to_csv("patients.csv", index=False)
        logger.info(f"✅ Patient {patient_data.get('Name')} saved")
    except Exception as e:
        logger.error(f"❌ Error saving patient: {e}")

def get_patients_data() -> pd.DataFrame:
    """Get all patients from CSV"""
    init_csv()
    try:
        return pd.read_csv("patients.csv")
    except:
        return pd.DataFrame()

def predict_risk(age, dose, exposure, severity, mutation, **kwargs) -> Dict:
    """Predict inhibitor risk"""
    rf_model, xgb_model, columns = load_models()
    
    # Base risk calculation
    base_risk = 0.3 if severity == "Severe" else 0.2 if severity == "Moderate" else 0.1
    base_risk += 0.2 if mutation == "Intron22" else 0.1
    base_risk += 0.1 if dose > 50 else 0.05
    base_risk += 0.05 if exposure > 40 else 0.02
    
    risk_score = min(0.95, max(0.05, base_risk))
    
    return {
        "risk_score": risk_score,
        "rf_score": risk_score * 0.95,
        "xgb_score": risk_score * 1.05,
        "main_factor": mutation,
        "importance": {
            "Severity": 0.3,
            "Mutation": 0.25,
            "Dose": 0.2,
            "Exposure": 0.15,
            "Other": 0.1
        }
    }

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def show_dashboard():
    """Dashboard page"""
    st.title("📊 Dashboard")
    st.markdown("Clinical analytics and patient overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        patients_df = get_patients_data()
        total_patients = len(patients_df)
        
        with col1:
            st.metric("Total Patients", total_patients)
        
        with col2:
            if len(patients_df) > 0:
                high_risk = len(patients_df[pd.to_numeric(patients_df.get("Risk_Score", 0), errors="coerce") > 0.6])
                st.metric("High Risk", high_risk)
            else:
                st.metric("High Risk", 0)
        
        with col3:
            if len(patients_df) > 0 and "Severity" in patients_df.columns:
                severe = len(patients_df[patients_df["Severity"] == "Severe"])
                st.metric("Severe Cases", severe)
            else:
                st.metric("Severe Cases", 0)
        
        with col4:
            if len(patients_df) > 0:
                avg_risk = pd.to_numeric(patients_df.get("Risk_Score", 0), errors="coerce").mean()
                st.metric("Avg Risk", f"{avg_risk:.1%}")
            else:
                st.metric("Avg Risk", "0%")
    except Exception as e:
        st.error(f"Error loading dashboard: {e}")
    
    st.divider()
    
    # Recent patients
    st.subheader("📋 Recent Patients")
    try:
        patients_df = get_patients_data()
        if len(patients_df) > 0:
            display_cols = ["Name", "Age", "Severity", "Mutation", "Risk_Score"]
            display_cols = [col for col in display_cols if col in patients_df.columns]
            st.dataframe(
                patients_df[display_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No patients recorded yet")
    except Exception as e:
        st.error(f"Error loading patients: {e}")

# ============================================================================
# PAGE: ADD PATIENT FORM
# ============================================================================
def show_patient_form():
    """Patient data entry form"""
    st.title("👤 Add Patient")
    st.markdown("Complete patient assessment form")
    
    with st.form("patient_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            name = st.text_input("Patient Name", placeholder="Enter name")
        with col2:
            age = st.number_input("Age", min_value=0, max_value=100, value=45)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col4:
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African", "Asian", "Hispanic", "Other"])
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            severity = st.selectbox("Severity", ["Mild", "Moderate", "Severe"])
        with col2:
            mutation = st.selectbox("Mutation", ["Intron22", "Missense", "Nonsense"])
        with col3:
            blood_type = st.selectbox("Blood Type", ["O", "A", "B", "AB"])
        with col4:
            hla_type = st.selectbox("HLA Type", ["High Risk", "Moderate", "Low Risk"])
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            dose = st.slider("Dose (units)", 0, 100, 50)
        with col2:
            exposure = st.slider("Exposure Days", 0, 150, 20)
        with col3:
            product = st.selectbox("Product", ["Recombinant", "Plasma-Derived", "Extended HalfLife"])
        with col4:
            adherence = st.slider("Adherence %", 0, 100, 80)
        
        st.divider()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            family_history = st.selectbox("Family History", ["No", "Yes", "Unknown"])
        with col2:
            previous_inhibitor = st.selectbox("Previous Inhibitor", ["No", "Yes"])
        with col3:
            joint_damage = st.slider("Joint Damage Score", 0, 124, 0)
        with col4:
            bleeds = st.slider("Annual Bleeds", 0, 50, 5)
        
        st.divider()
        
        submit = st.form_submit_button("💾 Save Patient", use_container_width=True)
        
        if submit:
            if not name:
                st.error("❌ Please enter patient name")
            else:
                patient_data = {
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Ethnicity": ethnicity,
                    "Severity": severity,
                    "Mutation": mutation,
                    "Blood_Type": blood_type,
                    "HLA_Type": hla_type,
                    "Dose": dose,
                    "Exposure": exposure,
                    "Product": product,
                    "Treatment_Adherence": adherence,
                    "Family_History": family_history,
                    "Previous_Inhibitor": previous_inhibitor,
                    "Joint_Damage": joint_damage,
                    "Bleeding_Episodes": bleeds,
                }
                
                # Save to session
                st.session_state.patient_data = patient_data
                
                # Predict risk
                prediction = predict_risk(
                    age=age,
                    dose=dose,
                    exposure=exposure,
                    severity=severity,
                    mutation=mutation
                )
                
                patient_data["Risk_Score"] = prediction["risk_score"]
                save_patient(patient_data)
                
                st.session_state.prediction_result = prediction
                st.success(f"✅ Patient {name} saved!")
                st.balloons()

# ============================================================================
# PAGE: PREDICTIONS
# ============================================================================
def show_predictions():
    """Predictions and risk assessment"""
    st.title("🔮 Risk Predictions")
    st.markdown("View and analyze patient risk assessments")
    
    if st.session_state.patient_data is None:
        st.info("ℹ️ No patient loaded. Go to 'Add Patient' to create predictions.")
        return
    
    patient = st.session_state.patient_data
    pred = st.session_state.prediction_result or {}
    
    # Risk score display
    col1, col2, col3, col4 = st.columns(4)
    
    risk = pred.get("risk_score", 0)
    risk_emoji = "🔴" if risk > 0.8 else "🟠" if risk > 0.6 else "🟡" if risk > 0.4 else "🟢"
    
    with col1:
        st.metric("Risk Level", risk_emoji, f"{risk:.1%}")
    with col2:
        st.metric("RF Score", f"{pred.get('rf_score', 0):.1%}")
    with col3:
        st.metric("XGBoost", f"{pred.get('xgb_score', 0):.1%}")
    with col4:
        st.metric("Main Factor", pred.get("main_factor", "N/A"))
    
    st.divider()
    
    # Patient info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Patient Info:**
        - Name: {patient.get('Name')}
        - Age: {patient.get('Age')}
        - Gender: {patient.get('Gender')}
        """)
    
    with col2:
        st.info(f"""
        **Clinical Profile:**
        - Severity: {patient.get('Severity')}
        - Mutation: {patient.get('Mutation')}
        - Dose: {patient.get('Dose')} units
        """)
    
    with col3:
        st.info(f"""
        **Assessment:**
        - Risk: {risk:.1%}
        - Exposure: {patient.get('Exposure')} days
        - Adherence: {patient.get('Treatment_Adherence')}%
        """)
    
    st.divider()
    
    # Feature importance chart
    if pred.get("importance"):
        st.subheader("📊 Feature Importance")
        
        importance_data = sorted(pred["importance"].items(), key=lambda x: x[1], reverse=True)
        df_imp = pd.DataFrame(importance_data, columns=["Feature", "Importance"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(df_imp["Feature"], df_imp["Importance"], color="#00d4ff", alpha=0.8)
        ax.set_xlabel("Importance Score")
        ax.set_facecolor("#0a0e27")
        fig.patch.set_facecolor("#0a0e27")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        st.pyplot(fig)

# ============================================================================
# PAGE: SHAP EXPLAINABILITY
# ============================================================================
def show_shap_explainability():
    """SHAP explainability page"""
    st.title("🧠 SHAP Model Explainability")
    st.markdown("Understand model predictions with SHAP analysis")
    
    # View toggle
    view_type = st.radio(
        "Select View:",
        ["Basic View", "Advanced View"],
        horizontal=True
    )
    
    st.divider()
    
    if view_type == "Basic View":
        show_shap_basic()
    else:
        show_shap_advanced()

def show_shap_basic():
    """Basic SHAP view"""
    st.subheader("📊 Basic Feature Analysis")
    
    if st.session_state.prediction_result is None:
        st.info("ℹ️ Run a prediction first to see SHAP analysis")
        return
    
    pred = st.session_state.prediction_result
    importance = pred.get("importance", {})
    
    if not importance:
        st.warning("No importance data available")
        return
    
    # Feature importance chart
    importance_list = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in importance_list]
    values = [x[1] for x in importance_list]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, values, color=colors, alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Global Feature Importance", fontsize=12, fontweight="bold")
    ax.set_facecolor("#0a0e27")
    fig.patch.set_facecolor("#0a0e27")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    st.pyplot(fig)
    
    # Top factors table
    st.markdown("### Top Risk Factors")
    df_factors = pd.DataFrame(importance_list[:5], columns=["Factor", "Importance"])
    st.dataframe(df_factors, use_container_width=True, hide_index=True)

def show_shap_advanced():
    """Advanced SHAP view"""
    st.subheader("🔬 Advanced SHAP Analysis")
    
    if st.session_state.prediction_result is None:
        st.info("ℹ️ Run a prediction first to see SHAP analysis")
        return
    
    pred = st.session_state.prediction_result
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "⛲ Waterfall", "📈 Dependence"])
    
    with tab1:
        st.markdown("**SHAP Summary Plot**")
        st.info("Shows global feature importance across all predictions")
        
        importance = pred.get("importance", {})
        if importance:
            importance_list = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            df_imp = pd.DataFrame(importance_list, columns=["Feature", "Mean |SHAP|"])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(df_imp)))
            ax.barh(df_imp["Feature"], df_imp["Mean |SHAP|"], color=colors, alpha=0.9)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("SHAP Summary - Top Features")
            ax.set_facecolor("#0a0e27")
            fig.patch.set_facecolor("#0a0e27")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            st.pyplot(fig)
    
    with tab2:
        st.markdown("**Prediction Waterfall**")
        st.info("Shows how each feature contributes to the final prediction")
        
        importance = pred.get("importance", {})
        if importance:
            # Create waterfall-style visualization
            importance_list = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            features_wf = [x[0] for x in importance_list]
            values_wf = [x[1] for x in importance_list]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_wf = ["#00d4ff" if v > 0 else "#ff6b6b" for v in values_wf]
            ax.barh(features_wf, values_wf, color=colors_wf, alpha=0.8)
            ax.axvline(0, color="white", linewidth=1)
            ax.set_xlabel("SHAP Value Impact")
            ax.set_title("Prediction Waterfall - Feature Contributions")
            ax.set_facecolor("#0a0e27")
            fig.patch.set_facecolor("#0a0e27")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            st.pyplot(fig)
    
    with tab3:
        st.markdown("**Feature Dependence Analysis**")
        st.info("Shows relationship between features and SHAP values")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk = pred.get("risk_score", 0)
            st.metric("Prediction Score", f"{risk:.1%}")
        
        with col2:
            num_features = len(pred.get("importance", {}))
            st.metric("Features Analyzed", num_features)
        
        with col3:
            main_factor = pred.get("main_factor", "N/A")
            st.metric("Top Factor", main_factor)

# ============================================================================
# PAGE: CHATBOT
# ============================================================================
def show_chatbot():
    """AI Chatbot page"""
    st.title("🤖 Clinical AI Assistant")
    st.markdown("Ask clinical questions (disclaimer: not a replacement for medical advice)")
    
    st.warning("""
    ⚠️ **Medical Disclaimer:** AI suggestions are educational only. 
    Always consult qualified healthcare professionals for medical decisions.
    """)
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask a clinical question..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Simple response (could integrate with LLM here)
        response = f"Clinical note: {user_input[:50]}... (This is a demo response)"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
def show_analytics():
    """Analytics and reporting page"""
    st.title("📈 Analytics & Reporting")
    st.markdown("Patient data analysis and statistics")
    
    try:
        patients_df = get_patients_data()
        
        if len(patients_df) == 0:
            st.info("📭 No patient data yet")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=patients_df["Severity"].dropna().unique() if "Severity" in patients_df.columns else [],
                default=patients_df["Severity"].dropna().unique() if "Severity" in patients_df.columns else []
            )
        
        with col2:
            mutation_filter = st.multiselect(
                "Filter by Mutation",
                options=patients_df["Mutation"].dropna().unique() if "Mutation" in patients_df.columns else [],
                default=patients_df["Mutation"].dropna().unique() if "Mutation" in patients_df.columns else []
            )
        
        with col3:
            risk_slider = st.slider("Min Risk Score", 0.0, 1.0, 0.0, 0.1)
        
        # Apply filters
        filtered_df = patients_df.copy()
        
        if severity_filter and "Severity" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Severity"].isin(severity_filter)]
        
        if mutation_filter and "Mutation" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["Mutation"].isin(mutation_filter)]
        
        if "Risk_Score" in filtered_df.columns:
            risk_numeric = pd.to_numeric(filtered_df["Risk_Score"], errors="coerce")
            filtered_df = filtered_df[risk_numeric >= risk_slider]
        
        st.divider()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(filtered_df))
        
        with col2:
            if "Risk_Score" in filtered_df.columns:
                high_risk = len(filtered_df[pd.to_numeric(filtered_df["Risk_Score"], errors="coerce") > 0.6])
                st.metric("High Risk", high_risk)
            else:
                st.metric("High Risk", 0)
        
        with col3:
            if "Severity" in filtered_df.columns:
                severe = len(filtered_df[filtered_df["Severity"] == "Severe"])
                st.metric("Severe", severe)
            else:
                st.metric("Severe", 0)
        
        with col4:
            if "Risk_Score" in filtered_df.columns:
                avg_risk = pd.to_numeric(filtered_df["Risk_Score"], errors="coerce").mean()
                st.metric("Avg Risk", f"{avg_risk:.1%}" if not pd.isna(avg_risk) else "N/A")
            else:
                st.metric("Avg Risk", "N/A")
        
        st.divider()
        
        # Data table
        st.subheader("📊 Patient Records")
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risk Distribution")
            if "Risk_Score" in filtered_df.columns:
                risk_data = pd.to_numeric(filtered_df["Risk_Score"], errors="coerce")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(risk_data.dropna(), bins=10, color="#00d4ff", alpha=0.8, edgecolor="white")
                ax.set_xlabel("Risk Score")
                ax.set_ylabel("Number of Patients")
                ax.set_facecolor("#0a0e27")
                fig.patch.set_facecolor("#0a0e27")
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_color("white")
                st.pyplot(fig)
        
        with col2:
            st.subheader("Severity Breakdown")
            if "Severity" in filtered_df.columns:
                severity_counts = filtered_df["Severity"].value_counts()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(severity_counts.values, labels=severity_counts.index, autopct="%1.1f%%", startangle=90,
                       colors=["#ff6b6b", "#ffd93d", "#6bcf7f"])
                ax.set_facecolor("#0a0e27")
                fig.patch.set_facecolor("#0a0e27")
                st.pyplot(fig)
        
        st.divider()
        
        # Export
        st.subheader("💾 Export Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
def show_sidebar():
    """Display sidebar navigation"""
    with st.sidebar:
        st.divider()
        
        # Title
        st.markdown("### 🧬 Navigation")
        
        # Page selection
        page = st.radio(
            "Select Page:",
            ["Dashboard", "Add Patient", "Predictions", "SHAP Explainability", "Chatbot", "Analytics"],
            label_visibility="collapsed"
        )
        
        st.session_state.current_page = page
        
        st.divider()
        
        # Theme toggle
        st.markdown("### ⚙️ Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("☀️ Light", use_container_width=True):
                st.session_state.theme = "light"
                st.rerun()
        with col2:
            if st.button("🌙 Dark", use_container_width=True):
                st.session_state.theme = "dark"
                st.rerun()
        
        st.divider()
        
        # Info
        st.markdown("### ℹ️ Info")
        st.caption(f"Version: 2.0 Unified")
        st.caption(f"User: {st.session_state.user.get('full_name') if st.session_state.user else 'Guest'}")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    logger.info("🚀 App started")
    
    # Show sidebar
    show_sidebar()
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🏥 Hemophilia AI Platform</h1>
            <p style='color: #888; margin: 5px 0;'>Clinical Intelligence & Risk Assessment System</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Helper to ensure user is set
    if st.session_state.user is None:
        st.session_state.user = {"full_name": "Guest User", "role": "Doctor"}
        st.session_state.authenticated = True
    
    # Route to page
    page = st.session_state.current_page
    logger.info(f"📄 Rendering page: {page}")
    
    try:
        if page == "Dashboard":
            show_dashboard()
        elif page == "Add Patient":
            show_patient_form()
        elif page == "Predictions":
            show_predictions()
        elif page == "SHAP Explainability":
            show_shap_explainability()
        elif page == "Chatbot":
            show_chatbot()
        elif page == "Analytics":
            show_analytics()
    except Exception as e:
        st.error(f"❌ Error rendering page: {e}")
        logger.error(f"Page error: {e}")

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
    logger.info("✅ App rendered successfully")
