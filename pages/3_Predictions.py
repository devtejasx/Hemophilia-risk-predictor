"""
Predictions Page - Risk assessment and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import metric_card, info_card, empty_state
from components.charts import plot_feature_importance, plot_risk_gauge
from database.db import get_database
from utils.helpers import get_risk_level
from services.ml_service import get_ml_service

st.set_page_config(page_title="Predictions", layout="wide")


def main():
    show_page_header("🔮 Risk Predictions", "Clinical risk assessment and predictions")
    
    # Load patients
    try:
        db = get_database()
        patients = db.get_patients()
    except:
        patients = []
    
    if not patients:
        empty_state("📭", "No Patients", "Add patients first to generate predictions")
        return
    
    # Patient selection
    st.markdown("### Select Patient")
    patient_names = [p['name'] for p in patients]
    selected_name = st.selectbox("Choose patient", patient_names)
    
    # Find selected patient
    patient = next((p for p in patients if p['name'] == selected_name), None)
    if not patient:
        st.error("Patient not found")
        return
    
    st.divider()
    
    # Display patient info
    st.markdown("### Patient Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**Name:** {patient.get('name', 'N/A')}")
    with col2:
        st.markdown(f"**Age:** {patient.get('age', 'N/A')} years")
    with col3:
        st.markdown(f"**Severity:** {patient.get('severity', 'N/A')}")
    with col4:
        st.markdown(f"**Mutation:** {patient.get('mutation', 'N/A')}")
    
    st.divider()
    
    # Get ML predictions
    st.markdown("### Risk Assessment")
    
    try:
        ml_service = get_ml_service()
        
        # Prepare features
        features = np.array([
            float(patient.get('age', 0)),
            float(patient.get('dose', 0)),
            float(patient.get('exposure', 0)),
            float(patient.get('treatment_adherence', 50)),
            float(patient.get('risk_score', 0)) / 100 if patient.get('risk_score') else 0,
        ])
        
        # Get predictions
        predictions = ml_service.predict(features)
        risk_score = predictions.get('ensemble_score', patient.get('risk_score', 0))
        rf_score = predictions.get('rf_score', risk_score)
        xgb_score = predictions.get('xgb_score', risk_score)
        
        # Display metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card("Ensemble Score", f"{risk_score:.1f}%", icon="🎯")
        with col2:
            metric_card("RF Model", f"{rf_score:.1f}%", icon="🌲")
        with col3:
            metric_card("XGB Model", f"{xgb_score:.1f}%", icon="⚡")
        
        st.divider()
        
        # Display risk gauge
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_risk_gauge(risk_score)
        
        with col2:
            st.markdown(f"""
            <div style='padding: 20px; background-color: #1a1f3a; border-radius: 10px; height: 100%;'>
                <h3 style='margin: 0 0 20px 0; color: #00d4ff;'>Classification</h3>
                <p style='font-size: 18px; font-weight: bold; margin: 0;'>{get_risk_level(risk_score)}</p>
                <hr style='border-color: #2a3f5f; margin: 15px 0;'>
                <h4 style='color: #888; margin: 10px 0;'>Clinical Factors:</h4>
                <p style='font-size: 12px; color: #ccc; margin: 5px 0;'>✓ Age: {patient.get('age')} years</p>
                <p style='font-size: 12px; color: #ccc; margin: 5px 0;'>✓ Adherence: {patient.get('treatment_adherence', 0):.0f}%</p>
                <p style='font-size: 12px; color: #ccc; margin: 5px 0;'>✓ Exposure: {patient.get('exposure', 0)} events</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Feature Importance
        st.markdown("### Feature Importance")
        importance = ml_service.get_feature_importance()
        
        if importance:
            plot_feature_importance(importance)
        else:
            st.info("Feature importance data not available")
        
        st.divider()
        
        # Recommendations
        st.markdown("### Clinical Recommendations")
        
        if risk_score > 75:
            st.warning("🔴 **CRITICAL RISK** - Immediate clinical intervention recommended")
            st.markdown("""
            - Increase monitoring frequency to weekly
            - Consider prophylaxis adjustment
            - Schedule urgent specialist consultation
            - Review treatment adherence
            """)
        elif risk_score > 50:
            st.info("🟠 **HIGH RISK** - Enhanced monitoring recommended")
            st.markdown("""
            - Increase monitoring to bi-weekly
            - Review current prophylaxis regimen
            - Assess treatment adherence
            - Consider specialist consultation
            """)
        elif risk_score > 25:
            st.success("🟡 **MODERATE RISK** - Standard monitoring")
            st.markdown("""
            - Continue standard monitoring schedule
            - Maintain current treatment plan
            - Regular adherence checks
            - Routine follow-ups
            """)
        else:
            st.success("🟢 **LOW RISK** - Maintenance care")
            st.markdown("""
            - Continue current treatment regimen
            - Routine monitoring
            - Periodic assessments
            - Standard clinical follow-up
            """)
    
    except Exception as e:
        st.error(f"Error generating predictions: {e}")


if __name__ == "__main__":
    main()
