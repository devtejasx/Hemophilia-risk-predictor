"""
Add Patient Page - Patient data entry form
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.navbar import show_page_header
from components.cards import info_card
from database.db import get_database
from utils.helpers import format_number, get_risk_level, calculate_patient_risk_score
from utils.session_state import update_patient_data

st.set_page_config(page_title="Add Patient", layout="wide")


def main():
    show_page_header("👤 Add New Patient", "Enter patient information for assessment")
    
    # Create form columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        name = st.text_input("Patient Name *", placeholder="John Doe")
        age = st.number_input("Age (years) *", min_value=0, max_value=150, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not Disclosed"])
        ethnicity = st.text_input("Ethnicity", placeholder="e.g., Caucasian")
    
    with col2:
        st.markdown("#### Clinical Information")
        severity = st.selectbox("Severity Level *", ["Mild", "Moderate", "Severe", "Critical"])
        mutation = st.text_input("Genetic Mutation", placeholder="e.g., F8, F9, FV")
        blood_type = st.selectbox("Blood Type", ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"])
    
    st.divider()
    
    st.markdown("#### Treatment Parameters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dose = st.number_input("Prophylaxis Dose (IU)", min_value=0, value=2500, step=500)
    
    with col2:
        exposure = st.number_input("Exposure Events (past 6 months)", min_value=0, max_value=100, value=0)
    
    with col3:
        adherence = st.slider("Treatment Adherence (%)", 0, 100, 80, step=5)
    
    st.divider()
    
    # Additional info
    st.markdown("#### Additional Notes")
    notes = st.text_area("Medical Notes", placeholder="Any additional clinical information...")
    
    st.divider()
    
    # Submit button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Add Patient", use_container_width=True):
            if not name:
                st.error("❌ Patient name is required")
                return
            
            # Calculate risk score
            risk_result = calculate_patient_risk_score(
                age=age,
                severity=severity,
                mutation=mutation,
                blood_type=blood_type,
                dose=dose,
                exposure=exposure,
                treatment_adherence=adherence
            )
            
            patient_data = {
                "id": str(uuid.uuid4())[:8],
                "name": name,
                "age": age,
                "gender": gender,
                "ethnicity": ethnicity,
                "severity": severity,
                "mutation": mutation,
                "blood_type": blood_type,
                "dose": dose,
                "exposure": exposure,
                "treatment_adherence": adherence,
                "risk_score": risk_result['risk_score'],
                "notes": notes,
                "created_at": datetime.now().isoformat()
            }
            
            # Save to database
            try:
                db = get_database()
                db.insert_patient(patient_data)
                
                # Update session state
                update_patient_data(patient_data)
                
                st.success("✅ Patient added successfully!")
                st.info(f"**Risk Score: {risk_result['risk_score']:.1f}%** - {risk_result['classification']}")
                
                # Display prediction details
                st.markdown("### Risk Assessment Details")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class='info-card' style='border-radius: 10px; padding: 15px; border-left: 4px solid #00d4ff;'>
                        <h4 style='margin: 0; color: #00d4ff;'>Risk Classification</h4>
                        <p style='margin: 10px 0; font-size: 18px; font-weight: bold; color: #fff;'>
                            {risk_result['classification']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='info-card' style='border-radius: 10px; padding: 15px; border-left: 4px solid #ffa500;'>
                        <h4 style='margin: 0; color: #ffa500;'>Key Factors</h4>
                        <p style='margin: 10px 0; font-size: 12px; color: #ccc;'>
                            Severity: {severity} | Adherence: {adherence}% | Exposure: {exposure} events
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"❌ Error saving patient: {e}")
    
    # Info section
    st.divider()
    info_card(
        "Severity Guidelines",
        "Mild: <5% clotting factor | Moderate: 5-40% | Severe: >40-1% | Critical: <1%",
        icon="📋"
    )


if __name__ == "__main__":
    main()
