"""
Patient Form Page - Clinical data input and patient selection
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_utils.state_manager import StateManager
from streamlit_utils.ui_components import (
    create_header, form_section, required_input, show_success,
    show_error, show_warning, create_tabs
)
from streamlit_utils.backend_client import get_backend_client


def render():
    """Render patient form page"""
    
    create_header("🔬 Patient Form", "Enter or select patient data for analysis")
    
    # Get clients
    state = StateManager()
    backend = get_backend_client()
    
    # Create tabs for different workflows
    tab1, tab2, tab3 = create_tabs(["New Patient", "Select Existing", "Quick Assessment"])
    
    # ========================================================================
    # TAB 1: NEW PATIENT
    # ========================================================================
    
    with tab1:
        st.markdown("### ➕ Add New Patient")
        st.markdown("Enter detailed patient information for initial assessment")
        st.divider()
        
        # Personal Information
        form_section("Personal Information", "👤")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            patient_id = st.text_input(
                "Patient ID *",
                placeholder="e.g., P001",
                key="new_patient_id"
            )
        with col2:
            first_name = st.text_input(
                "First Name *",
                placeholder="John",
                key="new_first_name"
            )
        with col3:
            last_name = st.text_input(
                "Last Name *",
                placeholder="Doe",
                key="new_last_name"
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input(
                "Age *",
                min_value=0,
                max_value=150,
                value=50,
                key="new_age"
            )
        with col2:
            gender = st.selectbox(
                "Gender *",
                ["Male", "Female", "Other"],
                key="new_gender"
            )
        with col3:
            contact = st.text_input(
                "Contact Number",
                placeholder="+1(555)123-4567",
                key="new_contact"
            )
        
        # Medical History
        form_section("Medical History", "⚕️")
        
        col1, col2 = st.columns(2)
        with col1:
            diagnosis = st.text_input(
                "Primary Diagnosis *",
                placeholder="e.g., Hypertension, Diabetes",
                key="new_diagnosis"
            )
        with col2:
            severity = st.selectbox(
                "Severity Level *",
                ["Mild", "Moderate", "Severe", "Critical"],
                key="new_severity"
            )
        
        st.markdown("**Comorbidities** (Select all that apply)")
        col1, col2, col3, col4 = st.columns(4)
        comorbidities = []
        with col1:
            if st.checkbox("Hypertension", key="new_comorbid_htn"):
                comorbidities.append("Hypertension")
        with col2:
            if st.checkbox("Diabetes", key="new_comorbid_dm"):
                comorbidities.append("Diabetes")
        with col3:
            if st.checkbox("Obesity", key="new_comorbid_obesity"):
                comorbidities.append("Obesity")
        with col4:
            if st.checkbox("Smoking", key="new_comorbid_smoking"):
                comorbidities.append("Smoking")
        
        # Clinical Measurements
        form_section("Clinical Measurements", "📏")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            systolic = st.number_input(
                "Systolic BP (mmHg)",
                min_value=50,
                max_value=250,
                value=120,
                key="new_systolic"
            )
        with col2:
            diastolic = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=30,
                max_value=150,
                value=80,
                key="new_diastolic"
            )
        with col3:
            heart_rate = st.number_input(
                "Heart Rate (bpm)",
                min_value=30,
                max_value=200,
                value=75,
                key="new_heart_rate"
            )
        with col4:
            oxygen_sat = st.number_input(
                "O2 Saturation (%)",
                min_value=50,
                max_value=100,
                value=98,
                key="new_oxygen_sat"
            )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=35.0,
                max_value=42.0,
                value=37.0,
                step=0.1,
                key="new_temperature"
            )
        with col2:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.0,
                step=0.1,
                key="new_bmi"
            )
        with col3:
            glucose = st.number_input(
                "Glucose (mg/dL)",
                min_value=40,
                max_value=500,
                value=100,
                key="new_glucose"
            )
        with col4:
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=50,
                max_value=400,
                value=200,
                key="new_cholesterol"
            )
        
        # Laboratory Values
        form_section("Laboratory Values", "🧪")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            creatinine = st.number_input(
                "Creatinine (mg/dL)",
                min_value=0.5,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="new_creatinine"
            )
        with col2:
            hemoglobin = st.number_input(
                "Hemoglobin (g/dL)",
                min_value=5.0,
                max_value=20.0,
                value=14.0,
                step=0.1,
                key="new_hemoglobin"
            )
        with col3:
            platelet_count = st.number_input(
                "Platelet Count (K/μL)",
                min_value=10,
                max_value=1000,
                value=250,
                key="new_platelets"
            )
        with col4:
            wbc = st.number_input(
                "WBC Count (K/μL)",
                min_value=1,
                max_value=50,
                value=7,
                key="new_wbc"
            )
        
        # Notes
        form_section("Additional Notes", "📝")
        notes = st.text_area(
            "Clinical Notes",
            placeholder="Enter any additional clinical observations...",
            height=100,
            key="new_notes"
        )
        
        # Submit button
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Validate & Preview", use_container_width=True):
                if not all([patient_id, first_name, last_name, diagnosis, severity]):
                    show_error("Please fill in all required fields marked with *")
                else:
                    st.success("✅ All required fields completed!")
        
        with col2:
            if st.button("💾 Save Patient", use_container_width=True):
                if not all([patient_id, first_name, last_name, diagnosis, severity]):
                    show_error("Please fill in all required fields marked with *")
                else:
                    # Prepare patient data
                    patient_data = {
                        "patient_id": patient_id,
                        "first_name": first_name,
                        "last_name": last_name,
                        "age": age,
                        "gender": gender,
                        "contact": contact,
                        "diagnosis": diagnosis,
                        "severity": severity,
                        "comorbidities": comorbidities,
                        "vitals": {
                            "systolic_bp": systolic,
                            "diastolic_bp": diastolic,
                            "heart_rate": heart_rate,
                            "oxygen_saturation": oxygen_sat,
                            "temperature": temperature,
                        },
                        "measurements": {
                            "bmi": bmi,
                            "glucose": glucose,
                            "cholesterol": cholesterol,
                        },
                        "labs": {
                            "creatinine": creatinine,
                            "hemoglobin": hemoglobin,
                            "platelet_count": platelet_count,
                            "wbc_count": wbc,
                        },
                        "notes": notes,
                        "created_at": datetime.now().isoformat(),
                    }
                    
                    # Save to backend
                    with st.spinner("Saving patient..."):
                        result = backend.create_patient(patient_data)
                    
                    if result:
                        state.set_current_patient(patient_id, patient_data)
                        show_success(f"Patient {patient_id} saved successfully!")
                        st.balloons()
                    else:
                        show_error("Failed to save patient. Please check backend connection.")
    
    # ========================================================================
    # TAB 2: SELECT EXISTING PATIENT
    # ========================================================================
    
    with tab2:
        st.markdown("### 🔍 Select Existing Patient")
        st.markdown("Choose a patient from the database")
        st.divider()
        
        # Fetch patients from backend
        with st.spinner("Loading patients..."):
            patients = backend.list_patients(limit=100)
        
        if patients:
            # Create selectbox
            patient_options = {f"{p.get('patient_id', '')} - {p.get('first_name', '')} {p.get('last_name', '')}" : p for p in patients}
            selected_option = st.selectbox(
                "Select Patient",
                options=list(patient_options.keys()),
                key="select_patient"
            )
            
            if selected_option:
                selected_patient = patient_options[selected_option]
                
                # Display patient info
                st.markdown("#### Patient Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Age", f"{selected_patient.get('age', 'N/A')} yrs")
                with col2:
                    st.metric("Gender", selected_patient.get('gender', 'N/A'))
                with col3:
                    st.metric("Diagnosis", selected_patient.get('diagnosis', 'N/A'))
                with col4:
                    st.metric("Severity", selected_patient.get('severity', 'N/A'))
                
                # Load button
                if st.button("📂 Load Patient", use_container_width=True):
                    state.set_current_patient(selected_patient.get('patient_id'), selected_patient)
                    show_success(f"Patient {selected_patient.get('patient_id')} loaded!")
        else:
            st.warning("No patients found in database. Create a new patient first.")
    
    # ========================================================================
    # TAB 3: QUICK ASSESSMENT
    # ========================================================================
    
    with tab3:
        st.markdown("### ⚡ Quick Assessment")
        st.markdown("Minimal fields for rapid risk assessment")
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            quick_age = st.number_input(
                "Age",
                min_value=0,
                max_value=150,
                value=50,
                key="quick_age"
            )
            quick_systolic = st.number_input(
                "Systolic BP",
                min_value=50,
                max_value=250,
                value=120,
                key="quick_systolic"
            )
            quick_heart_rate = st.number_input(
                "Heart Rate",
                min_value=30,
                max_value=200,
                value=75,
                key="quick_hr"
            )
            quick_glucose = st.number_input(
                "Glucose",
                min_value=40,
                max_value=500,
                value=100,
                key="quick_glucose"
            )
        
        with col2:
            quick_gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"],
                key="quick_gender"
            )
            quick_diastolic = st.number_input(
                "Diastolic BP",
                min_value=30,
                max_value=150,
                value=80,
                key="quick_diastolic"
            )
            quick_bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.0,
                step=0.1,
                key="quick_bmi"
            )
            quick_cholesterol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=400,
                value=200,
                key="quick_cholesterol"
            )
        
        if st.button("⚡ Run Quick Assessment", use_container_width=True):
            quick_data = {
                "patient_id": f"QUICK_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "age": quick_age,
                "gender": quick_gender,
                "vitals": {
                    "systolic_bp": quick_systolic,
                    "diastolic_bp": quick_diastolic,
                    "heart_rate": quick_heart_rate,
                },
                "measurements": {
                    "bmi": quick_bmi,
                    "glucose": quick_glucose,
                    "cholesterol": quick_cholesterol,
                },
            }
            
            with st.spinner("Running assessment..."):
                prediction = backend.predict(quick_data)
            
            if prediction:
                state.set_current_patient(quick_data["patient_id"], quick_data)
                state.set_prediction_results(prediction)
                show_success("Assessment complete! View results on the Results dashboard.")
                st.write(prediction)
            else:
                show_error("Assessment failed. Check backend connection.")
