"""
Add Patient Page - Patient Registration & Management
Register new patients with comprehensive clinical data
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# ============================================================================
# PATH SETUP & IMPORTS
# ============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="Add Patient", layout="wide")

from utils.session_state import init_session_state, get_session_var, set_session_var
from components.navbar import show_sidebar, show_page_header
from components.cards import info_card, empty_state
from database.db import get_database
from utils.helpers import (
    validate_email, calculate_age, format_date, truncate_text
)

# ============================================================================
# INITIALIZE
# ============================================================================
init_session_state()
show_sidebar()


# ============================================================================
# FORM VALIDATION
# ============================================================================
def validate_patient_form(data: dict) -> tuple[bool, str]:
    """Validate patient form data"""
    
    # Required fields
    required = ["name", "date_of_birth", "gender", "severity", "mutation_type"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"
    
    # Email validation
    if data.get("email") and not validate_email(data["email"]):
        return False, "Invalid email format"
    
    # Age check
    age = calculate_age(data["date_of_birth"])
    if age < 0 or age > 120:
        return False, f"Invalid age calculated: {age}"
    
    # Phone validation (simple)
    if data.get("phone") and len(data["phone"]) < 10:
        return False, "Invalid phone number"
    
    return True, ""


def prepare_patient_document(form_data: dict) -> dict:
    """Prepare patient data for database"""
    age = calculate_age(form_data["date_of_birth"])
    
    return {
        # Demographics
        "name": form_data.get("name", ""),
        "date_of_birth": form_data.get("date_of_birth"),
        "age": age,
        "gender": form_data.get("gender", ""),
        "ethnicity": form_data.get("ethnicity", ""),
        
        # Contact
        "email": form_data.get("email", ""),
        "phone": form_data.get("phone", ""),
        "address": form_data.get("address", ""),
        
        # Clinical
        "severity": form_data.get("severity", ""),
        "mutation_type": form_data.get("mutation_type", ""),
        "blood_type": form_data.get("blood_type", ""),
        "hla_typing": form_data.get("hla_typing", ""),
        
        # Treatment
        "current_treatment": form_data.get("current_treatment", ""),
        "treatment_frequency": form_data.get("treatment_frequency", ""),
        "treatment_adherence": form_data.get("treatment_adherence", 80),
        "dose_per_infusion": form_data.get("dose_per_infusion", 0),
        
        # History
        "family_history_inhibitor": form_data.get("family_history_inhibitor", "No"),
        "previous_inhibitor": form_data.get("previous_inhibitor", "No"),
        "joint_damage_score": form_data.get("joint_damage_score", 0),
        "bleeds_per_month": form_data.get("bleeds_per_month", 0),
        
        # Other
        "comorbidities": form_data.get("comorbidities", []),
        "medications": form_data.get("medications", ""),
        "allergies": form_data.get("allergies", ""),
        "notes": form_data.get("notes", ""),
        
        # Metadata
        "date_added": datetime.now(),
        "date_modified": datetime.now(),
        "status": "active"
    }


# ============================================================================
# MAIN PAGE
# ============================================================================
def main():
    show_page_header(
        "👤 Add Patient",
        "Register a new patient in the system"
    )
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["➕ New Patient", "📋 Patient List", "📊 Statistics"])
    
    # ========================================================================
    # TAB 1: NEW PATIENT FORM
    # ========================================================================
    with tab1:
        st.markdown("### Patient Registration Form")
        
        # Use form to group inputs
        with st.form("patient_form", clear_on_submit=True):
            
            # ================================================================
            # SECTION 1: DEMOGRAPHICS
            # ================================================================
            st.markdown("#### 👤 Demographics")
            
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input(
                    "Full Name *",
                    placeholder="John Doe",
                    help="Patient full name"
                )
            with col2:
                email = st.text_input(
                    "Email Address",
                    placeholder="john@example.com",
                    help="Patient email (optional)"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                dob = st.date_input(
                    "Date of Birth *",
                    help="Patient date of birth"
                )
            with col2:
                gender = st.selectbox(
                    "Gender *",
                    ["Male", "Female", "Other"],
                    help="Patient gender"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                ethnicity = st.selectbox(
                    "Ethnicity",
                    ["", "Caucasian", "African", "Asian", "Hispanic", "Middle Eastern", "Other"],
                    help="Patient ethnicity (optional)"
                )
            with col2:
                phone = st.text_input(
                    "Phone Number",
                    placeholder="+1-XXX-XXX-XXXX",
                    help="Patient phone (optional)"
                )
            
            address = st.text_area(
                "Address",
                placeholder="Street address, city, state, zip",
                max_chars=200,
                height=80
            )
            
            st.divider()
            
            # ================================================================
            # SECTION 2: CLINICAL DATA
            # ================================================================
            st.markdown("#### 🩸 Clinical Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                severity = st.selectbox(
                    "Hemophilia Severity *",
                    ["", "Mild", "Moderate", "Severe"],
                    help="Factor VIII/IX activity level"
                )
            with col2:
                mutation_type = st.selectbox(
                    "Mutation Type *",
                    ["", "Intron22", "Intron1", "Missense", "Nonsense", "Deletion", "Inversion", "Unknown"],
                    help="Genetic mutation classification"
                )
            with col3:
                blood_type = st.selectbox(
                    "Blood Type",
                    ["", "O", "A", "B", "AB"],
                    help="ABO blood type (optional)"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                hla_typing = st.text_input(
                    "HLA Typing",
                    placeholder="e.g., A2, B44",
                    help="HLA type (optional)"
                )
            with col2:
                baseline_factor = st.number_input(
                    "Baseline Factor Level (%)",
                    min_value=0, max_value=100, value=50,
                    help="Baseline factor activity level"
                )
            
            st.divider()
            
            # ================================================================
            # SECTION 3: TREATMENT
            # ================================================================
            st.markdown("#### 💊 Treatment Information")
            
            current_treatment = st.selectbox(
                "Current Treatment",
                ["", "Prophylaxis", "On-demand", "Gene therapy", "None"],
                help="Current treatment regimen"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                treatment_freq = st.selectbox(
                    "Treatment Frequency",
                    ["", "Daily", "2-3x/week", "Weekly", "Monthly", "As needed"],
                    help="How often treated"
                )
            with col2:
                dose = st.number_input(
                    "Dose per Infusion (IU/kg)",
                    min_value=0, max_value=200, value=50,
                    help="Standard dose"
                )
            with col3:
                adherence = st.slider(
                    "Treatment Adherence (%)",
                    min_value=0, max_value=100, value=85,
                    help="Estimated adherence percentage"
                )
            
            st.divider()
            
            # ================================================================
            # SECTION 4: MEDICAL HISTORY
            # ================================================================
            st.markdown("#### 📖 Medical History")
            
            col1, col2 = st.columns(2)
            with col1:
                family_history = st.selectbox(
                    "Family History of Inhibitor",
                    ["", "No", "Yes", "Unknown"],
                    help="Does family have inhibitor history?"
                )
            with col2:
                previous_inhib = st.selectbox(
                    "Previous Inhibitor Development",
                    ["", "No", "Yes", "Unknown"],
                    help="Has patient developed inhibitors before?"
                )
            
            col1, col2 = st.columns(2)
            with col1:
                joint_damage = st.number_input(
                    "Joint Damage Score (Pettersson 0-78)",
                    min_value=0, max_value=78, value=0,
                    help="Radiographic joint damage"
                )
            with col2:
                bleeds_month = st.number_input(
                    "Average Bleeds per Month",
                    min_value=0, max_value=100, value=2,
                    help="Spontaneous + post-trauma bleeds"
                )
            
            comorbidities_input = st.multiselect(
                "Comorbidities",
                ["None", "HBV", "HCV", "HIV", "Liver disease", "Arthritis", "Other"],
                help="Select any comorbidities"
            )
            
            medications = st.text_area(
                "Current Medications",
                placeholder="List all medications (comma-separated)",
                max_chars=200,
                height=60
            )
            
            allergies = st.text_input(
                "Allergies",
                placeholder="Any known allergies?",
                help="Medication or substance allergies"
            )
            
            st.divider()
            
            # ================================================================
            # SECTION 5: NOTES
            # ================================================================
            st.markdown("#### 📝 Additional Notes")
            
            notes = st.text_area(
                "Clinical Notes",
                placeholder="Any additional information...",
                max_chars=500,
                height=100
            )
            
            st.divider()
            
            # ================================================================
            # SUBMIT BUTTON
            # ================================================================
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                submit_btn = st.form_submit_button(
                    "✅ Save Patient",
                    use_container_width=True
                )
            
            with col2:
                clear_btn = st.form_submit_button(
                    "🗑️ Clear",
                    use_container_width=True
                )
            
            # ================================================================
            # FORM SUBMISSION
            # ================================================================
            if submit_btn:
                if not name or not dob or not gender or not severity or not mutation_type:
                    st.error("❌ Please fill all required fields (*)")
                else:
                    # Prepare form data
                    form_data = {
                        "name": name,
                        "email": email,
                        "date_of_birth": dob,
                        "gender": gender,
                        "ethnicity": ethnicity or "",
                        "phone": phone,
                        "address": address,
                        "severity": severity,
                        "mutation_type": mutation_type,
                        "blood_type": blood_type or "",
                        "hla_typing": hla_typing,
                        "baseline_factor_level": baseline_factor,
                        "current_treatment": current_treatment or "",
                        "treatment_frequency": treatment_freq or "",
                        "dose_per_infusion": dose,
                        "treatment_adherence": adherence,
                        "family_history_inhibitor": family_history or "Unknown",
                        "previous_inhibitor": previous_inhib or "Unknown",
                        "joint_damage_score": joint_damage,
                        "bleeds_per_month": bleeds_month,
                        "comorbidities": comorbidities_input,
                        "medications": medications,
                        "allergies": allergies,
                        "notes": notes
                    }
                    
                    # Validate
                    is_valid, error_msg = validate_patient_form(form_data)
                    
                    if not is_valid:
                        st.error(f"❌ {error_msg}")
                    else:
                        try:
                            # Prepare document
                            patient_doc = prepare_patient_document(form_data)
                            
                            # Save to database
                            db = get_database()
                            patient_id = db.save_patient(patient_doc)
                            
                            # Update session
                            set_session_var("current_patient_id", str(patient_id))
                            set_session_var("current_patient", patient_doc)
                            
                            st.success(
                                f"✅ Patient '{name}' added successfully! (ID: {patient_id})"
                            )
                            st.balloons()
                            
                        except Exception as e:
                            st.error(f"❌ Error saving patient: {str(e)}")
    
    # ========================================================================
    # TAB 2: PATIENT LIST
    # ========================================================================
    with tab2:
        st.markdown("### 📋 Patient List")
        
        try:
            db = get_database()
            patients = db.get_patients(limit=100)
            
            if not patients:
                empty_state(
                    icon="📭",
                    title="No Patients",
                    message="Add a patient to get started"
                )
            else:
                # Convert to DataFrame
                df = pd.DataFrame(patients)
                
                # Display columns
                display_cols = ["name", "age", "gender", "severity", "mutation_type"]
                if "date_added" in df.columns:
                    df["date_added"] = pd.to_datetime(df["date_added"]).dt.strftime("%Y-%m-%d")
                
                # Filter columns that exist
                display_cols = [c for c in display_cols if c in df.columns]
                
                st.dataframe(
                    df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.caption(f"Total: {len(patients)} patients")
        
        except Exception as e:
            st.error(f"Error loading patients: {e}")
    
    # ========================================================================
    # TAB 3: STATISTICS
    # ========================================================================
    with tab3:
        st.markdown("### 📊 Patient Statistics")
        
        try:
            db = get_database()
            patients = db.get_patients(limit=1000)
            
            if patients:
                df = pd.DataFrame(patients)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Patients", len(df))
                
                with col2:
                    avg_age = df.get("age", pd.Series()).mean() if "age" in df.columns else 0
                    st.metric("Average Age", f"{avg_age:.1f} years")
                
                with col3:
                    severe = len(df[df.get("severity", "") == "Severe"]) if "severity" in df.columns else 0
                    st.metric("Severe Cases", severe)
                
                with col4:
                    st.metric("Recent (7d)", "Coming soon")
                
                st.divider()
                
                # Distribution charts
                col1, col2 = st.columns(2)
                
                with col1:
                    if "severity" in df.columns:
                        st.markdown("**Severity Distribution**")
                        severity_counts = df["severity"].value_counts()
                        st.bar_chart(severity_counts)
                
                with col2:
                    if "gender" in df.columns:
                        st.markdown("**Gender Distribution**")
                        gender_counts = df["gender"].value_counts()
                        st.bar_chart(gender_counts)
            else:
                st.info("No patients to analyze yet")
        
        except Exception as e:
            st.error(f"Error loading statistics: {e}")


if __name__ == "__main__":
    main()
