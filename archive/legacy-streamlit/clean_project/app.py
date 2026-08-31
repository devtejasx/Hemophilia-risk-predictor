"""
Main Streamlit application for Hemophilia Clinical Decision Support.
Orchestrates all components, services, and utilities into a unified interface.
"""

import streamlit as st
from datetime import datetime

# Import all utilities and services
from utils import (
    init_session_state, is_logged_in, get_user_name,
    is_dark_mode, set_dark_mode, set_logged_in, set_logged_out,
    get_patients, add_patient, get_prediction, set_prediction,
    get_chat_history, add_chat_message
)

# Import components
from components import render_header, render_sidebar
from components.cards import render_kpi_card, render_risk_card
from components.charts import render_risk_distribution, render_trend_chart
from components.forms import render_patient_form, render_login_form, render_registration_form

# Import services
from services import predict_risk, get_response, explain_prediction
from database import get_database
from config import config
from constants import APP_NAME, APP_EMOJI


# Page configuration
st.set_page_config(
    page_title=f"{APP_NAME} - {APP_EMOJI}",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
init_session_state()


def render_login_page():
    """Render login/registration page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"# {APP_EMOJI} {APP_NAME}")
        st.markdown("---")
        
        # Tab selection
        auth_tab = st.radio(
            "Choose action",
            ["Login", "Register"],
            horizontal=True
        )
        
        if auth_tab == "Login":
            result = render_login_form()
            if result:
                username, password = result
                st.success(f"✓ Logged in as {username}")
                set_logged_in({"username": username, "name": username}, "demo_token")
                st.rerun()
        
        else:  # Register
            result = render_registration_form()
            if result:
                username, email, password = result
                st.success(f"✓ Account created! Welcome {username}")
                set_logged_in({"username": username, "name": username}, "demo_token")
                st.rerun()


def render_dashboard_page():
    """Render main dashboard page."""
    render_header()
    
    # KPI cards
    st.subheader("📊 Dashboard Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_card("Total Patients", len(get_patients()), icon="👥", help_text="Number of managed patients")
    
    with col2:
        render_kpi_card("Avg Risk", "42%", icon="⚠️", help_text="Average patient risk score")
    
    with col3:
        render_kpi_card("High Risk", "3", icon="🔴", help_text="Patients requiring attention")
    
    with col4:
        render_kpi_card("Compliance", "85%", icon="✓", help_text="Treatment adherence rate")
    
    # Main content
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("➕ Add Patient")
        patient_data = render_patient_form()
        
        if patient_data:
            add_patient(patient_data)
            st.success(f"✓ Patient {patient_data['name']} added successfully!")
            st.rerun()
    
    with col2:
        st.subheader("📋 Patient Summary")
        patients = get_patients()
        
        if patients:
            for patient in patients[:5]:
                st.info(f"👤 {patient['name']}", icon="ℹ️")
        else:
            st.info("No patients added yet")


def render_predictions_page():
    """Render predictions and risk assessment page."""
    render_header()
    
    st.subheader("🎯 Risk Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patient Selection")
        patients = get_patients()
        
        if patients:
            patient_names = [p.get('name', 'Unknown') for p in patients]
            selected_idx = st.selectbox("Select patient", range(len(patient_names)), format_func=lambda i: patient_names[i])
            selected_patient = patients[selected_idx]
            
            if st.button("🔮 Calculate Risk", use_container_width=True):
                # Prepare patient data
                patient_input = {
                    "age": selected_patient.get("age", 40),
                    "clotting_factor": selected_patient.get("clotting_factor", 50),
                    "activity_level": selected_patient.get("activity_level", 5),
                    "compliance": selected_patient.get("compliance", 0.8),
                    "bleeds": selected_patient.get("bleeds", 0),
                    "hospitalization": selected_patient.get("hospitalization", False),
                }
                
                # Get prediction
                prediction = predict_risk(patient_input)
                set_prediction(prediction)
                st.rerun()
        else:
            st.warning("No patients available. Please add a patient first.")
    
    with col2:
        st.markdown("### Risk Assessment")
        prediction = get_prediction()
        
        if prediction:
            risk = prediction["risk_score"]
            category = prediction["risk_category"]
            
            render_risk_card(risk, risk_label=f"{category} RISK")
            
            # Show confidence
            st.metric("Model Confidence", f"{prediction['confidence']:.1%}")
            
            # Show explanation
            with st.expander("📖 View Explanation"):
                explanation = explain_prediction(
                    prediction["features"],
                    risk,
                    category
                )
                
                st.write(f"**Interpretation:** {explanation['interpretation']}")
                
                st.write("**Contributing Factors:**")
                for factor, description in explanation['supporting_factors'].items():
                    st.write(f"- {factor}: {description}")
        else:
            st.info("Select a patient and click 'Calculate Risk' to see predictions")


def render_chat_page():
    """Render chatbot interface page."""
    render_header()
    
    st.subheader("💬 Clinical Chat Assistant")
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        history = get_chat_history()
        for msg in history[-10:]:  # Show last 10 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            with st.chat_message(role):
                st.write(content)
    
    # Input area
    user_input = st.chat_input("Ask a clinical question...")
    
    if user_input:
        # Add user message
        add_chat_message("user", user_input)
        
        # Get response
        response = get_response(user_input)
        
        # Add assistant message
        add_chat_message("assistant", response)
        
        st.rerun()


def render_analytics_page():
    """Render analytics dashboard."""
    render_header()
    
    st.subheader("📈 Analytics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Distribution")
        patients = get_patients()
        
        if patients:
            risk_scores = [
                predict_risk({
                    "age": p.get("age", 40),
                    "clotting_factor": p.get("clotting_factor", 50),
                    "activity_level": p.get("activity_level", 5),
                    "compliance": p.get("compliance", 0.8),
                    "bleeds": p.get("bleeds", 0),
                })["risk_score"]
                for p in patients
            ]
            patient_names = [p.get("name", "Unknown") for p in patients]
            
            render_risk_distribution(risk_scores, patient_names)
        else:
            st.info("No patient data available for analytics")
    
    with col2:
        st.markdown("### Statistics")
        patients = get_patients()
        
        if patients:
            st.metric("Total Patients", len(patients))
            st.metric("Average Age", f"{sum(p.get('age', 0) for p in patients) / len(patients):.0f} years")
            
            avg_compliance = sum(p.get('compliance', 0.8) for p in patients) / len(patients)
            st.metric("Avg Compliance", f"{avg_compliance:.1%}")


def render_settings_page():
    """Render settings page."""
    render_header()
    
    st.subheader("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Appearance")
        
        dark_mode = is_dark_mode()
        new_dark_mode = st.toggle("Dark Mode", value=dark_mode)
        
        if new_dark_mode != dark_mode:
            set_dark_mode(new_dark_mode)
            st.rerun()
    
    with col2:
        st.markdown("### Account")
        
        user_name = get_user_name()
        st.write(f"Logged in as: **{user_name}**")
        
        if st.button("🚪 Logout", use_container_width=True):
            set_logged_out()
            st.rerun()


def main():
    """Main application entry point."""
    
    # Check authentication
    if not is_logged_in():
        render_login_page()
        return
    
    # Authenticated user - show main app
    with st.sidebar:
        user_name = get_user_name()
        st.markdown(f"👤 {user_name}")
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Predictions", "Chat", "Analytics", "Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        if st.button("🚪 Logout", use_container_width=True):
            set_logged_out()
            st.rerun()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Predictions":
        render_predictions_page()
    elif page == "Chat":
        render_chat_page()
    elif page == "Analytics":
        render_analytics_page()
    elif page == "Settings":
        render_settings_page()


if __name__ == "__main__":
    main()
