"""
Form components for user input and data collection.
Reusable form widgets and input handling.
"""

import streamlit as st
from typing import Dict, Any, Optional, Tuple
from utils import Validator
from constants import (
    MIN_AGE, MAX_AGE,
    MIN_CLOTTING_FACTOR, MAX_CLOTTING_FACTOR,
    MIN_ACTIVITY_LEVEL, MAX_ACTIVITY_LEVEL,
    MIN_COMPLIANCE, MAX_COMPLIANCE,
)


def render_patient_form() -> Optional[Dict[str, Any]]:
    """Render complete patient form with validation.
    
    Returns:
        Dictionary with form data on submit, None if not submitted
    """
    
    st.subheader("📋 Patient Information")
    
    with st.form(key="patient_form"):
        # Create two columns for layout
        col1, col2 = st.columns(2)
        
        # Column 1: Demographics
        with col1:
            st.markdown("**Demographics**")
            name = st.text_input(
                "Patient Name",
                placeholder="Enter full name",
                help="Patient full name"
            )
            age = st.number_input(
                "Age (years)",
                min_value=MIN_AGE,
                max_value=MAX_AGE,
                step=1,
                help="Patient age in years"
            )
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"],
                help="Patient gender"
            )
        
        # Column 2: Clinical Parameters
        with col2:
            st.markdown("**Clinical Parameters**")
            clotting_factor = st.slider(
                "Clotting Factor Level (%)",
                min_value=MIN_CLOTTING_FACTOR,
                max_value=MAX_CLOTTING_FACTOR,
                step=1,
                value=50,
                help="Current clotting factor percentage"
            )
            activity_level = st.slider(
                "Activity Level (1-10)",
                min_value=MIN_ACTIVITY_LEVEL,
                max_value=MAX_ACTIVITY_LEVEL,
                step=1,
                value=5,
                help="Patient's typical activity level"
            )
            compliance = st.slider(
                "Treatment Compliance (%)",
                min_value=0,
                max_value=100,
                step=1,
                value=80,
                help="Adherence to treatment protocol"
            )
        
        # Additional info
        st.markdown("**Recent History**")
        col1, col2 = st.columns(2)
        
        with col1:
            bleeds = st.number_input(
                "Recent Bleeds (past 3 months)",
                min_value=0,
                step=1,
                value=0,
                help="Number of bleeding episodes"
            )
        
        with col2:
            hospitalization = st.checkbox(
                "Recent Hospitalization",
                help="Any hospitalization in past month?"
            )
        
        # Notes
        notes = st.text_area(
            "Clinical Notes",
            placeholder="Additional clinical information...",
            height=100,
            help="Any relevant clinical notes or observations"
        )
        
        # Submit button
        submitted = st.form_submit_button("✓ Submit Patient Data", use_container_width=True)
    
    if submitted:
        # Validate form
        form_data = {
            "name": name,
            "age": age,
            "gender": gender,
            "clotting_factor": clotting_factor,
            "activity_level": activity_level,
            "compliance": compliance / 100.0,  # Convert to 0-1 range
            "bleeds": bleeds,
            "hospitalization": hospitalization,
            "notes": notes,
        }
        
        valid, errors = Validator.validate_patient_form(form_data)
        
        if valid:
            return form_data
        else:
            # Show validation errors
            for field, error_msg in errors.items():
                st.error(f"❌ {field}: {error_msg}")
            return None
    
    return None


def render_demographics_inputs() -> Dict[str, Any]:
    """Render only demographic input fields.
    
    Returns:
        Dictionary with demographic data
    """
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name", placeholder="Patient name")
        age = st.number_input(
            "Age",
            min_value=MIN_AGE,
            max_value=MAX_AGE,
            step=1
        )
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        dob = st.date_input("Date of Birth (optional)")
    
    return {
        "name": name,
        "age": age,
        "gender": gender,
        "dob": dob,
    }


def render_clinical_inputs() -> Dict[str, float]:
    """Render only clinical parameter input fields.
    
    Returns:
        Dictionary with clinical parameters
    """
    
    st.markdown("**Clinical Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        clotting_factor = st.slider(
            "Clotting Factor Level (%)",
            min_value=MIN_CLOTTING_FACTOR,
            max_value=MAX_CLOTTING_FACTOR,
            step=1,
            value=50,
        )
        
        bleeds = st.number_input(
            "Recent Bleeds",
            min_value=0,
            step=1,
            value=0,
        )
    
    with col2:
        activity_level = st.slider(
            "Activity Level (1-10)",
            min_value=MIN_ACTIVITY_LEVEL,
            max_value=MAX_ACTIVITY_LEVEL,
            step=1,
            value=5,
        )
        
        compliance = st.slider(
            "Treatment Compliance (%)",
            min_value=0,
            max_value=100,
            step=1,
            value=80,
        )
    
    return {
        "clotting_factor": clotting_factor,
        "activity_level": activity_level,
        "compliance": compliance / 100.0,
        "bleeds": bleeds,
    }


def render_login_form() -> Optional[Tuple[str, str]]:
    """Render login form.
    
    Returns:
        Tuple of (username, password) on submit, None otherwise
    """
    
    st.markdown("### 🔐 Login")
    
    with st.form(key="login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            help="Your account username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Your account password"
        )
        
        submitted = st.form_submit_button("Login", use_container_width=True)
    
    if submitted:
        valid, message = Validator.validate_login(username, password)
        if valid:
            return username, password
        else:
            st.error(message)
            return None
    
    return None


def render_registration_form() -> Optional[Tuple[str, str, str]]:
    """Render user registration form.
    
    Returns:
        Tuple of (username, email, password) on submit, None otherwise
    """
    
    st.markdown("### 📝 Create Account")
    
    with st.form(key="registration_form"):
        username = st.text_input(
            "Username",
            placeholder="Choose a username",
            help="3-50 characters, alphanumeric with - and _"
        )
        email = st.text_input(
            "Email",
            placeholder="your.email@example.com",
            help="Valid email address"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a strong password",
            help="Minimum 6 characters"
        )
        password_confirm = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Confirm your password"
        )
        
        submitted = st.form_submit_button("Register", use_container_width=True)
    
    if submitted:
        if password != password_confirm:
            st.error("Passwords do not match")
            return None
        
        valid, errors = Validator.validate_registration(username, email, password)
        
        if valid:
            return username, email, password
        else:
            for field, error_msg in errors.items():
                st.error(f"❌ {field}: {error_msg}")
            return None
    
    return None


def render_filter_form() -> Dict[str, Any]:
    """Render patient filter/search form.
    
    Returns:
        Dictionary with filter parameters
    """
    
    with st.form(key="filter_form"):
        st.markdown("**Filter Patients**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_text = st.text_input(
                "Search by name",
                placeholder="Patient name..."
            )
        
        with col2:
            risk_level = st.selectbox(
                "Risk Level",
                ["All", "Low", "Medium", "High"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Name", "Risk", "Age", "Recent"]
            )
        
        submitted = st.form_submit_button("🔍 Apply Filters", use_container_width=True)
    
    return {
        "search": search_text,
        "risk_level": risk_level,
        "sort_by": sort_by,
        "submitted": submitted,
    }


def render_simple_input_form(fields: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Render a dynamic form from field definitions.
    
    Args:
        fields: Dictionary of field_name -> field_config
                field_config should include 'type', 'label', 'required', etc.
    
    Returns:
        Dictionary with form data on submit, None otherwise
    """
    
    with st.form(key="dynamic_form"):
        form_data = {}
        
        for field_name, field_config in fields.items():
            field_type = field_config.get("type", "text")
            label = field_config.get("label", field_name)
            required = field_config.get("required", False)
            help_text = field_config.get("help", "")
            
            if field_type == "text":
                form_data[field_name] = st.text_input(
                    label,
                    help=help_text,
                    value=field_config.get("value", "")
                )
            elif field_type == "number":
                form_data[field_name] = st.number_input(
                    label,
                    help=help_text,
                    min_value=field_config.get("min", 0),
                    max_value=field_config.get("max", 100),
                    value=field_config.get("value", 0)
                )
            elif field_type == "select":
                form_data[field_name] = st.selectbox(
                    label,
                    field_config.get("options", []),
                    help=help_text
                )
            elif field_type == "checkbox":
                form_data[field_name] = st.checkbox(
                    label,
                    help=help_text,
                    value=field_config.get("value", False)
                )
        
        submitted = st.form_submit_button("Submit", use_container_width=True)
    
    return form_data if submitted else None
