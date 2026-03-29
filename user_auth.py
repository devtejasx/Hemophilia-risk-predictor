"""
User Authentication and Management Module
Handles user authentication, registration, and role-based access control
"""

import streamlit as st
from database import (
    register_user, authenticate_user, get_user, get_all_users, 
    get_doctors, hash_password, assign_patient_to_doctor,
    get_assigned_doctor, get_doctor_patients, get_all_patients
)
from typing import Optional, Dict, List

class UserManager:
    """Manage user authentication and profiles"""
    
    @staticmethod
    def login_page() -> Optional[Dict]:
        """Display login page and return authenticated user if successful"""
        st.set_page_config(page_title="🏥 Hemophilia AI Platform - Login", layout="wide")
        
        # Custom CSS for login page
        st.markdown("""
        <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 40px;
                border-radius: 15px;
                background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 255, 0.05) 100%);
                border: 1px solid rgba(0, 212, 255, 0.3);
            }
            .login-title {
                text-align: center;
                margin-bottom: 30px;
                font-size: 28px;
                font-weight: bold;
                background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        """, unsafe_allow_html=True)
        
        col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
        
        with col_center2:
            st.markdown("## 🏥 Hemophilia AI Platform")
            st.markdown("### Advanced Clinical Intelligence System")
            
            with st.container():
                st.divider()
                
                login_type = st.radio(
                    "Select Action",
                    ["Login", "Register New Account"],
                    horizontal=True
                )
                
                if login_type == "Login":
                    return UserManager._handle_login()
                else:
                    return UserManager._handle_registration()
    
    @staticmethod
    def _handle_login() -> Optional[Dict]:
        """Handle user login"""
        st.markdown("### 🔐 Login to Your Account")
        
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username"
        )
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔓 Login", use_container_width=True, key="login_btn"):
                if not username or not password:
                    st.error("❌ Please enter both username and password")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.success(f"✅ Welcome, {user['full_name']}!")
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        with col2:
            st.markdown("")
        
        # Demo credentials info
        with st.expander("ℹ️ Demo Credentials"):
            st.info("""
            For testing:
            - Doctor Login: doctor / password123
            - Nurse Login: nurse / password123
            
            Note: First time users should register an account.
            """)
        
        return None
    
    @staticmethod
    def _handle_registration() -> Optional[Dict]:
        """Handle new user registration"""
        st.markdown("### 📝 Create New Account")
        
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input(
                "Full Name",
                placeholder="Enter your full name",
                key="reg_fullname"
            )
            email = st.text_input(
                "Email",
                placeholder="Enter your email",
                key="reg_email"
            )
        
        with col2:
            username = st.text_input(
                "Username",
                placeholder="Choose a username",
                key="reg_username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Create a strong password",
                key="reg_password"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            role = st.selectbox(
                "Role",
                ["nurse", "doctor", "administrator"],
                key="reg_role"
            )
        
        with col4:
            department = st.text_input(
                "Department",
                placeholder="e.g., Hematology",
                key="reg_dept"
            )
        
        if st.button("📋 Create Account", use_container_width=True, key="register_btn"):
            # Validate registration
            if not all([full_name, email, username, password]):
                st.error("❌ Please fill in all required fields")
            elif len(password) < 8:
                st.error("❌ Password must be at least 8 characters")
            elif "@" not in email:
                st.error("❌ Please enter a valid email")
            else:
                success = register_user(username, password, email, full_name, role, department)
                if success:
                    st.success("✅ Account created successfully! Please login.")
                    # Auto-login after registration
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.authenticated = True
                        st.rerun()
                else:
                    st.error("❌ Username or email already exists")
        
        return None
    
    @staticmethod
    def initialize_demo_users():
        """Initialize demo users for testing"""
        demo_users = [
            {
                "username": "doctor",
                "password": "password123",
                "email": "doctor@hemophilia.clinic",
                "full_name": "Dr. Sarah Clinical",
                "role": "doctor",
                "department": "Hematology"
            },
            {
                "username": "nurse",
                "password": "password123",
                "email": "nurse@hemophilia.clinic",
                "full_name": "Nurse John Care",
                "role": "nurse",
                "department": "Patient Care"
            },
            {
                "username": "admin",
                "password": "password123",
                "email": "admin@hemophilia.clinic",
                "full_name": "Admin System",
                "role": "administrator",
                "department": "Administration"
            }
        ]
        
        for user in demo_users:
            try:
                register_user(
                    user["username"],
                    user["password"],
                    user["email"],
                    user["full_name"],
                    user["role"],
                    user["department"]
                )
            except:
                # User already exists
                pass
    
    @staticmethod
    def display_user_profile():
        """Display current user profile"""
        if "user" in st.session_state:
            user = st.session_state.user
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"**👤 {user.get('full_name')}**")
                st.caption(f"Role: {user.get('role').upper()}")
            
            with col2:
                if st.button("👤 Profile"):
                    st.session_state.show_profile = not st.session_state.get("show_profile", False)
            
            with col3:
                if st.button("⚙️ Settings"):
                    st.session_state.show_settings = not st.session_state.get("show_settings", False)
            
            with col4:
                if st.button("🚪 Logout"):
                    st.session_state.authenticated = False
                    st.session_state.user = None
                    st.rerun()
    
    @staticmethod
    def check_permission(user_role: str, required_role: str) -> bool:
        """Check if user has required role"""
        role_hierarchy = {
            "patient": 1,
            "nurse": 2,
            "doctor": 3,
            "administrator": 4
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    @staticmethod
    def admin_manage_users_interface():
        """Admin interface for managing users"""
        st.markdown("### 👥 User Management")
        
        users = get_all_users()
        
        if users:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader("Active Users")
            
            with col2:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            with col3:
                if st.button("➕ New User", use_container_width=True):
                    st.session_state.show_create_user = True
            
            # Display users table
            import pandas as pd
            users_df = pd.DataFrame(users)
            users_df = users_df[['id', 'full_name', 'username', 'email', 'role', 'department', 'is_active', 'last_login']]
            
            st.dataframe(users_df, use_container_width=True, height=300)
            
            # Edit user form
            if st.session_state.get("show_create_user"):
                st.markdown("#### Create New User")
                with st.form("new_user_form"):
                    fn = st.text_input("Full Name")
                    un = st.text_input("Username")
                    em = st.text_input("Email")
                    pwd = st.text_input("Password", type="password")
                    rle = st.selectbox("Role", ["nurse", "doctor", "administrator"])
                    dept = st.text_input("Department")
                    
                    if st.form_submit_button("Create User"):
                        if register_user(un, pwd, em, fn, rle, dept):
                            st.success(f"✅ User {fn} created successfully")
                            st.session_state.show_create_user = False
                            st.rerun()
                        else:
                            st.error("❌ Failed to create user")
    
    @staticmethod
    def doctor_assign_patients_interface(doctor_id: int):
        """Interface for doctors to view their assigned patients"""
        st.markdown("### 👥 My Assigned Patients")
        
        from database import get_all_patients
        
        patients = get_doctor_patients(doctor_id)
        
        if patients:
            import pandas as pd
            patients_df = pd.DataFrame(patients)[['id', 'name', 'age', 'severity', 'mutation', 'risk_score', 'updated_at']]
            st.dataframe(patients_df, use_container_width=True, height=400)
        else:
            st.info("ℹ️ No patients assigned to you yet")
        
        with st.expander("➕ Request Patient Assignment"):
            st.info("Contact your administrator to assign patients to your account")

    @staticmethod
    def doctor_interface():
        """Display all patient data for doctors"""
        if "user" in st.session_state and st.session_state.user.get("role") == "doctor":
            st.markdown("### 🩺 Doctor's Dashboard")

            # Fetch all patient data
            patients = get_all_patients()  # Assuming this function exists in database.py

            if patients:
                st.dataframe(patients)
            else:
                st.info("No patient data available.")

        else:
            st.error("Unauthorized access. Only doctors can view this page.")

    @staticmethod
    def patient_registration():
        """Handle patient registration"""
        st.markdown("### 📝 Register as a Patient")

        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="Enter your email")
        username = st.text_input("Username", placeholder="Choose a username")
        password = st.text_input("Password", type="password", placeholder="Create a strong password")

        if st.button("Register"):
            if not all([full_name, email, username, password]):
                st.error("Please fill in all fields.")
            else:
                success = register_user(username, password, email, full_name, "patient", None)
                if success:
                    st.success("Registration successful! Please log in.")
                else:
                    st.error("Registration failed. Username or email might already exist.")
