# Streamlit + FastAPI Authentication Integration

Complete guide for integrating JWT authentication with your Streamlit medical AI application.

---

## 🎯 Overview

Your Streamlit app (port 8501) will communicate with the FastAPI backend (port 8000) for authentication. The flow:

1. User logs in in Streamlit
2. Streamlit sends credentials to FastAPI `/api/auth/login`
3. FastAPI returns access & refresh tokens
4. Streamlit stores tokens in session state
5. Streamlit includes token in all API requests (header: `Authorization: Bearer <token>`)
6. FastAPI validates token and processes request

---

## 📝 Complete Streamlit Integration Code

Save as `streamlit_auth_integration.py`:

```python
"""
Streamlit Authentication Integration Module
Handles JWT token management and API communication with FastAPI backend
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"
API_AUTH_URL = f"{API_BASE_URL}/api/auth"
API_PROTECTED_URL = f"{API_BASE_URL}/api/protected"
API_MEDICAL_URL = f"{API_BASE_URL}/api/medical"
API_ADMIN_URL = f"{API_BASE_URL}/api/admin"

# Session state keys
SESSION_ACCESS_TOKEN = "access_token"
SESSION_REFRESH_TOKEN = "refresh_token"
SESSION_USER = "user"
SESSION_TOKEN_EXPIRES = "token_expires_at"
SESSION_LOGIN_TIME = "login_time"

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        SESSION_ACCESS_TOKEN: None,
        SESSION_REFRESH_TOKEN: None,
        SESSION_USER: None,
        SESSION_TOKEN_EXPIRES: None,
        SESSION_LOGIN_TIME: None,
        "api_error": None,
        "api_success": None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# TOKEN MANAGEMENT
# ============================================================================

def is_token_expired() -> bool:
    """Check if access token is expired"""
    if not st.session_state.get(SESSION_TOKEN_EXPIRES):
        return True
    
    expires_at = st.session_state[SESSION_TOKEN_EXPIRES]
    return datetime.utcnow() > expires_at


def should_refresh_token() -> bool:
    """Check if token should be refreshed (expires in less than 5 minutes)"""
    if not st.session_state.get(SESSION_TOKEN_EXPIRES):
        return False
    
    expires_at = st.session_state[SESSION_TOKEN_EXPIRES]
    refresh_threshold = datetime.utcnow() + timedelta(minutes=5)
    return expires_at < refresh_threshold


def refresh_access_token() -> bool:
    """Refresh access token using refresh token"""
    refresh_token = st.session_state.get(SESSION_REFRESH_TOKEN)
    
    if not refresh_token:
        return False
    
    try:
        response = requests.post(
            f"{API_AUTH_URL}/refresh",
            json={"refresh_token": refresh_token},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Update tokens
            st.session_state[SESSION_ACCESS_TOKEN] = data["access_token"]
            st.session_state[SESSION_REFRESH_TOKEN] = data["refresh_token"]
            
            # Calculate expiration time
            expires_in_seconds = data.get("expires_in", 1800)
            st.session_state[SESSION_TOKEN_EXPIRES] = (
                datetime.utcnow() + timedelta(seconds=expires_in_seconds)
            )
            
            return True
        else:
            # Refresh token expired, logout
            logout()
            st.error("Session expired. Please login again.")
            return False
            
    except requests.exceptions.RequestException as e:
        st.error(f"Token refresh failed: {str(e)}")
        return False


def ensure_token_valid():
    """Ensure access token is valid, refresh if needed"""
    if is_token_expired():
        refresh_access_token()
    elif should_refresh_token():
        refresh_access_token()


# ============================================================================
# API REQUEST HELPERS
# ============================================================================

def get_auth_headers() -> Dict[str, str]:
    """Get headers with access token"""
    access_token = st.session_state.get(SESSION_ACCESS_TOKEN)
    
    headers = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    
    return headers


def api_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    require_auth: bool = True
) -> tuple[bool, Any]:
    """
    Make API request to FastAPI backend
    
    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        endpoint: API endpoint path (e.g., "/api/protected/user-stats")
        data: Request body data
        require_auth: Whether authentication is required
        
    Returns:
        (success: bool, data: response_data or error_message)
    """
    
    if require_auth:
        ensure_token_valid()
        
        if not st.session_state.get(SESSION_ACCESS_TOKEN):
            return False, "Not authenticated. Please login."
    
    url = f"{API_BASE_URL}{endpoint}"
    headers = get_auth_headers()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=10)
        elif method == "PUT":
            response = requests.put(url, json=data, headers=headers, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=10)
        else:
            return False, f"Unsupported HTTP method: {method}"
        
        if response.status_code in [200, 201]:
            return True, response.json()
        elif response.status_code == 401:
            logout()
            return False, "Unauthorized. Please login again."
        elif response.status_code == 403:
            return False, "Access denied. Insufficient permissions."
        elif response.status_code == 404:
            return False, f"Not found: {endpoint}"
        else:
            error_msg = response.json().get("detail", "Unknown error")
            return False, f"API Error ({response.status_code}): {error_msg}"
            
    except requests.exceptions.Timeout:
        return False, "Request timeout. API server may be down."
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to API server at {API_BASE_URL}"
    except Exception as e:
        return False, f"Request failed: {str(e)}"


# ============================================================================
# AUTHENTICATION FUNCTIONS
# ============================================================================

def login(email: str, password: str) -> bool:
    """
    Login with email and password
    
    Args:
        email: User email
        password: User password
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        response = requests.post(
            f"{API_AUTH_URL}/login",
            json={"email": email, "password": password},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store tokens and user info
            st.session_state[SESSION_ACCESS_TOKEN] = data["access_token"]
            st.session_state[SESSION_REFRESH_TOKEN] = data["refresh_token"]
            st.session_state[SESSION_USER] = data["user"]
            st.session_state[SESSION_LOGIN_TIME] = datetime.utcnow()
            
            # Calculate token expiration
            expires_in = data.get("expires_in", 1800)  # Default 30 minutes
            st.session_state[SESSION_TOKEN_EXPIRES] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            )
            
            st.session_state["api_success"] = "Login successful!"
            return True
            
        elif response.status_code == 401:
            st.session_state["api_error"] = "Invalid email or password"
            return False
        elif response.status_code == 403:
            st.session_state["api_error"] = "Account is inactive. Please contact support."
            return False
        else:
            error_detail = response.json().get("detail", "Login failed")
            st.session_state["api_error"] = f"Login error: {error_detail}"
            return False
            
    except requests.exceptions.Timeout:
        st.session_state["api_error"] = "Login request timed out"
        return False
    except requests.exceptions.ConnectionError:
        st.session_state["api_error"] = f"Cannot connect to server: {API_BASE_URL}"
        return False
    except Exception as e:
        st.session_state["api_error"] = f"Login error: {str(e)}"
        return False


def signup(email: str, username: str, password: str, role: str = "patient") -> bool:
    """
    Create new user account
    
    Args:
        email: User email
        username: Username
        password: Password
        role: User role (patient, doctor, admin)
        
    Returns:
        True if signup successful, False otherwise
    """
    try:
        response = requests.post(
            f"{API_AUTH_URL}/signup",
            json={
                "email": email,
                "username": username,
                "password": password,
                "role": role
            },
            timeout=10
        )
        
        if response.status_code == 201:
            data = response.json()
            
            # Auto-login after signup
            st.session_state[SESSION_ACCESS_TOKEN] = data["access_token"]
            st.session_state[SESSION_REFRESH_TOKEN] = data["refresh_token"]
            st.session_state[SESSION_USER] = data["user"]
            st.session_state[SESSION_LOGIN_TIME] = datetime.utcnow()
            
            expires_in = data.get("expires_in", 1800)
            st.session_state[SESSION_TOKEN_EXPIRES] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            )
            
            st.session_state["api_success"] = "Account created and logged in!"
            return True
            
        elif response.status_code == 409:
            st.session_state["api_error"] = "Email already registered. Please login or use different email."
            return False
        elif response.status_code == 400:
            error_detail = response.json().get("detail", "Invalid input")
            st.session_state["api_error"] = f"Signup error: {error_detail}"
            return False
        else:
            st.session_state["api_error"] = "Signup failed. Please try again."
            return False
            
    except requests.exceptions.RequestException as e:
        st.session_state["api_error"] = f"Signup error: {str(e)}"
        return False


def logout():
    """Logout current user"""
    try:
        # Send logout request to server
        if st.session_state.get(SESSION_ACCESS_TOKEN):
            requests.post(
                f"{API_AUTH_URL}/logout",
                headers=get_auth_headers(),
                timeout=5
            )
    except:
        pass  # Ignore errors during logout
    
    # Clear session state
    st.session_state[SESSION_ACCESS_TOKEN] = None
    st.session_state[SESSION_REFRESH_TOKEN] = None
    st.session_state[SESSION_USER] = None
    st.session_state[SESSION_TOKEN_EXPIRES] = None
    st.session_state[SESSION_LOGIN_TIME] = None
    st.session_state["api_success"] = "Logged out successfully"


def is_logged_in() -> bool:
    """Check if user is currently logged in"""
    return bool(st.session_state.get(SESSION_ACCESS_TOKEN))


def get_current_user() -> Optional[Dict]:
    """Get current logged-in user info"""
    return st.session_state.get(SESSION_USER)


def get_user_role() -> Optional[str]:
    """Get current user's role"""
    user = get_current_user()
    return user.get("role") if user else None


# ============================================================================
# API FUNCTIONS - PROTECTED ENDPOINTS
# ============================================================================

def get_user_profile() -> tuple[bool, Any]:
    """Get current user profile"""
    return api_request("GET", "/api/protected/user-stats", require_auth=True)


def get_patient_list() -> tuple[bool, Any]:
    """Get list of patients (doctor only)"""
    return api_request("GET", "/api/medical/patient-list", require_auth=True)


def run_prediction(patient_id: str) -> tuple[bool, Any]:
    """Run ML prediction for patient (doctor only)"""
    return api_request(
        "POST",
        "/api/medical/predict",
        data={"patient_id": patient_id},
        require_auth=True
    )


def get_public_data() -> tuple[bool, Any]:
    """Get public data (authentication optional)"""
    return api_request("GET", "/api/protected/public-data", require_auth=False)


# ============================================================================
# UI COMPONENTS
# ============================================================================

def show_login_page():
    """Display login page"""
    st.set_page_config(
        page_title="Medical AI - Login",
        page_icon="🏥",
        layout="centered"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🏥 Medical AI Assistant")
        st.markdown("---")
        
        # Tabs for login and signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.markdown("### Login")
            
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button", use_container_width=True):
                if email and password:
                    if login(email, password):
                        st.success("Login successful! Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(st.session_state.get("api_error", "Login failed"))
                else:
                    st.warning("Please enter email and password")
        
        with tab2:
            st.markdown("### Create Account")
            
            signup_email = st.text_input("Email", key="signup_email")
            signup_username = st.text_input("Username", key="signup_username")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_role = st.selectbox(
                "Role",
                ["patient", "doctor"],
                key="signup_role"
            )
            
            if st.button("Sign Up", key="signup_button", use_container_width=True):
                if signup_email and signup_username and signup_password:
                    if signup(signup_email, signup_username, signup_password, signup_role):
                        st.success("Account created! Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(st.session_state.get("api_error", "Signup failed"))
                else:
                    st.warning("Please fill in all fields")
        
        st.markdown("---")
        st.markdown("Test Credentials:")
        st.code(
            """Email: admin@medical-ai.com
Password: AdminPassword123!

Email: doctor1@medical-ai.com
Password: DoctorPassword123!

Email: patient1@medical-ai.com
Password: PatientPassword123!""",
            language="text"
        )


def show_user_profile():
    """Display user profile section"""
    user = get_current_user()
    
    with st.sidebar:
        st.markdown("### 👤 Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Role", user.get("role", "N/A").upper())
        with col2:
            verified = "✓ Verified" if user.get("is_verified") else "⏳ Pending"
            st.info(verified, icon="ℹ️")
        
        st.write(f"**Email:** {user.get('email')}")
        st.write(f"**Username:** {user.get('username')}")
        
        if st.button("Logout", key="logout_button", use_container_width=True):
            logout()
            st.rerun()


def show_protected_content():
    """Display protected content based on user role"""
    user = get_current_user()
    role = user.get("role")
    
    if role == "admin":
        st.markdown("## 🔐 Admin Panel")
        
        if st.button("Get System Stats"):
            success, data = api_request("GET", "/api/admin/stats", require_auth=True)
            if success:
                st.json(data)
            else:
                st.error(data)
    
    elif role == "doctor":
        st.markdown("## 👨‍⚕️ Doctor Portal")
        
        tab1, tab2 = st.tabs(["Patients", "Predictions"])
        
        with tab1:
            if st.button("Load Patient List"):
                success, data = get_patient_list()
                if success:
                    st.json(data)
                else:
                    st.error(data)
        
        with tab2:
            patient_id = st.text_input("Patient ID for prediction")
            if st.button("Run Prediction"):
                if patient_id:
                    success, data = run_prediction(patient_id)
                    if success:
                        st.json(data)
                    else:
                        st.error(data)
                else:
                    st.warning("Please enter patient ID")
    
    else:  # patient
        st.markdown("## 🏥 Patient Portal")
        
        if st.button("View My Profile"):
            success, data = get_user_profile()
            if success:
                st.json(data)
            else:
                st.error(data)


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Initialize session state
    initialize_session_state()
    
    # Show login page if not authenticated
    if not is_logged_in():
        show_login_page()
        return
    
    # Main app layout (logged in)
    st.set_page_config(
        page_title="Medical AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    
    st.markdown("# 🏥 Medical AI Assistant")
    
    # Show profile in sidebar
    show_user_profile()
    
    # Show success/error messages
    if st.session_state.get("api_success"):
        st.success(st.session_state["api_success"])
        st.session_state["api_success"] = None
    
    if st.session_state.get("api_error"):
        st.error(st.session_state["api_error"])
        st.session_state["api_error"] = None
    
    # Show content based on role
    show_protected_content()


if __name__ == "__main__":
    main()
```

---

## 🚀 Running the Full Stack

### Terminal 1: Start FastAPI Backend
```bash
python main.py
# Output: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Streamlit Frontend
```bash
streamlit run streamlit_auth_integration.py
# Output: Streamlit running on http://localhost:8501
```

### Access the Application
- **Streamlit App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

---

## 🔍 How It Works

### Login Flow
```
1. User enters email/password in Streamlit
2. Streamlit POST to /api/auth/login
3. FastAPI validates credentials
4. FastAPI returns JWT tokens + user info
5. Streamlit stores tokens in session_state
6. Page reloads to show main app
```

### Protected API Call Flow
```
1. User clicks button (e.g., "Get Patient List")
2. Streamlit checks token expiration
3. If expires soon, refresh token automatically
4. Streamlit adds "Authorization: Bearer <token>" header
5. Streamlit POST/GET to protected endpoint
6. FastAPI validates token using JWT
7. If valid, processes request and returns data
8. Streamlit displays data to user
```

### Token Refresh Flow
```
1. Token expires in < 5 minutes
2. Streamlit automatically calls /api/auth/refresh
3. FastAPI validates refresh token
4. FastAPI generates new access token
5. Streamlit updates session_state with new token
6. User never knows token was refreshed (seamless)
```

---

## 🧪 Testing the Integration

### Test Users

```
Admin:
  Email: admin@medical-ai.com
  Password: AdminPassword123!

Doctor:
  Email: doctor1@medical-ai.com
  Password: DoctorPassword123!

Patient:
  Email: patient1@medical-ai.com
  Password: PatientPassword123!
```

### Test Scenarios

1. **Login Success**
   - Open http://localhost:8501
   - Enter admin credentials
   - Click "Login"
   - Should see admin portal

2. **Login Failure**
   - Enter wrong password
   - Should see error message

3. **Role-Based Access**
   - Log in as doctor
   - Doctor portal should show
   - Admin panel should be hidden

4. **Token Refresh**
   - Login
   - Wait and make API calls
   - Token should refresh automatically

5. **Session Persistence**
   - Login
   - Refresh browser (F5)
   - Should stay logged in (tokens in session state)

---

## 🔒 Security Best Practices

1. **HTTPS in Production**
   - Always use HTTPS for API
   - Set `Authorization` header to HTTPS-only

2. **Token Storage**
   - Tokens stored in Streamlit `session_state`
   - Cleared on logout
   - Not persisted to disk

3. **Token Expiration**
   - Access tokens expire in 30 minutes
   - Refresh tokens expire in 7 days
   - Automatic refresh before expiration

4. **CORS Security**
   - Only localhost:8501 allowed for Streamlit
   - Configure in `auth_config.py`

5. **Environment Variables**
   - Never store SECRET_KEY in code
   - Use `.env` file
   - Don't commit `.env` to git

---

## 📊 Example: Doctor Dashboard

```python
# Show patient list with predictions
if role == "doctor":
    if st.button("Load Patients"):
        success, data = get_patient_list()
        
        if success:
            patients = data.get("patients", [])
            
            for patient in patients:
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**{patient['name']}** (ID: {patient['id']})")
                    
                    with col2:
                        st.metric("Risk Score", patient['risk_score'])
                    
                    with col3:
                        if st.button("Predict", key=patient['id']):
                            success, pred = run_prediction(patient['id'])
                            if success:
                                st.json(pred)
        else:
            st.error(data)
```

---

## 🆘 Troubleshooting

### "Cannot connect to API server"
- Make sure FastAPI is running: `python main.py`
- Check URL in code matches: `http://localhost:8000`

### "Unauthorized" Error
- Token may be expired or invalid
- Try logging out and logging back in
- Check token refresh is working

### "Insufficient permissions" Error
- Endpoint requires different role
- Try with different user (e.g., doctor for medical endpoints)

### CORS Error in Browser
- Add Streamlit URL to CORS origins in `auth_config.py`
- Make sure server is restarted after CORS change

---

## 📚 Next Steps

1. ✅ Basic login/logout - **Integrated**
2. ✅ Token refresh - **Implemented**
3. ✅ Protected API calls - **Implemented**
4. 📧 Add email verification
5. 🔐 Add password reset
6. 📱 Add logout all sessions
7. ⏱️ Add rate limiting

---

**All authentication is now fully integrated with your Streamlit medical AI application!**
