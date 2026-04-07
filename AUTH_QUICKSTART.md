"""
Authentication Quick Start Guide
Complete setup instructions for JWT authentication system
"""

# ============================================================================
# AUTHENTICATION QUICK START GUIDE
# ============================================================================

"""
# STEP 1: ENVIRONMENT SETUP
# ============================================================================

1. Copy the example environment file:
   
   cp .env.example .env

2. Edit .env with your settings:
   
   # Critical - Change this to a random 32+ character string!
   SECRET_KEY=your-super-secret-key-change-this-min-32-chars-very-important
   
   # Database URL (use SQLite for development)
   DATABASE_URL=sqlite:///./medical_ai.db
   
   # Other settings (can use defaults)
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   REFRESH_TOKEN_EXPIRE_DAYS=7
   BCRYPT_ROUNDS=12


# STEP 2: INSTALL DEPENDENCIES
# ============================================================================

1. Install required packages:
   
   pip install fastapi uvicorn sqlalchemy pydantic pyjwt passlib[bcrypt] python-multipart email-validator

2. Or update requirements.txt:
   
   fastapi==0.104.1
   uvicorn[standard]==0.24.0
   sqlalchemy==2.0.23
   pydantic==2.5.0
   pydantic-settings==2.1.0
   pyjwt==2.8.1
   passlib[bcrypt]==1.7.4
   python-multipart==0.0.6
   email-validator==2.1.0
   
   Then: pip install -r requirements.txt


# STEP 3: INITIALIZE DATABASE
# ============================================================================

1. Create database and seed test data:
   
   python init_db.py
   
   This will:
   - Create database tables
   - Add test users (admin, doctors, patients)
   - Display all created users

2. Available init_db.py commands:
   
   python init_db.py              # Initialize + seed (default)
   python init_db.py seed         # Add test data
   python init_db.py reset        # Drop, recreate, seed
   python init_db.py clear        # Delete all data
   python init_db.py show         # Display users
   python init_db.py help         # Help menu

3. Test Users Created:
   
   ADMIN:
     Email: admin@medical-ai.com
     Password: AdminPassword123!
   
   DOCTOR:
     Email: doctor1@medical-ai.com
     Password: DoctorPassword123!
     
     Email: doctor2@medical-ai.com
     Password: DoctorPassword456!
   
   PATIENT:
     Email: patient1@medical-ai.com
     Password: PatientPassword123!
     
     Email: patient2@medical-ai.com
     Password: PatientPassword456!


# STEP 4: RUN THE APPLICATION
# ============================================================================

1. Start the FastAPI server:
   
   python main.py
   
   Or manually:
   
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

2. The server should start at:
   
   http://localhost:8000

3. Access API documentation:
   
   Swagger UI: http://localhost:8000/docs
   ReDoc: http://localhost:8000/redoc


# STEP 5: TEST THE AUTHENTICATION
# ============================================================================

1. Login and get tokens:
   
   POST http://localhost:8000/api/auth/login
   
   Body (JSON):
   {
     "email": "admin@medical-ai.com",
     "password": "AdminPassword123!"
   }
   
   Response:
   {
     "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
     "expires_in": 1800,
     "user": {
       "user_id": "550e8400-e29b-41d4-a716-446655440123",
       "email": "admin@medical-ai.com",
       "username": "admin",
       "role": "admin",
       ...
     }
   }

2. Use access token in requests:
   
   GET http://localhost:8000/api/protected/user-stats
   
   Headers:
   {
     "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
   }

3. Refresh the access token:
   
   POST http://localhost:8000/api/auth/refresh
   
   Body (JSON):
   {
     "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
   }

4. Get current user profile:
   
   GET http://localhost:8000/api/protected/user-stats
   
   Headers:
   {
     "Authorization": "Bearer <your_access_token>"
   }


# STEP 6: FILE STRUCTURE
# ============================================================================

Capstone/
├── auth_config.py              # Configuration and settings
├── auth_models.py              # SQLAlchemy database models
├── auth_schemas.py             # Pydantic request/response schemas
├── auth_security.py            # JWT and password utilities
├── auth_database.py            # Database CRUD operations
├── auth_dependencies.py        # FastAPI dependency injection
├── auth_routes.py              # Authentication API endpoints
├── auth_examples.py            # Example protected routes
├── main.py                     # Main FastAPI application
├── init_db.py                  # Database initialization script
├── .env                        # Environment variables (create from .env.example)
├── .env.example                # Example environment file
└── requirements.txt            # Python dependencies


# STEP 7: INTEGRATE WITH YOUR FASTAPI APP
# ============================================================================

In your main FastAPI app file:

```python
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from auth_config import settings
from auth_models import Base, User
from auth_routes import router as auth_router
from auth_dependencies import get_current_user, require_doctor, require_admin, get_db

# Setup database
engine = create_engine(settings.DATABASE_URL)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)

# Create FastAPI app
app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes
app.include_router(auth_router)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Protected endpoint example
@app.get("/api/user/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    return {"user": current_user.email, "role": current_user.role.value}

# Doctor-only endpoint
@app.get("/api/medical/patients")
async def get_patients(current_doctor: User = Depends(require_doctor)):
    return {"patients": [...]}

# Run: uvicorn main:app --reload
```


# STEP 8: INTEGRATE WITH STREAMLIT FRONTEND
# ============================================================================

In your Streamlit app:

```python
import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'user' not in st.session_state:
    st.session_state.user = None

def login(email: str, password: str):
    \"\"\"Login and store tokens\"\"\"
    response = requests.post(
        f"{API_URL}/api/auth/login",
        json={"email": email, "password": password}
    )
    if response.status_code == 200:
        data = response.json()
        st.session_state.access_token = data["access_token"]
        st.session_state.refresh_token = data["refresh_token"]
        st.session_state.user = data["user"]
        st.success("Login successful!")
        return True
    else:
        st.error(f"Login failed: {response.json()}")
        return False

def get_api_headers():
    \"\"\"Get headers with auth token\"\"\"
    return {
        "Authorization": f"Bearer {st.session_state.access_token}",
        "Content-Type": "application/json"
    }

# Login form
st.title("Medical AI Assistant")

if not st.session_state.access_token:
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        login(email, password)
else:
    st.subheader(f"Welcome, {st.session_state.user['email']}!")
    st.write(f"Role: {st.session_state.user['role']}")
    
    # Example: Call protected API
    if st.button("Get Patients"):
        response = requests.get(
            f"{API_URL}/api/medical/patient-list",
            headers=get_api_headers()
        )
        st.json(response.json())
    
    if st.button("Logout"):
        requests.post(
            f"{API_URL}/api/auth/logout",
            headers=get_api_headers()
        )
        st.session_state.access_token = None
        st.session_state.user = None
        st.rerun()
```


# STEP 9: USE IN CURL/POSTMAN
# ============================================================================

1. Login (get tokens):

   curl -X POST "http://localhost:8000/api/auth/login" \\
     -H "Content-Type: application/json" \\
     -d '{
       "email": "admin@medical-ai.com",
       "password": "AdminPassword123!"
     }'

2. Use access token in header:

   curl -X GET "http://localhost:8000/api/protected/user-stats" \\
     -H "Authorization: Bearer <your_access_token>"

3. Refresh token:

   curl -X POST "http://localhost:8000/api/auth/refresh" \\
     -H "Content-Type: application/json" \\
     -d '{
       "refresh_token": "<your_refresh_token>"
     }'


# STEP 10: PRODUCTION DEPLOYMENT
# ============================================================================

For production deployment:

1. Update .env:
   - Change SECRET_KEY to a random 32+ character string
   - Set DATABASE_URL to PostgreSQL/MySQL
   - Set ENVIRONMENT=production
   - Increase BCRYPT_ROUNDS to 14-16 (slower but more secure)

2. Use production ASGI server:

   pip install gunicorn
   
   gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker

3. Setup HTTPS/SSL (required in production)

4. Use environment variable for SECRET_KEY (not in .env file)

5. Keep database backups

6. Monitor logs and errors

7. Setup email service for verification and password reset (optional)


# TROUBLESHOOTING
# ============================================================================

1. "Secret key too short" error:
   - Make sure SECRET_KEY in .env is at least 32 characters
   - Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"

2. "Database connection error":
   - Check DATABASE_URL setting
   - Verify database file/server exists and is accessible
   - Try: python init_db.py

3. "Token expired":
   - Use refresh token to get new access token
   - POST /api/auth/refresh with refresh_token

4. "Invalid credentials":
   - Check email and password are correct
   - Run: python init_db.py show (view all users)
   - Try: admin@medical-ai.com / AdminPassword123!

5. "Role not allowed":
   - Endpoint requires specific role (e.g., doctor)
   - Login with appropriate user role
   - Check endpoint documentation

6. "CORS error in browser":
   - Check CORS_ORIGINS in .env
   - Add your frontend URL to CORS origins in auth_config.py
   - Restart server


# API DOCUMENTATION
# ============================================================================

All endpoints documented at:

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc

Test endpoints directly from Swagger UI documentation!


# SECURITY NOTES
# ============================================================================

1. NEVER commit .env file to version control
   - Add .env to .gitignore

2. Always use HTTPS in production
   - Install SSL certificate

3. Use environment variables for secrets
   - Don't hardcode passwords or keys

4. Rotate SECRET_KEY regularly in production
   - Will invalidate all existing tokens

5. Monitor failed login attempts
   - Consider rate limiting (future enhancement)

6. Keep dependencies updated
   - pip install --upgrade -r requirements.txt

7. Use strong passwords for initial users
   - At least 8 characters, uppercase, digit, special char

8. Always verify email in production
   - Implement email verification flow

9. Use HTTPS-only cookies in production
   - Configure in dependencies


# NEXT STEPS
# ============================================================================

1. Optional: Add email verification
   - Create email verification endpoint
   - Send verification emails via SMTP

2. Optional: Add password reset flow
   - Create password reset request endpoint
   - Send reset token via email

3. Optional: Add two-factor authentication
   - Add TOTP/SMS-based 2FA
   - Implement 2FA verification

4. Optional: Add audit logging
   - Log all authentication events
   - Track user actions

5. Optional: Add rate limiting
   - Prevent brute force attacks
   - Rate limit login attempts

6. Optional: Add API key authentication
   - Support API key-based access
   - Create API key management endpoints


# SUPPORT
# ============================================================================

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in auth_*.py files
3. Check FastAPI documentation: https://fastapi.tiangolo.com/
4. Check SQLAlchemy documentation: https://docs.sqlalchemy.org/
5. Check JWT documentation: https://tools.ietf.org/html/rfc7519

"""

if __name__ == "__main__":
    # This is a documentation file, not meant to be executed
    print(__doc__)
