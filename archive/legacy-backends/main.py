"""
Main FastAPI Application with Authentication Integration
Shows how to set up authentication in your FastAPI app
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

# Import authentication modules
from auth_config import settings
from auth_models import Base, User
from auth_routes import router as auth_router
from auth_dependencies import get_current_user, require_doctor, require_admin
from auth_examples import router as examples_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE SETUP
# ============================================================================

# Create database engine
# For development: SQLite
# For production: PostgreSQL or MySQL
DATABASE_URL = settings.DATABASE_URL or "sqlite:///./medical_ai.db"
logger.info(f"Using database: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all database tables
Base.metadata.create_all(bind=engine)
logger.info("Database tables created/verified")


def get_db():
    """Dependency for database session in routes"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# FASTAPI APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Medical AI Assistant API with Authentication",
    description="FastAPI backend with JWT authentication, user roles, and hospital data integration",
    version="1.0.0"
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# Add CORS middleware to allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React frontend
        "http://localhost:8501",      # Streamlit frontend
        "http://localhost:8000",      # Local backend
        "127.0.0.1:3000",
        "127.0.0.1:8501",
        "127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

logger.info("CORS middleware configured")

# ============================================================================
# ROUTE REGISTRATION
# ============================================================================

# Include authentication routes
app.include_router(auth_router)
logger.info("Authentication routes included: /api/auth/*")

# Include example protected routes
app.include_router(examples_router)
logger.info("Example routes included: /api/protected/*")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Medical AI Assistant API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "authentication": "/api/auth/login"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint
    
    Returns:
        Health status and database connection status
    """
    try:
        # Test database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "status": "healthy",
        "database": db_status,
        "authentication": "enabled"
    }


# ============================================================================
# EXAMPLE: MEDICAL DATA ENDPOINT (PROTECTED)
# ============================================================================

@app.get(
    "/api/medical/patient-list",
    summary="Get list of patients (doctors only)",
    tags=["Medical Data"]
)
async def get_patient_list(
    current_doctor: User = Depends(require_doctor),
    db: Session = Depends(get_db)
):
    """Get list of patients in the system
    
    Requirements:
    - User must be authenticated
    - User must have 'doctor' role
    
    Args:
        current_doctor: Current authenticated doctor user
        db: Database session
        
    Returns:
        List of patients with basic info
    """
    # Example: In real app, query patient database
    return {
        "doctor": current_doctor.email,
        "patients": [
            {
                "id": "P001",
                "name": "John Doe",
                "age": 52,
                "condition": "Hypertension",
                "risk_score": 0.65
            },
            {
                "id": "P002",
                "name": "Jane Smith",
                "age": 45,
                "condition": "Diabetes",
                "risk_score": 0.42
            }
        ]
    }


@app.post(
    "/api/medical/predict",
    summary="Run ML prediction (doctors only)",
    tags=["Medical Data"]
)
async def run_prediction(
    patient_id: str,
    current_doctor: User = Depends(require_doctor),
    db: Session = Depends(get_db)
):
    """Run ML model prediction for patient
    
    Requirements:
    - User must be authenticated
    - User must have 'doctor' role
    
    Args:
        patient_id: Patient ID for prediction
        current_doctor: Current authenticated doctor
        db: Database session
        
    Returns:
        Prediction results
    """
    import random
    
    return {
        "patient_id": patient_id,
        "prediction_by": current_doctor.email,
        "risk_score": round(random.uniform(0.1, 0.9), 2),
        "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "confidence": round(random.uniform(0.7, 0.99), 2),
        "recommendations": [
            "Schedule follow-up appointment",
            "Monitor vital signs",
            "Consider preventive measures"
        ]
    }


@app.get(
    "/api/admin/system-stats",
    summary="Get system statistics (admin only)",
    tags=["Admin"]
)
async def get_system_stats(
    current_admin: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get system statistics - admin only
    
    Requirements:
    - User must be authenticated
    - User must have 'admin' role
    
    Args:
        current_admin: Current authenticated admin
        db: Database session
        
    Returns:
        System statistics
    """
    from datetime import datetime, timedelta
    
    # Example: Count users by role
    try:
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        total_users = 0
        active_users = 0
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "admin": current_admin.email,
        "users": {
            "total": total_users,
            "active": active_users,
            "verified": db.query(User).filter(User.is_verified == True).count()
        },
        "system": {
            "uptime": "N/A",
            "database": "Connected",
            "authentication": "Active"
        }
    }


# ============================================================================
# EXAMPLE: USER-SPECIFIC DATA ENDPOINT
# ============================================================================

@app.get(
    "/api/user/profile",
    summary="Get current user profile",
    tags=["User"]
)
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get profile of currently logged-in user
    
    Requirements:
    - User must be authenticated
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        User profile information
    """
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat(),
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "type": "HTTPException"
    }


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("Medical AI API Starting Up")
    logger.info("=" * 60)
    logger.info(f"Database: {DATABASE_URL}")
    logger.info(f"JWT Algorithm: {settings.ALGORITHM}")
    logger.info(f"Access Token Expiry: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    logger.info(f"Refresh Token Expiry: {settings.REFRESH_TOKEN_EXPIRE_DAYS} days")
    logger.info(f"Allowed Roles: {', '.join(settings.ALLOWED_ROLES)}")
    logger.info("=" * 60)
    logger.info("API Documentation available at: http://localhost:8000/docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Medical AI API Shutting Down")
    engine.dispose()


# ============================================================================
# RUNNING THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run with: python main.py
    # Or: uvicorn main:app --reload
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )
