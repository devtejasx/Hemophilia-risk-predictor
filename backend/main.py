"""
FastAPI Backend for Hemophilia AI Platform

REST API for ML predictions, chat, patient management, and analytics
Separates backend from Streamlit frontend
"""

from fastapi import FastAPI, HTTPException, Depends, logging
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from database import init_database
from backend.users import init_user_database, create_admin_if_not_exists
from backend.auth import SECRET_KEY
import backend.auth as auth_module

# Set JWT secret from environment or use default
jwt_secret = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
if jwt_secret != auth_module.SECRET_KEY:
    auth_module.SECRET_KEY = jwt_secret

# Import routers
from .routers import predict, chat, patients, analytics, auth
from .models import HealthResponse

# Configure logging
logging.getLogger(__name__)

# ============= STARTUP/SHUTDOWN EVENTS =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    print("🚀 Starting Hemophilia AI FastAPI Backend...")
    try:
        init_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
    
    try:
        init_user_database()
        print("✅ User authentication database initialized")
        create_admin_if_not_exists()
        print("✅ Admin user verified")
    except Exception as e:
        print(f"❌ User database initialization error: {e}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Hemophilia AI FastAPI Backend...")


# ============= CREATE FASTAPI APP =============

app = FastAPI(
    title="Hemophilia AI Platform - Backend API",
    description="REST API for ML predictions, clinical AI chat, patient management, and analytics",
    version="1.0.0",
    lifespan=lifespan
)

# ============= CORS CONFIGURATION =============

# Allow requests from Streamlit frontend and other clients
origins = [
    "http://localhost:8501",      # Streamlit default
    "http://localhost:3000",      # React default
    "http://localhost:8000",      # FastAPI default
    "http://127.0.0.1:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Development: Allow all origins
allow_all_origins = True

if allow_all_origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= ROUTES REGISTRATION =============

# Include routers
app.include_router(auth.router)        # Auth must be first (login/register)
app.include_router(predict.router)
app.include_router(chat.router)
app.include_router(patients.router)
app.include_router(analytics.router)

# ============= ROOT ENDPOINTS =============

@app.get("/", redirect_url="/docs")
async def root():
    """API root - redirects to documentation"""
    return {"message": "Hemophilia AI Platform Backend API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        database="connected",
        ml_models="loaded"
    )


# ============= EXAMPLE ENDPOINTS =============

@app.get("/info")
async def get_info():
    """Get API information"""
    return {
        "name": "Hemophilia AI Platform Backend",
        "version": "1.0.0",
        "endpoints": {
            "authentication": "/auth",
            "prediction": "/predict",
            "chat": "/chat",
            "patients": "/patients",
            "analytics": "/analytics"
        },
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "authentication": {
            "scheme": "Bearer JWT",
            "endpoints": {
                "login": "POST /auth/login",
                "register": "POST /auth/register",
                "refresh": "POST /auth/refresh",
                "me": "GET /auth/me"
            }
        }
    }


# ============= ERROR HANDLERS =============

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return {
        "error": str(exc),
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }


# ============= MIDDLEWARE =============

@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests"""
    print(f"📨 {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"📤 {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║  🏥 Hemophilia AI Platform - FastAPI Backend                  ║
    ║  ─────────────────────────────────────────────────────────────║
    ║  Starting on: http://localhost:8000                           ║
    ║  Documentation: http://localhost:8000/docs                    ║
    ║                                                                ║
    ║  🔐 AUTHENTICATION (NEW):                                     ║
    ║    • POST /auth/login      - Login with credentials          ║
    ║    • POST /auth/register   - Create new account               ║
    ║    • POST /auth/refresh    - Refresh access token            ║
    ║    • GET  /auth/me         - Get current user                 ║
    ║                                                                ║
    ║  📊 API ENDPOINTS:                                             ║
    ║    • /predict      - ML prediction endpoint                   ║
    ║    • /chat         - Clinical AI chat endpoint                ║
    ║    • /patients     - Patient management (protected)           ║
    ║    • /analytics    - Dashboard analytics (protected)          ║
    ║                                                                ║
    ║  Default Admin:                                               ║
    ║    Username: admin                                            ║
    ║    Password: Admin@2026                                       ║
    ║  ─────────────────────────────────────────────────────────────║
    ║  Press CTRL+C to stop                                         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
