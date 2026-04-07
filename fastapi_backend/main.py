"""
Main FastAPI Application
Entry point for the Medical AI Platform backend
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime

from config import settings
from database import db
from services.prediction_service import prediction_service
from services.chat_service import chat_service
from routers import predictions_router, chat_router, patients_router, analytics_router
from models import HealthResponse


# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    
    # Startup
    print("🚀 Starting Medical AI Platform API...")
    try:
        # Initialize database
        db.init_tables()
        print("✅ Database initialized")
        
        # Load models
        if not prediction_service.models_loaded():
            print("⚠️ Warning: ML models not loaded")
        else:
            print("✅ ML models loaded")
        
        # Check chat service
        if chat_service.is_available():
            print("✅ Chat service available")
        else:
            print("⚠️ Warning: Chat service not available")
        
        print("✨ Application started successfully")
    except Exception as e:
        print(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down Medical AI Platform API...")


# ==================== Create FastAPI App ====================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Clinical decision support system with ML predictions and AI chat",
    lifespan=lifespan
)


# ==================== CORS Middleware ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check Endpoints ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        System health status
    """
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
        database_connected=True,
        models_loaded=prediction_service.models_loaded()
    )


@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint (all dependencies ready)
    
    Returns:
        Readiness status
    """
    ready = True
    
    if not prediction_service.models_loaded():
        ready = False
    
    if not chat_service.is_available():
        print("Warning: Chat service not available")
    
    return {
        "ready": ready,
        "models_loaded": prediction_service.models_loaded(),
        "chat_available": chat_service.is_available(),
        "timestamp": datetime.utcnow()
    }


# ==================== Root Endpoint ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health",
        "ready_url": "/ready",
        "endpoints": {
            "Predictions": "/api/v1/predictions",
            "Chat": "/api/v1/chat",
            "Patients": "/api/v1/patients",
            "Analytics": "/api/v1/analytics"
        },
        "timestamp": datetime.utcnow()
    }


# ==================== Register Routers ====================

app.include_router(predictions_router.router)
app.include_router(chat_router.router)
app.include_router(patients_router.router)
app.include_router(analytics_router.router)


# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return {
        "error": "HTTP_ERROR",
        "status_code": exc.status_code,
        "message": exc.detail,
        "timestamp": datetime.utcnow()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Unhandled exception: {exc}")
    return {
        "error": "INTERNAL_SERVER_ERROR",
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "message": "An unexpected error occurred",
        "timestamp": datetime.utcnow()
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
