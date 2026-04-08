"""
HEMOPHILIA CLINICAL DECISION SUPPORT SYSTEM
FastAPI Backend

Handles patient management, ML predictions, authentication, and analytics.

Author: AI Clinical Systems
Version: 2.0
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
import jwt
import os
from typing import Optional, List
import numpy as np
import sqlite3
import hashlib
import logging
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
DATABASE_URL = os.getenv("DATABASE_URL", "hemophilia_clinic.db")

# ============================================================================
# MODELS
# ============================================================================

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class PatientBase(BaseModel):
    name: str
    age: int = Field(..., ge=0, le=150)
    gender: str
    clotting_factor: float = Field(..., ge=0, le=100)
    previous_bleeds: int = Field(..., ge=0)
    activity_level: int = Field(..., ge=1, le=10)
    medication_compliance: float = Field(..., ge=0, le=1)
    treatment_type: str
    notes: Optional[str] = None


class PatientCreate(PatientBase):
    pass


class PatientResponse(PatientBase):
    id: int
    created_at: datetime


class PredictionRequest(BaseModel):
    patient_id: int


class PredictionResponse(BaseModel):
    risk_score: float
    risk_label: str
    risk_probability: float
    factors: dict
    timestamp: datetime


class ChatMessage(BaseModel):
    role: str
    message: str


class ChatResponse(BaseModel):
    messages: List[ChatMessage]
    latest_response: str


class AnalyticsResponse(BaseModel):
    total_patients: int
    high_risk_count: int
    avg_risk_score: float
    active_cases: int
    risk_distribution: dict
    trends: dict


# ============================================================================
# DATABASE SETUP
# ============================================================================

@contextmanager
def get_db():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_URL)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize database tables."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Patients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                clotting_factor REAL NOT NULL,
                previous_bleeds INTEGER NOT NULL,
                activity_level INTEGER NOT NULL,
                medication_compliance REAL NOT NULL,
                treatment_type TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                risk_score REAL NOT NULL,
                risk_label TEXT NOT NULL,
                factors TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        """)
        
        # Chat history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        conn.commit()
        logger.info("Database initialized successfully")


# ============================================================================
# AUTHENTICATION
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password."""
    return hash_password(plain_password) == hashed_password


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )


# ============================================================================
# DEPENDENCIES
# ============================================================================

def get_current_user(token: str):
    """Get current user from token."""
    payload = verify_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    
    return dict(user)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Hemophilia Clinical AI API",
    description="Backend API for clinical decision support",
    version="2.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501", "http://localhost:8502", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    init_database()
    logger.info("Application started")


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/register", response_model=Token)
async def register(user: UserRegister):
    """Register a new user."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE email = ? OR username = ?", 
                      (user.email, user.username))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create user
        password_hash = hash_password(user.password)
        cursor.execute(
            """INSERT INTO users (username, email, password_hash, full_name)
               VALUES (?, ?, ?, ?)""",
            (user.username, user.email, password_hash, user.full_name)
        )
        conn.commit()
        
        user_id = cursor.lastrowid
    
    # Create token
    access_token = create_access_token(
        data={"sub": user_id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"User registered: {user.username}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
        }
    }


@app.post("/api/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (credentials.email,))
        user = cursor.fetchone()
        
        if not user or not verify_password(credentials.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    access_token = create_access_token(
        data={"sub": user["id"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"User logged in: {user['email']}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
        }
    }


# ============================================================================
# PATIENT ENDPOINTS
# ============================================================================

@app.post("/api/patients", response_model=PatientResponse)
async def create_patient(patient: PatientCreate, current_user: dict = Depends(get_current_user)):
    """Create a new patient."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO patients 
               (user_id, name, age, gender, clotting_factor, previous_bleeds, 
                activity_level, medication_compliance, treatment_type, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (current_user["id"], patient.name, patient.age, patient.gender,
             patient.clotting_factor, patient.previous_bleeds,
             patient.activity_level, patient.medication_compliance,
             patient.treatment_type, patient.notes)
        )
        conn.commit()
        patient_id = cursor.lastrowid
    
    logger.info(f"Patient created: {patient.name} (ID: {patient_id})")
    
    return {
        **patient.dict(),
        "id": patient_id,
        "created_at": datetime.now(),
    }


@app.get("/api/patients", response_model=List[PatientResponse])
async def get_patients(current_user: dict = Depends(get_current_user)):
    """Get all patients for current user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, name, age, gender, clotting_factor, previous_bleeds,
                      activity_level, medication_compliance, treatment_type, notes, created_at
               FROM patients WHERE user_id = ? ORDER BY created_at DESC""",
            (current_user["id"],)
        )
        patients = [dict(row) for row in cursor.fetchall()]
    
    return patients


@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: int, current_user: dict = Depends(get_current_user)):
    """Get a specific patient."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM patients WHERE id = ? AND user_id = ?",
            (patient_id, current_user["id"])
        )
        patient = cursor.fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
    
    return dict(patient)


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

def calculate_risk_score(patient_data: dict) -> float:
    """Calculate risk score from patient data."""
    # Weighted risk calculation
    risk_score = (
        0.2 * (patient_data["age"] / 100) +
        0.3 * (1 - patient_data["clotting_factor"] / 100) +
        0.2 * (patient_data["previous_bleeds"] / 20) +
        0.15 * (patient_data["activity_level"] / 10) +
        0.15 * (1 - patient_data["medication_compliance"])
    )
    
    return float(np.clip(risk_score, 0, 1))


def get_risk_label(score: float) -> str:
    """Get risk label from score."""
    if score < 0.4:
        return "LOW RISK"
    elif score < 0.7:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"


@app.post("/api/predictions", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest, current_user: dict = Depends(get_current_user)):
    """Predict risk for a patient."""
    # Get patient
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM patients WHERE id = ? AND user_id = ?",
            (request.patient_id, current_user["id"])
        )
        patient = cursor.fetchone()
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_dict = dict(patient)
    
    # Calculate risk
    risk_score = calculate_risk_score(patient_dict)
    risk_label = get_risk_label(risk_score)
    
    # Save prediction
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO predictions (patient_id, risk_score, risk_label, factors)
               VALUES (?, ?, ?, ?)""",
            (request.patient_id, risk_score, risk_label, str(patient_dict))
        )
        conn.commit()
    
    logger.info(f"Prediction created for patient {request.patient_id}: {risk_label} ({risk_score:.3f})")
    
    return {
        "risk_score": risk_score,
        "risk_label": risk_label,
        "risk_probability": risk_score,
        "factors": {
            "age": patient_dict["age"],
            "clotting_factor": patient_dict["clotting_factor"],
            "previous_bleeds": patient_dict["previous_bleeds"],
            "activity_level": patient_dict["activity_level"],
            "compliance": patient_dict["medication_compliance"],
        },
        "timestamp": datetime.now(),
    }


# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/api/chat")
async def send_message(message: ChatMessage, current_user: dict = Depends(get_current_user)):
    """Send a chat message and get response."""
    
    # Save message
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
            (current_user["id"], message.role, message.message)
        )
        conn.commit()
    
    # Generate AI response
    ai_responses = {
        "help": "I can help you with clinical decisions, patient analysis, and treatment recommendations.",
        "risk": "Risk assessment depends on clotting factor levels, activity, and previous bleeds.",
        "treatment": "Treatment should be personalized based on individual patient profiles.",
        "protocol": "Please refer to the latest clinical guidelines for hemophilia management.",
        "default": "That's a great clinical question. Can you provide more details?",
    }
    
    keyword = next(
        (key for key in ai_responses.keys() if key in message.message.lower()),
        "default"
    )
    
    response = ai_responses[keyword]
    
    # Save AI response
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
            (current_user["id"], "ai", response)
        )
        conn.commit()
    
    return {"response": response}


@app.get("/api/chat-history")
async def get_chat_history(current_user: dict = Depends(get_current_user)):
    """Get chat history for current user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, message, created_at FROM chat_history 
               WHERE user_id = ? ORDER BY created_at ASC LIMIT 50""",
            (current_user["id"],)
        )
        messages = [dict(row) for row in cursor.fetchall()]
    
    return {"messages": messages}


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/analytics", response_model=AnalyticsResponse)
async def get_analytics(current_user: dict = Depends(get_current_user)):
    """Get analytics for current user."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total patients
        cursor.execute(
            "SELECT COUNT(*) FROM patients WHERE user_id = ?",
            (current_user["id"],)
        )
        total_patients = cursor.fetchone()[0]
        
        # Get all patients with latest predictions
        cursor.execute(
            """SELECT p.id, p.clotting_factor, p.previous_bleeds, p.activity_level, 
                      p.medication_compliance, p.age
               FROM patients p WHERE p.user_id = ?""",
            (current_user["id"],)
        )
        patients = [dict(row) for row in cursor.fetchall()]
        
        # Calculate metrics
        risk_scores = [calculate_risk_score(p) for p in patients]
        high_risk_count = sum(1 for r in risk_scores if r > 0.7)
        avg_risk_score = float(np.mean(risk_scores)) if risk_scores else 0
        active_cases = total_patients - high_risk_count
        
        # Risk distribution
        risk_distribution = {
            "low": sum(1 for r in risk_scores if r < 0.4),
            "medium": sum(1 for r in risk_scores if 0.4 <= r < 0.7),
            "high": sum(1 for r in risk_scores if r >= 0.7),
        }
        
        # Trends (mock data)
        trends = {
            "daily": [float(np.random.uniform(0.4, 0.8)) for _ in range(7)],
            "weekly": [float(np.random.uniform(0.4, 0.8)) for _ in range(4)],
        }
    
    return {
        "total_patients": total_patients,
        "high_risk_count": high_risk_count,
        "avg_risk_score": avg_risk_score,
        "active_cases": active_cases,
        "risk_distribution": risk_distribution,
        "trends": trends,
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Hemophilia Clinical AI API",
        "timestamp": datetime.now(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
