"""
Database Configuration and Connection Management
PostgreSQL with SQLAlchemy ORM
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from backend.models_orm import Base, DatabaseSession
from dotenv import load_dotenv

load_dotenv()

# ============= DATABASE CONFIGURATION =============

# PostgreSQL Connection URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://hemophilia_user:hemophilia_pass@localhost:5432/hemophilia_db"
)

# Development mode - use SQLite fallback if PostgreSQL not available
USE_SQLITE = os.getenv("USE_SQLITE", "False").lower() == "true"

if USE_SQLITE:
    DATABASE_URL = "sqlite:///./hemophilia.db"

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "False").lower() == "true",
    poolclass=QueuePool,
    pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "40")),
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


# ============= INITIALIZATION =============

def init_db():
    """Initialize database - create all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        raise


def drop_db():
    """Drop all tables (USE WITH CAUTION)"""
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped")


# ============= DEPENDENCY INJECTION =============

def get_db() -> Session:
    """
    Dependency for FastAPI to get database session
    Usage: @app.get("/endpoint")
           async def endpoint(db: Session = Depends(get_db))
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============= CONNECTION UTILITIES =============

def get_db_session() -> Session:
    """Get a database session outside of FastAPI context"""
    return SessionLocal()


def close_db_session(session: Session):
    """Close a database session"""
    if session:
        session.close()


def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def get_db_status() -> dict:
    """Get database status information"""
    try:
        session = SessionLocal()
        
        # Count records in each table
        from backend.models_orm import Patient, Doctor, Conversation, TreatmentHistory, MonitoringRecord
        
        status = {
            "status": "connected",
            "database_url": DATABASE_URL.replace("postgresql://", "").split("@")[0] if "@" in DATABASE_URL else "sqlite",
            "patients": session.query(Patient).count(),
            "doctors": session.query(Doctor).count(),
            "conversations": session.query(Conversation).count(),
            "treatment_records": session.query(TreatmentHistory).count(),
            "monitoring_records": session.query(MonitoringRecord).count(),
        }
        
        session.close()
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ============= EXPORT FOR QUICK ACCESS =============

__all__ = [
    'engine',
    'SessionLocal',
    'init_db',
    'get_db',
    'get_db_session',
    'close_db_session',
    'test_connection',
    'get_db_status',
]
