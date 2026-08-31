"""
Database Initialization Script
Creates database schema and optionally seeds test data
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from auth_config import settings
from auth_models import Base, User, UserRole, RefreshToken
from auth_security import hash_password, generate_user_id, generate_token_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATABASE SETUP
# ============================================================================

DATABASE_URL = settings.DATABASE_URL or "sqlite:///./medical_ai.db"
logger.info(f"Database URL: {DATABASE_URL}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize database - create all tables"""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✓ Database tables created successfully")


def seed_test_data():
    """Seed database with test users"""
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing_users = db.query(User).count()
        if existing_users > 0:
            logger.info(f"Database already contains {existing_users} users, skipping seed")
            return
        
        logger.info("Seeding test data...")
        
        # Create admin user
        admin = User(
            user_id=generate_user_id(),
            email="admin@medical-ai.com",
            username="admin",
            full_name="System Administrator",
            hashed_password=hash_password("AdminPassword123!"),
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_login=datetime.utcnow()
        )
        db.add(admin)
        logger.info("✓ Created admin user: admin@medical-ai.com / AdminPassword123!")
        
        # Create doctor users
        doctors = [
            {
                "email": "doctor1@medical-ai.com",
                "username": "dr_smith",
                "full_name": "Dr. James Smith",
                "password": "DoctorPassword123!"
            },
            {
                "email": "doctor2@medical-ai.com",
                "username": "dr_johnson",
                "full_name": "Dr. Sarah Johnson",
                "password": "DoctorPassword456!"
            }
        ]
        
        for doc_data in doctors:
            doctor = User(
                user_id=generate_user_id(),
                email=doc_data["email"],
                username=doc_data["username"],
                full_name=doc_data["full_name"],
                hashed_password=hash_password(doc_data["password"]),
                role=UserRole.DOCTOR,
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
            db.add(doctor)
            logger.info(f"✓ Created doctor user: {doc_data['email']} / {doc_data['password']}")
        
        # Create patient users
        patients = [
            {
                "email": "patient1@medical-ai.com",
                "username": "john_doe",
                "full_name": "John Doe",
                "password": "PatientPassword123!"
            },
            {
                "email": "patient2@medical-ai.com",
                "username": "jane_smith",
                "full_name": "Jane Smith",
                "password": "PatientPassword456!"
            }
        ]
        
        for pat_data in patients:
            patient = User(
                user_id=generate_user_id(),
                email=pat_data["email"],
                username=pat_data["username"],
                full_name=pat_data["full_name"],
                hashed_password=hash_password(pat_data["password"]),
                role=UserRole.PATIENT,
                is_active=True,
                is_verified=False,  # Patients start unverified
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                last_login=None
            )
            db.add(patient)
            logger.info(f"✓ Created patient user: {pat_data['email']} / {pat_data['password']}")
        
        db.commit()
        logger.info(f"✓ Successfully seeded {len(doctors) + len(patients) + 1} test users")
        
    except Exception as e:
        db.rollback()
        logger.error(f"✗ Error seeding data: {e}")
        raise
    finally:
        db.close()


def clear_all_data():
    """Clear all data from database (use with caution!)"""
    logger.warning("⚠ WARNING: Clearing all data from database!")
    db = SessionLocal()
    
    try:
        # Delete all tokens first (foreign key constraint)
        deleted_tokens = db.query(RefreshToken).delete()
        logger.info(f"Deleted {deleted_tokens} refresh tokens")
        
        # Delete all users
        deleted_users = db.query(User).delete()
        logger.info(f"Deleted {deleted_users} users")
        
        db.commit()
        logger.info("✓ All data cleared")
        
    except Exception as e:
        db.rollback()
        logger.error(f"✗ Error clearing data: {e}")
        raise
    finally:
        db.close()


def drop_all_tables():
    """Drop all tables (use with caution!)"""
    logger.warning("⚠ WARNING: Dropping all database tables!")
    Base.metadata.drop_all(bind=engine)
    logger.info("✓ All tables dropped")


def show_users():
    """Display all users in the database"""
    db = SessionLocal()
    
    try:
        users = db.query(User).all()
        
        if not users:
            logger.info("No users in database")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("USER DATABASE CONTENTS")
        logger.info("=" * 80)
        
        for user in users:
            logger.info(f"\nUser ID: {user.user_id}")
            logger.info(f"  Email: {user.email}")
            logger.info(f"  Username: {user.username}")
            logger.info(f"  Full Name: {user.full_name}")
            logger.info(f"  Role: {user.role.value}")
            logger.info(f"  Active: {user.is_active}")
            logger.info(f"  Verified: {user.is_verified}")
            logger.info(f"  Created: {user.created_at}")
            logger.info(f"  Last Login: {user.last_login}")
        
        logger.info("\n" + "=" * 80)
        logger.info(f"Total Users: {len(users)}")
        logger.info("=" * 80 + "\n")
        
    finally:
        db.close()


def main():
    """Main initialization function"""
    import sys
    
    logger.info("=" * 80)
    logger.info("MEDICAL AI DATABASE INITIALIZER")
    logger.info("=" * 80)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "reset":
            """Reset: Drop tables and recreate"""
            logger.warning("Resetting database...")
            drop_all_tables()
            init_database()
            seed_test_data()
            show_users()
            
        elif command == "seed":
            """Seed: Add test data to existing database"""
            init_database()
            seed_test_data()
            show_users()
            
        elif command == "clear":
            """Clear: Delete all data but keep tables"""
            clear_all_data()
            
        elif command == "drop":
            """Drop: Delete all tables"""
            response = input("Are you sure? This will delete all tables. (yes/no): ")
            if response.lower() == "yes":
                drop_all_tables()
            else:
                logger.info("Cancelled")
                
        elif command == "show":
            """Show: Display all users"""
            show_users()
            
        elif command == "help":
            print_help()
            
        else:
            logger.error(f"Unknown command: {command}")
            print_help()
    else:
        # Default: Initialize and seed
        logger.info("Running default initialization (init + seed)...")
        init_database()
        seed_test_data()
        show_users()
    
    logger.info("=" * 80)
    logger.info("Done!")
    logger.info("=" * 80)


def print_help():
    """Print help message"""
    help_text = """
Database Initialization Commands:
    
    python init_db.py              - Initialize database and seed test data (default)
    python init_db.py seed         - Seed test data to existing database
    python init_db.py reset        - Drop all tables and recreate with seed data
    python init_db.py clear        - Delete all data but keep tables
    python init_db.py drop         - Drop all tables (WARNING: Irreversible)
    python init_db.py show         - Display all users in database
    python init_db.py help         - Show this help message

Test Users (after seeding):
    
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
"""
    print(help_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
        exit(1)
