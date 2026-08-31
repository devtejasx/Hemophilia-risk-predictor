"""
Database Connection and Session Management
Dependency injection for database connections
"""

import sqlite3
from typing import Generator
from contextlib import contextmanager
from config import settings
from exceptions import DatabaseException


class DatabaseConnection:
    """SQLite database connection manager"""
    
    def __init__(self, db_path: str = settings.DATABASE_PATH):
        self.db_path = db_path
        self._ensure_connection()
    
    def _ensure_connection(self) -> None:
        """Ensure database file exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.close()
        except sqlite3.Error as e:
            raise DatabaseException(f"Failed to connect to database: {str(e)}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            raise DatabaseException(f"Database error: {str(e)}")
        finally:
            if conn:
                conn.close()
    
    def execute(self, query: str, params: tuple = ()):
        """Execute query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_one(self, query: str, params: tuple = ()):
        """Execute query and return first result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def execute_insert(self, query: str, params: tuple = ()):
        """Execute INSERT query and return last insert ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid
    
    def init_tables(self) -> None:
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Patients table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    mutation TEXT NOT NULL,
                    dose_intensity REAL NOT NULL,
                    exposure_days INTEGER NOT NULL,
                    risk_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(patient_id) REFERENCES patients(id)
                )
            ''')
            
            conn.commit()


# Global database instance
db = DatabaseConnection()


def get_db() -> Generator:
    """Dependency injection for database"""
    with db.get_connection() as connection:
        yield connection
