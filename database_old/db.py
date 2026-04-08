"""
Database operations module
"""

import sqlite3
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    """SQLite database wrapper"""
    
    def __init__(self, db_path: str = "hemophilia.db"):
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize database"""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _create_tables(self) -> None:
        """Create necessary tables"""
        try:
            cursor = self.conn.cursor()
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    ethnicity TEXT,
                    severity TEXT,
                    mutation TEXT,
                    blood_type TEXT,
                    dose REAL,
                    exposure REAL,
                    treatment_adherence REAL,
                    risk_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    risk_score REAL,
                    rf_score REAL,
                    xgb_score REAL,
                    features JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def insert_patient(self, patient_data: Dict[str, Any]) -> bool:
        """Insert new patient"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO patients (
                    id, name, age, gender, ethnicity, severity, mutation,
                    blood_type, dose, exposure, treatment_adherence, risk_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_data.get('id'),
                patient_data.get('name'),
                patient_data.get('age'),
                patient_data.get('gender'),
                patient_data.get('ethnicity'),
                patient_data.get('severity'),
                patient_data.get('mutation'),
                patient_data.get('blood_type'),
                patient_data.get('dose'),
                patient_data.get('exposure'),
                patient_data.get('treatment_adherence'),
                patient_data.get('risk_score', 0)
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting patient: {e}")
            return False
    
    def get_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all patients"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM patients ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching patients: {e}")
            return []
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching patient: {e}")
            return None
    
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Insert prediction"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    id, patient_id, risk_score, rf_score, xgb_score, features
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction_data.get('id'),
                prediction_data.get('patient_id'),
                prediction_data.get('risk_score'),
                prediction_data.get('rf_score'),
                prediction_data.get('xgb_score'),
                json.dumps(prediction_data.get('features', {}))
            ))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error inserting prediction: {e}")
            return False
    
    def add_chat_message(self, patient_id: str, role: str, content: str) -> bool:
        """Add chat message"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO chat_history (patient_id, role, content)
                VALUES (?, ?, ?)
            """, (patient_id, role, content))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding chat message: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Singleton instance
_database = None


def get_database(db_path: str = "hemophilia.db") -> Database:
    """Get database instance"""
    global _database
    if _database is None:
        _database = Database(db_path)
    return _database
