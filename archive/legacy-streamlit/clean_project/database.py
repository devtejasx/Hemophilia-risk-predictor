"""
Database layer for persistent data storage.
Consolidated SQLite database operations.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path


class Database:
    """SQLite database manager for the application."""
    
    def __init__(self, db_path: str = "app_data.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            
            cursor = self.connection.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            """)
            
            # Patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    clotting_factor REAL,
                    activity_level INTEGER,
                    compliance REAL,
                    recent_bleeds INTEGER DEFAULT 0,
                    hospitalization BOOLEAN DEFAULT 0,
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
                    risk_score REAL,
                    risk_category TEXT,
                    confidence REAL,
                    features TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            """)
            
            # Chat history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            # Analytics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
    
    # User operations
    
    def create_user(self, username: str, email: str, password_hash: str) -> int:
        """Create a new user.
        
        Args:
            username: Username
            email: Email address
            password_hash: Hashed password
        
        Returns:
            User ID
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            """, (username, email, password_hash))
            self.connection.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user by username.
        
        Args:
            username: Username
        
        Returns:
            User dictionary or None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp.
        
        Args:
            user_id: User ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,)
        )
        self.connection.commit()
    
    # Patient operations
    
    def add_patient(
        self,
        user_id: int,
        name: str,
        age: int,
        gender: str,
        clotting_factor: float,
        activity_level: int,
        compliance: float,
        bleeds: int = 0,
        hospitalization: bool = False,
        notes: str = ""
    ) -> int:
        """Add a new patient.
        
        Args:
            user_id: User ID
            name: Patient name
            age: Age
            gender: Gender
            clotting_factor: Clotting factor level
            activity_level: Activity level
            compliance: Treatment compliance
            bleeds: Number of bleeds
            hospitalization: Hospitalization flag
            notes: Clinical notes
        
        Returns:
            Patient ID
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO patients
            (user_id, name, age, gender, clotting_factor, activity_level,
             compliance, recent_bleeds, hospitalization, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, name, age, gender, clotting_factor, activity_level,
              compliance, bleeds, hospitalization, notes))
        self.connection.commit()
        return cursor.lastrowid
    
    def get_patients(self, user_id: int) -> List[Dict]:
        """Get all patients for a user.
        
        Args:
            user_id: User ID
        
        Returns:
            List of patient dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM patients WHERE user_id = ?
            ORDER BY updated_at DESC
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get a specific patient.
        
        Args:
            patient_id: Patient ID
        
        Returns:
            Patient dictionary or None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def update_patient(self, patient_id: int, **kwargs) -> None:
        """Update patient information.
        
        Args:
            patient_id: Patient ID
            **kwargs: Fields to update
        """
        if not kwargs:
            return
        
        kwargs['updated_at'] = datetime.now().isoformat()
        
        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [patient_id]
        
        cursor = self.connection.cursor()
        cursor.execute(f"""
            UPDATE patients SET {fields}
            WHERE id = ?
        """, values)
        self.connection.commit()
    
    def delete_patient(self, patient_id: int) -> None:
        """Delete a patient.
        
        Args:
            patient_id: Patient ID
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        self.connection.commit()
    
    # Prediction operations
    
    def add_prediction(
        self,
        patient_id: int,
        risk_score: float,
        risk_category: str,
        confidence: float,
        features: Dict
    ) -> int:
        """Save a prediction.
        
        Args:
            patient_id: Patient ID
            risk_score: Risk score
            risk_category: Risk category (LOW/MEDIUM/HIGH)
            confidence: Prediction confidence
            features: Feature dictionary
        
        Returns:
            Prediction ID
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO predictions
            (patient_id, risk_score, risk_category, confidence, features)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_id, risk_score, risk_category, confidence,
              json.dumps(features)))
        self.connection.commit()
        return cursor.lastrowid
    
    def get_predictions(self, patient_id: int, limit: int = 10) -> List[Dict]:
        """Get recent predictions for a patient.
        
        Args:
            patient_id: Patient ID
            limit: Number of predictions to retrieve
        
        Returns:
            List of prediction dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM predictions WHERE patient_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (patient_id, limit))
        
        predictions = []
        for row in cursor.fetchall():
            pred = dict(row)
            pred['features'] = json.loads(pred['features'])
            predictions.append(pred)
        
        return predictions
    
    # Chat operations
    
    def add_chat_message(self, user_id: int, role: str, message: str) -> int:
        """Add a chat message.
        
        Args:
            user_id: User ID
            role: Message role (user/assistant)
            message: Message content
        
        Returns:
            Message ID
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO chat_history (user_id, role, message)
            VALUES (?, ?, ?)
        """, (user_id, role, message))
        self.connection.commit()
        return cursor.lastrowid
    
    def get_chat_history(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get recent chat messages.
        
        Args:
            user_id: User ID
            limit: Number of messages to retrieve
        
        Returns:
            List of message dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM chat_history WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def clear_chat_history(self, user_id: int) -> None:
        """Clear chat history for a user.
        
        Args:
            user_id: User ID
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        self.connection.commit()
    
    # Analytics operations
    
    def record_metric(self, user_id: int, metric_name: str, metric_value: float) -> None:
        """Record an analytics metric.
        
        Args:
            user_id: User ID
            metric_name: Metric name
            metric_value: Metric value
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO analytics (user_id, metric_name, metric_value)
            VALUES (?, ?, ?)
        """, (user_id, metric_name, metric_value))
        self.connection.commit()
    
    def get_metrics(self, user_id: int, metric_name: str, days: int = 30) -> List[Dict]:
        """Get analytics metrics.
        
        Args:
            user_id: User ID
            metric_name: Metric name
            days: Number of days to look back
        
        Returns:
            List of metric dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM analytics
            WHERE user_id = ? AND metric_name = ?
            AND timestamp > datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        """, (user_id, metric_name, days))
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()


# Global database instance
_db_instance = None


def get_database(db_path: str = "app_data.db") -> Database:
    """Get or create database instance.
    
    Args:
        db_path: Path to database file
    
    Returns:
        Database instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance
