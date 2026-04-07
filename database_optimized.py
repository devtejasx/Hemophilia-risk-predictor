"""
Optimized Database Module
- Adds indexes for faster queries
- Implements pagination for large datasets
- Reduces query complexity
- Includes query result caching
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib
from cache_manager import cache_query

DB_PATH = "hemophilia_clinic.db"

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def _get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable query optimizer
    conn.execute("PRAGMA query_only = OFF")
    conn.execute("PRAGMA synchronous = NORMAL")  # Better performance
    conn.execute("PRAGMA cache_size = -64000")   # 64MB cache
    return conn


def init_database():
    """Initialize database with optimized indexes"""
    conn = _get_connection()
    c = conn.cursor()
    
    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT DEFAULT 'nurse',
            department TEXT,
            is_active INTEGER DEFAULT 1,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patients table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            ethnicity TEXT,
            severity TEXT,
            mutation TEXT,
            blood_type TEXT,
            hla_type TEXT,
            dose INTEGER,
            exposure INTEGER,
            product_type TEXT,
            treatment_adherence INTEGER,
            family_history TEXT,
            previous_inhibitor TEXT,
            joint_damage INTEGER,
            bleeding_episodes INTEGER,
            factor_level INTEGER,
            immunosuppression TEXT,
            active_infection TEXT,
            vaccination_status TEXT,
            physical_activity TEXT,
            stress_level TEXT,
            comorbidities TEXT,
            risk_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            user_id INTEGER,
            message_text TEXT,
            response_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Monitoring records for trend analysis
    c.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            factor_level REAL,
            bleeding_episodes INTEGER,
            joint_damage_score INTEGER,
            medication_adherence REAL,
            inhibitor_status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Create indexes for faster queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_severity ON patients(severity)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_mutation ON patients(mutation)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_risk_score ON patients(risk_score)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_created_at ON patients(created_at)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_age ON patients(age)')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_patient ON conversations(patient_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_patient ON monitoring_records(patient_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_monitoring_date ON monitoring_records(created_at)')
    
    # Composite indexes for common queries
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_severity_mutation ON patients(severity, mutation)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_patients_age_risk ON patients(age, risk_score)')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized with optimized indexes")


# ============ OPTIMIZED PAGINATION FUNCTIONS ============

def get_all_patients(page: int = 1, page_size: int = 50, 
                    order_by: str = 'created_at', 
                    order_dir: str = 'DESC') -> Dict[str, any]:
    """
    Get paginated patient list with sorting
    
    Args:
        page: Page number (1-indexed)
        page_size: Items per page
        order_by: Column to sort by
        order_dir: ASC or DESC
    
    Returns:
        Dict with patients list, total count, and pagination info
    """
    conn = _get_connection()
    c = conn.cursor()
    
    # Allowed columns for safety
    safe_columns = {
        'created_at', 'name', 'age', 'risk_score', 
        'id', 'severity', 'mutation'
    }
    order_by = order_by if order_by in safe_columns else 'created_at'
    order_dir = order_dir.upper()
    if order_dir not in ['ASC', 'DESC']:
        order_dir = 'DESC'
    
    # Get total count (cached)
    total_count = _get_cached_count("patients")
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    query = f'''
        SELECT * FROM patients 
        ORDER BY {order_by} {order_dir}
        LIMIT ? OFFSET ?
    '''
    
    c.execute(query, (page_size, offset))
    patients = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return {
        "data": patients,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }


def search_patients_paginated(query: str, page: int = 1, 
                             page_size: int = 50) -> Dict[str, any]:
    """
    Search patients with pagination
    Reduces dataset before filtering
    """
    conn = _get_connection()
    c = conn.cursor()
    
    search_pattern = f"%{query}%"
    
    # First get count for this search
    count_query = '''
        SELECT COUNT(*) as total FROM patients
        WHERE name LIKE ? OR mutation LIKE ? OR ethnicity LIKE ?
    '''
    c.execute(count_query, (search_pattern, search_pattern, search_pattern))
    total_count = c.fetchone()['total']
    
    # Then get paginated results
    offset = (page - 1) * page_size
    search_query = '''
        SELECT * FROM patients
        WHERE name LIKE ? OR mutation LIKE ? OR ethnicity LIKE ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    '''
    
    c.execute(search_query, (search_pattern, search_pattern, search_pattern, page_size, offset))
    patients = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return {
        "data": patients,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }


def get_monitoring_records_paginated(patient_id: int, page: int = 1,
                                     page_size: int = 100) -> Dict[str, any]:
    """
    Get paginated monitoring records for a patient
    Useful for large trend datasets
    """
    conn = _get_connection()
    c = conn.cursor()
    
    # Get count
    c.execute('SELECT COUNT(*) as total FROM monitoring_records WHERE patient_id = ?',
              (patient_id,))
    total_count = c.fetchone()['total']
    
    # Get paginated records
    offset = (page - 1) * page_size
    c.execute('''
        SELECT * FROM monitoring_records
        WHERE patient_id = ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    ''', (patient_id, page_size, offset))
    
    records = [dict(row) for row in c.fetchall()]
    conn.close()
    
    return {
        "data": records,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }


@cache_query(ttl=300)
def _get_cached_count(table: str) -> int:
    """Cached count query (5 min TTL)"""
    conn = _get_connection()
    c = conn.cursor()
    c.execute(f'SELECT COUNT(*) as count FROM {table}')
    count = c.fetchone()['count']
    conn.close()
    return count


# ============ OPTIMIZED CORE FUNCTIONS ============

def add_patient(name: str, age: int, gender: str, **kwargs) -> int:
    """Add patient with minimal data"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO patients 
        (name, age, gender, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (name, age, gender, datetime.now(), datetime.now()))
    
    conn.commit()
    patient_id = c.lastrowid
    conn.close()
    
    # Invalidate cache
    return patient_id


@cache_query(ttl=600)
def get_patient(patient_id: int) -> Optional[Dict]:
    """Get patient by ID (cached 10 min)"""
    conn = _get_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    patient = dict(c.fetchone()) if c.fetchone() else None
    conn.close()
    return patient


def add_monitoring_record(patient_id: int, factor_level: float,
                         bleeding_episodes: int, **kwargs) -> int:
    """Add monitoring record - optimized for frequent inserts"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO monitoring_records
        (patient_id, factor_level, bleeding_episodes, created_at)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, factor_level, bleeding_episodes, datetime.now()))
    
    conn.commit()
    record_id = c.lastrowid
    conn.close()
    return record_id


@cache_query(ttl=300)
def get_dashboard_stats() -> Dict:
    """Get aggregated statistics (cached 5 min)"""
    conn = _get_connection()
    c = conn.cursor()
    
    # Use aggregate functions to minimize data transfer
    c.execute('''
        SELECT 
            COUNT(*) as total_patients,
            COUNT(CASE WHEN risk_score > 0.5 THEN 1 END) as high_risk,
            COUNT(CASE WHEN risk_score <= 0.5 THEN 1 END) as low_risk,
            AVG(age) as avg_age,
            AVG(risk_score) as avg_risk,
            MIN(created_at) as oldest_patient,
            MAX(created_at) as newest_patient
        FROM patients
    ''')
    
    result = c.fetchone()
    conn.close()
    
    return dict(result) if result else {}


def register_user(username: str, password: str, email: str, 
                 full_name: str, role: str = 'nurse') -> bool:
    """Register new user"""
    conn = _get_connection()
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO users 
            (username, password, email, full_name, role, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (username, hash_password(password), email, full_name, role, 
              datetime.now(), datetime.now()))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


@cache_query(ttl=300)
def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user (cached)"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT id, username, email, full_name, role, is_active
        FROM users WHERE username = ? AND password = ?
    ''', (username, hash_password(password)))
    
    user = dict(c.fetchone()) if c.fetchone() else None
    conn.close()
    return user


def add_conversation(patient_id: int, message: str, 
                    response: str, user_id: Optional[int] = None) -> int:
    """Store conversation - optimized for fast inserts"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO conversations
        (patient_id, user_id, message_text, response_text, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, user_id, message, response, datetime.now()))
    
    conn.commit()
    conv_id = c.lastrowid
    conn.close()
    return conv_id


def get_conversation_history(patient_id: int, limit: int = 50) -> List[Dict]:
    """Get recent conversation history (optimized with LIMIT)"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM conversations
        WHERE patient_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (patient_id, limit))
    
    conversations = [dict(row) for row in c.fetchall()]
    conn.close()
    return conversations


def get_high_risk_patients(limit: int = 100) -> List[Dict]:
    """Get high-risk patients efficiently"""
    conn = _get_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT id, name, age, severity, risk_score, created_at
        FROM patients
        WHERE risk_score > 0.7
        ORDER BY risk_score DESC
        LIMIT ?
    ''', (limit,))
    
    patients = [dict(row) for row in c.fetchall()]
    conn.close()
    return patients


def batch_update_monitoring(patient_id: int, records: List[Dict]) -> bool:
    """Batch insert monitoring records for efficiency"""
    conn = _get_connection()
    c = conn.cursor()
    
    try:
        data = [
            (patient_id, r['factor_level'], r['bleeding_episodes'], datetime.now())
            for r in records
        ]
        
        c.executemany('''
            INSERT INTO monitoring_records
            (patient_id, factor_level, bleeding_episodes, created_at)
            VALUES (?, ?, ?, ?)
        ''', data)
        
        conn.commit()
        return True
    finally:
        conn.close()
