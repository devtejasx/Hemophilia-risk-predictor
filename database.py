import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import hashlib

DB_PATH = "hemophilia_clinic.db"

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_database():
    """Initialize database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table for multi-user system
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
    
    # User roles and permissions
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT UNIQUE NOT NULL,
            permissions TEXT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patient-Doctor assignments
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            assigned_doctor_id INTEGER NOT NULL,
            assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(assigned_doctor_id) REFERENCES users(id)
        )
    ''')
    
    # Audit trail for patient data changes
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            user_id INTEGER,
            field_name TEXT,
            old_value TEXT,
            new_value TEXT,
            action_type TEXT,
            change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(user_id) REFERENCES users(id)
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
    
    # Conversations table for storing chatbot interactions
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            user_message TEXT,
            gpt_response TEXT,
            conversation_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Doctor notes table
    c.execute('''
        CREATE TABLE IF NOT EXISTS doctor_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_name TEXT,
            note_content TEXT,
            note_category TEXT,
            severity TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Monitoring records table
    c.execute('''
        CREATE TABLE IF NOT EXISTS monitoring_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            test_type TEXT,
            result_value REAL,
            result_status TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Treatment history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS treatment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            treatment_date DATE,
            dose_units INTEGER,
            product_used TEXT,
            indication TEXT,
            bleeding_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')
    
    # Doctor dashboard analytics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS dashboard_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            metric_value REAL,
            metric_date DATE,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def add_patient(patient_data: Dict) -> int:
    """Add a new patient to the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO patients (
            name, age, gender, ethnicity, severity, mutation, blood_type, hla_type,
            dose, exposure, product_type, treatment_adherence, family_history,
            previous_inhibitor, joint_damage, bleeding_episodes, factor_level,
            immunosuppression, active_infection, vaccination_status, physical_activity,
            stress_level, comorbidities, risk_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_data.get('Name'),
        patient_data.get('Age'),
        patient_data.get('Gender'),
        patient_data.get('Ethnicity'),
        patient_data.get('Severity'),
        patient_data.get('Mutation'),
        patient_data.get('Blood_Type'),
        patient_data.get('HLA_Type'),
        patient_data.get('Dose'),
        patient_data.get('Exposure'),
        patient_data.get('Product_Type'),
        patient_data.get('Treatment_Adherence'),
        patient_data.get('Family_History'),
        patient_data.get('Previous_Inhibitor'),
        patient_data.get('Joint_Damage'),
        patient_data.get('Bleeding_Episodes'),
        patient_data.get('Factor_Level'),
        patient_data.get('Immunosuppression'),
        patient_data.get('Active_Infection'),
        patient_data.get('Vaccination_Status'),
        patient_data.get('Physical_Activity'),
        patient_data.get('Stress_Level'),
        patient_data.get('Comorbidities'),
        patient_data.get('Risk_Score')
    ))
    
    patient_id = c.lastrowid
    conn.commit()
    conn.close()
    return patient_id

def get_patient(patient_id: int) -> Optional[Dict]:
    """Retrieve patient information"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = c.fetchone()
    conn.close()
    
    return dict(row) if row else None

def get_all_patients() -> List[Dict]:
    """Fetch all patient data from the database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM patients")
    rows = c.fetchall()
    conn.close()

    return [dict(row) for row in rows]

def add_conversation(patient_id: int, user_message: str, gpt_response: str, conversation_type: str = "general"):
    """Store chatbot conversation"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO conversations (patient_id, user_message, gpt_response, conversation_type)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, user_message, gpt_response, conversation_type))
    
    conn.commit()
    conn.close()

def get_conversation_history(patient_id: int, limit: int = 50) -> List[Dict]:
    """Retrieve conversation history for a patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM conversations WHERE patient_id = ?
        ORDER BY created_at DESC LIMIT ?
    ''', (patient_id, limit))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def add_doctor_note(patient_id: int, doctor_name: str, note_content: str, category: str = "general", severity: str = "normal"):
    """Add doctor note for patient"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO doctor_notes (patient_id, doctor_name, note_content, note_category, severity)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, doctor_name, note_content, category, severity))
    
    conn.commit()
    conn.close()

def get_doctor_notes(patient_id: int) -> List[Dict]:
    """Get doctor notes for a patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM doctor_notes WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def add_monitoring_record(patient_id: int, test_type: str, result_value: float, result_status: str, notes: str = ""):
    """Add monitoring record"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO monitoring_records (patient_id, test_type, result_value, result_status, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, test_type, result_value, result_status, notes))
    
    conn.commit()
    conn.close()

def get_monitoring_records(patient_id: int) -> List[Dict]:
    """Get monitoring records for patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM monitoring_records WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def add_treatment_record(patient_id: int, treatment_date: str, dose_units: int, product_used: str, indication: str, bleeding_response: str):
    """Add treatment record"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO treatment_history (patient_id, treatment_date, dose_units, product_used, indication, bleeding_response)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, treatment_date, dose_units, product_used, indication, bleeding_response))
    
    conn.commit()
    conn.close()

def get_treatment_history(patient_id: int) -> List[Dict]:
    """Get treatment history for patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM treatment_history WHERE patient_id = ?
        ORDER BY treatment_date DESC
    ''', (patient_id,))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_dashboard_stats() -> Dict:
    """Get dashboard statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Total patients
    c.execute('SELECT COUNT(*) as count FROM patients')
    total_patients = c.fetchone()[0]
    
    # High risk patients
    c.execute('SELECT COUNT(*) as count FROM patients WHERE risk_score > 0.6')
    high_risk = c.fetchone()[0]
    
    # Average risk
    c.execute('SELECT AVG(risk_score) as avg_risk FROM patients')
    avg_risk = c.fetchone()[0] or 0
    
    # Severe cases
    c.execute('SELECT COUNT(*) as count FROM patients WHERE severity = "Severe"')
    severe_count = c.fetchone()[0]
    
    # Total conversations
    c.execute('SELECT COUNT(*) as count FROM conversations')
    total_conversations = c.fetchone()[0]
    
    # Recent doctor notes
    c.execute('SELECT COUNT(*) as count FROM doctor_notes WHERE created_at >= datetime("now", "-7 days")')
    recent_notes = c.fetchone()[0]
    
    conn.close()
    
    return {
        "total_patients": total_patients,
        "high_risk_patients": high_risk,
        "average_risk": avg_risk,
        "severe_cases": severe_count,
        "total_conversations": total_conversations,
        "recent_notes": recent_notes
    }

def search_patients(search_term: str) -> List[Dict]:
    """Search patients by name"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM patients WHERE name LIKE ? OR mutation LIKE ? OR severity LIKE ?
        ORDER BY created_at DESC
    ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
    
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def update_patient(patient_id: int, patient_data: Dict):
    """Update patient information"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Build dynamic update query
    update_fields = []
    values = []
    
    for key, value in patient_data.items():
        if key not in ['id', 'created_at']:
            update_fields.append(f"{key} = ?")
            values.append(value)
    
    if not update_fields:
        conn.close()
        return
    
    values.append(patient_id)
    query = f"UPDATE patients SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
    
    c.execute(query, values)
    conn.commit()
    conn.close()

def delete_patient(patient_id: int):
    """Delete patient and related records"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Delete all related records
    c.execute('DELETE FROM conversations WHERE patient_id = ?', (patient_id,))
    c.execute('DELETE FROM doctor_notes WHERE patient_id = ?', (patient_id,))
    c.execute('DELETE FROM monitoring_records WHERE patient_id = ?', (patient_id,))
    c.execute('DELETE FROM treatment_history WHERE patient_id = ?', (patient_id,))
    c.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
    
    conn.commit()
    conn.close()

# ============ USER MANAGEMENT FUNCTIONS ============

def register_user(username: str, password: str, email: str, full_name: str, role: str, department: Optional[str]) -> bool:
    """Register a new user in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        hashed_password = hash_password(password)
        c.execute(
            '''
            INSERT INTO users (username, password, email, full_name, role, department)
            VALUES (?, ?, ?, ?, ?, ?)
            ''',
            (username, hashed_password, email, full_name, role, department)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user and return user data"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    hashed_pwd = hash_password(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ? AND is_active = 1', (username, hashed_pwd))
    row = c.fetchone()
    
    if row:
        # Update last login
        c.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (row['id'],))
        conn.commit()
    
    conn.close()
    return dict(row) if row else None

def get_user(user_id: int) -> Optional[Dict]:
    """Get user by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    row = c.fetchone()
    conn.close()
    
    return dict(row) if row else None

def get_all_users() -> List[Dict]:
    """Get all active users"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE is_active = 1 ORDER BY full_name')
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_doctors() -> List[Dict]:
    """Get all doctors"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE role = \'doctor\' AND is_active = 1 ORDER BY full_name')
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def assign_patient_to_doctor(patient_id: int, doctor_id: int):
    """Assign patient to doctor"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Remove existing assignments
    c.execute('DELETE FROM patient_assignments WHERE patient_id = ?', (patient_id,))
    
    # Add new assignment
    c.execute('''
        INSERT INTO patient_assignments (patient_id, assigned_doctor_id)
        VALUES (?, ?)
    ''', (patient_id, doctor_id))
    
    conn.commit()
    conn.close()

def get_assigned_doctor(patient_id: int) -> Optional[Dict]:
    """Get doctor assigned to patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT u.* FROM users u
        JOIN patient_assignments pa ON u.id = pa.assigned_doctor_id
        WHERE pa.patient_id = ?
    ''', (patient_id,))
    row = c.fetchone()
    conn.close()
    
    return dict(row) if row else None

def get_doctor_patients(doctor_id: int) -> List[Dict]:
    """Get all patients assigned to a doctor"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT p.* FROM patients p
        JOIN patient_assignments pa ON p.id = pa.patient_id
        WHERE pa.assigned_doctor_id = ?
        ORDER BY p.updated_at DESC
    ''', (doctor_id,))
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def log_patient_change(patient_id: int, user_id: int, field_name: str, old_value: str, new_value: str, action_type: str = 'update'):
    """Log changes to patient data for audit trail"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO patient_audit_log (patient_id, user_id, field_name, old_value, new_value, action_type)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, user_id, field_name, str(old_value), str(new_value), action_type))
    
    conn.commit()
    conn.close()

def get_patient_audit_trail(patient_id: int) -> List[Dict]:
    """Get audit trail for a patient"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT pal.*, u.full_name FROM patient_audit_log pal
        LEFT JOIN users u ON pal.user_id = u.id
        WHERE pal.patient_id = ?
        ORDER BY pal.change_timestamp DESC
    ''', (patient_id,))
    rows = c.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
