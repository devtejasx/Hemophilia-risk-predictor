"""
Patient Service
Patient data management and database operations
"""

import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime
from models import PatientCreate, PatientUpdate, PatientResponse
from exceptions import PatientNotFound, DatabaseException, ValidationException


class PatientService:
    """Service for patient CRUD operations"""
    
    def __init__(self, db):
        self.db = db
    
    def create_patient(self, patient_data: PatientCreate) -> PatientResponse:
        """Create new patient"""
        try:
            query = '''
                INSERT INTO patients 
                (name, age, gender, severity, mutation, dose_intensity, exposure_days)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                patient_data.name,
                patient_data.age,
                patient_data.gender,
                patient_data.severity,
                patient_data.mutation,
                patient_data.dose_intensity,
                patient_data.exposure_days
            )
            
            patient_id = self.db.execute_insert(query, params)
            return self.get_patient(patient_id)
            
        except Exception as e:
            raise DatabaseException(f"Failed to create patient: {str(e)}")
    
    def get_patient(self, patient_id: int) -> PatientResponse:
        """Get patient by ID"""
        try:
            query = 'SELECT * FROM patients WHERE id = ?'
            row = self.db.execute_one(query, (patient_id,))
            
            if not row:
                raise PatientNotFound(patient_id)
            
            return self._row_to_patient(row)
            
        except PatientNotFound:
            raise
        except Exception as e:
            raise DatabaseException(f"Failed to get patient: {str(e)}")
    
    def get_all_patients(self, skip: int = 0, limit: int = 100) -> List[PatientResponse]:
        """Get all patients with pagination"""
        try:
            query = 'SELECT * FROM patients ORDER BY id DESC LIMIT ? OFFSET ?'
            rows = self.db.execute(query, (limit, skip))
            return [self._row_to_patient(row) for row in rows]
            
        except Exception as e:
            raise DatabaseException(f"Failed to fetch patients: {str(e)}")
    
    def update_patient(self, patient_id: int, update_data: PatientUpdate) -> PatientResponse:
        """Update patient data"""
        try:
            # Get existing patient first
            existing = self.get_patient(patient_id)
            
            # Build update query with only provided fields
            update_fields = []
            params = []
            
            if update_data.name is not None:
                update_fields.append('name = ?')
                params.append(update_data.name)
            if update_data.age is not None:
                update_fields.append('age = ?')
                params.append(update_data.age)
            if update_data.dose_intensity is not None:
                update_fields.append('dose_intensity = ?')
                params.append(update_data.dose_intensity)
            if update_data.exposure_days is not None:
                update_fields.append('exposure_days = ?')
                params.append(update_data.exposure_days)
            
            if not update_fields:
                return existing
            
            update_fields.append('updated_at = CURRENT_TIMESTAMP')
            params.append(patient_id)
            
            query = f'UPDATE patients SET {", ".join(update_fields)} WHERE id = ?'
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            
            return self.get_patient(patient_id)
            
        except Exception as e:
            raise DatabaseException(f"Failed to update patient: {str(e)}")
    
    def delete_patient(self, patient_id: int) -> bool:
        """Delete patient"""
        try:
            # Check if patient exists
            self.get_patient(patient_id)
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                # Delete related data first
                cursor.execute('DELETE FROM conversations WHERE patient_id = ?', (patient_id,))
                cursor.execute('DELETE FROM predictions WHERE patient_id = ?', (patient_id,))
                # Delete patient
                cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
                conn.commit()
            
            return True
            
        except Exception as e:
            raise DatabaseException(f"Failed to delete patient: {str(e)}")
    
    def search_patients(self, severity: Optional[str] = None, 
                       mutation: Optional[str] = None) -> List[PatientResponse]:
        """Search patients by criteria"""
        try:
            query = 'SELECT * FROM patients WHERE 1=1'
            params = []
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity)
            
            if mutation:
                query += ' AND mutation = ?'
                params.append(mutation)
            
            rows = self.db.execute(query, tuple(params))
            return [self._row_to_patient(row) for row in rows]
            
        except Exception as e:
            raise DatabaseException(f"Failed to search patients: {str(e)}")
    
    def update_patient_risk_score(self, patient_id: int, risk_score: float) -> None:
        """Update patient's risk score"""
        try:
            query = 'UPDATE patients SET risk_score = ? WHERE id = ?'
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (risk_score, patient_id))
                conn.commit()
        except Exception as e:
            raise DatabaseException(f"Failed to update risk score: {str(e)}")
    
    @staticmethod
    def _row_to_patient(row: sqlite3.Row) -> PatientResponse:
        """Convert database row to PatientResponse"""
        return PatientResponse(
            id=row['id'],
            name=row['name'],
            age=row['age'],
            gender=row['gender'],
            severity=row['severity'],
            mutation=row['mutation'],
            dose_intensity=row['dose_intensity'],
            exposure_days=row['exposure_days'],
            risk_score=row['risk_score'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )


def get_patient_service(db) -> PatientService:
    """Dependency injection for patient service"""
    return PatientService(db)
