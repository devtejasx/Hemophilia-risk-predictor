"""
Dashboard Persistence Module
Handles saving and updating patient data from the doctor dashboard
with audit trail tracking and validation
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from database import (
    update_patient, log_patient_change, get_patient, 
    add_monitoring_record, add_treatment_record, add_doctor_note,
    DB_PATH
)

class DashboardPersistence:
    """Manager for doctor dashboard data persistence operations"""
    
    @staticmethod
    def validate_patient_data(patient_data: Dict) -> Tuple[bool, List[str]]:
        """Validate patient data before saving"""
        errors = []
        
        # Validate required fields
        if not patient_data.get('name') or patient_data.get('name').strip() == '':
            errors.append("Patient name is required")
        
        if 'age' in patient_data and patient_data['age'] is not None:
            try:
                age = int(patient_data['age'])
                if age < 0 or age > 150:
                    errors.append("Age must be between 0 and 150")
            except (ValueError, TypeError):
                errors.append("Age must be a valid number")
        
        # Validate numeric fields
        numeric_fields = ['dose', 'exposure', 'treatment_adherence', 'joint_damage', 
                         'bleeding_episodes', 'factor_level']
        for field in numeric_fields:
            if field in patient_data and patient_data[field] is not None:
                try:
                    val = float(patient_data[field])
                    if val < 0:
                        errors.append(f"{field} cannot be negative")
                except (ValueError, TypeError):
                    errors.append(f"{field} must be a valid number")
        
        # Validate risk_score is between 0 and 1
        if 'risk_score' in patient_data and patient_data['risk_score'] is not None:
            try:
                risk = float(patient_data['risk_score'])
                if not (0 <= risk <= 1):
                    errors.append("Risk score must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("Risk score must be a valid number")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def save_patient_update(patient_id: int, updated_data: Dict, user_id: Optional[int] = None) -> Tuple[bool, str]:
        """Save patient data updates with audit trail"""
        try:
            # Validate data
            is_valid, errors = DashboardPersistence.validate_patient_data(updated_data)
            if not is_valid:
                return False, f"Validation failed: {', '.join(errors)}"
            
            # Get current patient data for comparison
            current_patient = get_patient(patient_id)
            if not current_patient:
                return False, f"Patient {patient_id} not found"
            
            # Log changes for audit trail
            for key, new_value in updated_data.items():
                if key not in ['id', 'created_at', 'updated_at']:
                    old_value = current_patient.get(key)
                    if old_value != new_value:
                        log_patient_change(
                            patient_id=patient_id,
                            user_id=user_id,
                            field_name=key,
                            old_value=old_value,
                            new_value=new_value,
                            action_type='update'
                        )
            
            # Update patient record
            update_patient(patient_id, updated_data)
            return True, "Patient data saved successfully"
        
        except Exception as e:
            return False, f"Error saving patient data: {str(e)}"
    
    @staticmethod
    def save_monitoring_data(patient_id: int, test_type: str, result_value: float, 
                            result_status: str, notes: str = "") -> Tuple[bool, str]:
        """Save monitoring record with validation"""
        try:
            if not test_type or test_type.strip() == '':
                return False, "Test type is required"
            
            if result_value is None:
                return False, "Result value is required"
            
            try:
                result_value = float(result_value)
            except (ValueError, TypeError):
                return False, "Result value must be a valid number"
            
            add_monitoring_record(patient_id, test_type, result_value, result_status, notes)
            return True, "Monitoring record saved successfully"
        
        except Exception as e:
            return False, f"Error saving monitoring data: {str(e)}"
    
    @staticmethod
    def save_treatment_data(patient_id: int, treatment_date: str, dose_units: int, 
                           product_used: str, indication: str, bleeding_response: str) -> Tuple[bool, str]:
        """Save treatment record with validation"""
        try:
            if not treatment_date or treatment_date.strip() == '':
                return False, "Treatment date is required"
            
            if not product_used or product_used.strip() == '':
                return False, "Product used is required"
            
            if not indication or indication.strip() == '':
                return False, "Indication is required"
            
            try:
                dose_units = int(dose_units)
                if dose_units <= 0:
                    return False, "Dose units must be positive"
            except (ValueError, TypeError):
                return False, "Dose units must be a valid number"
            
            add_treatment_record(patient_id, treatment_date, dose_units, product_used, 
                               indication, bleeding_response)
            return True, "Treatment record saved successfully"
        
        except Exception as e:
            return False, f"Error saving treatment data: {str(e)}"
    
    @staticmethod
    def save_clinical_note(patient_id: int, doctor_name: str, note_content: str, 
                          category: str = "General", severity: str = "Normal") -> Tuple[bool, str]:
        """Save clinical note with validation"""
        try:
            if not doctor_name or doctor_name.strip() == '':
                return False, "Doctor name is required"
            
            if not note_content or note_content.strip() == '':
                return False, "Note content is required"
            
            if len(note_content) > 5000:
                return False, "Note content exceeds maximum length (5000 characters)"
            
            add_doctor_note(patient_id, doctor_name, note_content, category, severity)
            return True, "Clinical note saved successfully"
        
        except Exception as e:
            return False, f"Error saving clinical note: {str(e)}"
    
    @staticmethod
    def batch_update_patient_data(patient_updates: List[Dict]) -> Dict:
        """Batch update multiple patients' data"""
        results = {
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        for update in patient_updates:
            patient_id = update.get('id')
            if not patient_id:
                results["errors"].append("Patient ID missing in update")
                results["failed"] += 1
                continue
            
            success, message = DashboardPersistence.save_patient_update(
                patient_id, 
                update,
                update.get('user_id')
            )
            
            if success:
                results["successful"] += 1
            else:
                results["failed"] += 1
                results["errors"].append(f"Patient {patient_id}: {message}")
        
        return results
    
    @staticmethod
    def get_edit_summary(patient_id: int, days: int = 7) -> Dict:
        """Get summary of recent edits/changes to patient record"""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            c.execute('''
                SELECT * FROM patient_audit_log
                WHERE patient_id = ? AND change_timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY change_timestamp DESC
            ''', (patient_id, days))
            
            changes = c.fetchall()
            conn.close()
            
            # Group changes by field
            summary = {}
            for change in changes:
                field = change['field_name']
                if field not in summary:
                    summary[field] = []
                
                summary[field].append({
                    'old_value': change['old_value'],
                    'new_value': change['new_value'],
                    'changed_by': change['full_name'] or 'System',
                    'timestamp': change['change_timestamp']
                })
            
            return {
                "total_changes": len(changes),
                "fields_modified": len(summary),
                "changes_by_field": summary
            }
        
        except Exception as e:
            return {
                "error": str(e),
                "total_changes": 0,
                "fields_modified": 0
            }
    
    @staticmethod
    def export_patient_snapshot(patient_id: int) -> Optional[Dict]:
        """Export complete patient snapshot at current time"""
        try:
            patient = get_patient(patient_id)
            if not patient:
                return None
            
            snapshot = {
                "patient": patient,
                "exported_at": datetime.now().isoformat(),
                "record_type": "patient_snapshot"
            }
            
            return snapshot
        
        except Exception as e:
            return None

    @staticmethod
    def bulk_edit_start(patient_ids: List[int], user_id: int) -> str:
        """Start bulk edit session and return session ID"""
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            # Create session tracking (could be enhanced with dedicated table)
            session_id = f"bulk_{user_id}_{datetime.now().timestamp()}"
            return session_id
        
        except Exception as e:
            return None
