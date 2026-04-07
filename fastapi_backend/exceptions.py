"""
Custom Exception Classes
Application-level exceptions with proper HTTP mappings
"""

from fastapi import HTTPException, status


class MedicalAIException(Exception):
    """Base exception for Medical AI Platform"""
    
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)


class PredictionException(MedicalAIException):
    """Raised when ML prediction fails"""
    
    def __init__(self, message: str = "Prediction failed"):
        super().__init__(message, status.HTTP_400_BAD_REQUEST, "PREDICTION_ERROR")


class ModelLoadException(MedicalAIException):
    """Raised when model loading fails"""
    
    def __init__(self, message: str = "Failed to load ML model"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, "MODEL_LOAD_ERROR")


class ChatException(MedicalAIException):
    """Raised when chat processing fails"""
    
    def __init__(self, message: str = "Chat processing failed"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, "CHAT_ERROR")


class DatabaseException(MedicalAIException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR, "DATABASE_ERROR")


class PatientNotFound(MedicalAIException):
    """Raised when patient is not found"""
    
    def __init__(self, patient_id: int):
        super().__init__(
            f"Patient with ID {patient_id} not found",
            status.HTTP_404_NOT_FOUND,
            "PATIENT_NOT_FOUND"
        )


class ValidationException(MedicalAIException):
    """Raised when input validation fails"""
    
    def __init__(self, message: str = "Validation failed"):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY, "VALIDATION_ERROR")


def exception_to_http_exception(exc: MedicalAIException) -> HTTPException:
    """Convert MedicalAIException to HTTPException"""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error": exc.error_code,
            "message": exc.message
        }
    )
