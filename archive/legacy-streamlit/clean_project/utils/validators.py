"""
Input validation functions for form data and user inputs.
Ensures data integrity and provides helpful error messages.
"""

from typing import Tuple, Optional, Dict, Any
from constants import (
    MIN_CLOTTING_FACTOR, MAX_CLOTTING_FACTOR,
    MIN_ACTIVITY_LEVEL, MAX_ACTIVITY_LEVEL,
    MIN_COMPLIANCE, MAX_COMPLIANCE,
    MIN_AGE, MAX_AGE, MAX_BLEEDS
)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validator:
    """Collection of validation methods."""
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username format."""
        if not username:
            return False, "Username is required"
        if len(username) < 3:
            return False, "Username must be at least 3 characters"
        if len(username) > 50:
            return False, "Username must be less than 50 characters"
        if not username.replace("_", "").replace("-", "").isalnum():
            return False, "Username can only contain letters, numbers, hyphens, and underscores"
        return True, "Username is valid"
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """Validate email format."""
        if not email:
            return False, "Email is required"
        
        if "@" not in email:
            return False, "Email must contain @"
        
        parts = email.split("@")
        if len(parts) != 2:
            return False, "Email format is invalid"
        
        local_part, domain = parts
        
        if not local_part:
            return False, "Email must have content before @"
        
        if "." not in domain:
            return False, "Email domain must contain a dot"
        
        if domain.endswith("."):
            return False, "Email domain cannot end with a dot"
        
        return True, "Email is valid"
    
    @staticmethod
    def validate_password(password: str, min_length: int = 6) -> Tuple[bool, str]:
        """Validate password strength."""
        if not password:
            return False, "Password is required"
        
        if len(password) < min_length:
            return False, f"Password must be at least {min_length} characters"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        return True, "Password is valid"
    
    @staticmethod
    def validate_age(age: Any) -> Tuple[bool, str]:
        """Validate age."""
        try:
            age_int = int(age)
        except (ValueError, TypeError):
            return False, "Age must be a number"
        
        if age_int < MIN_AGE:
            return False, f"Age cannot be less than {MIN_AGE}"
        
        if age_int > MAX_AGE:
            return False, f"Age cannot be greater than {MAX_AGE}"
        
        return True, "Age is valid"
    
    @staticmethod
    def validate_clotting_factor(value: Any) -> Tuple[bool, str]:
        """Validate clotting factor level."""
        try:
            val = float(value)
        except (ValueError, TypeError):
            return False, "Clotting factor must be a number"
        
        if val < MIN_CLOTTING_FACTOR:
            return False, f"Clotting factor cannot be less than {MIN_CLOTTING_FACTOR}"
        
        if val > MAX_CLOTTING_FACTOR:
            return False, f"Clotting factor cannot be greater than {MAX_CLOTTING_FACTOR}"
        
        return True, "Clotting factor is valid"
    
    @staticmethod
    def validate_activity_level(value: Any) -> Tuple[bool, str]:
        """Validate activity level."""
        try:
            val = int(value)
        except (ValueError, TypeError):
            return False, "Activity level must be a number"
        
        if val < MIN_ACTIVITY_LEVEL:
            return False, f"Activity level cannot be less than {MIN_ACTIVITY_LEVEL}"
        
        if val > MAX_ACTIVITY_LEVEL:
            return False, f"Activity level cannot be greater than {MAX_ACTIVITY_LEVEL}"
        
        return True, "Activity level is valid"
    
    @staticmethod
    def validate_compliance(value: Any) -> Tuple[bool, str]:
        """Validate treatment compliance (0-1 or 0-100%)."""
        try:
            val = float(value)
        except (ValueError, TypeError):
            return False, "Compliance must be a number"
        
        # Allow both 0-1 and 0-100 ranges
        if val <= 1:
            # Assume 0-1 range
            if val < MIN_COMPLIANCE or val > MAX_COMPLIANCE:
                return False, f"Compliance must be between {MIN_COMPLIANCE} and {MAX_COMPLIANCE}"
        else:
            # Assume 0-100 range, convert to 0-1
            if val < 0 or val > 100:
                return False, "Compliance must be between 0 and 100 (or 0-1)"
        
        return True, "Compliance is valid"
    
    @staticmethod
    def validate_bleeds(value: Any) -> Tuple[bool, str]:
        """Validate number of bleeds."""
        try:
            val = int(value)
        except (ValueError, TypeError):
            return False, "Number of bleeds must be a whole number"
        
        if val < 0:
            return False, "Number of bleeds cannot be negative"
        
        if val > MAX_BLEEDS:
            return False, f"Number of bleeds cannot exceed {MAX_BLEEDS}"
        
        return True, "Bleed count is valid"
    
    @staticmethod
    def validate_patient_form(form_data: Dict[str, Any]) -> Tuple[bool, Dict[str, str]]:
        """Validate entire patient form with multiple fields."""
        errors = {}
        
        # Name
        if "name" in form_data:
            name = form_data["name"]
            if not name or len(name.strip()) == 0:
                errors["name"] = "Name is required"
            elif len(name) > 100:
                errors["name"] = "Name must be less than 100 characters"
        
        # Age
        if "age" in form_data:
            valid, msg = Validator.validate_age(form_data["age"])
            if not valid:
                errors["age"] = msg
        
        # Clotting factor
        if "clotting_factor" in form_data:
            valid, msg = Validator.validate_clotting_factor(form_data["clotting_factor"])
            if not valid:
                errors["clotting_factor"] = msg
        
        # Activity level
        if "activity_level" in form_data:
            valid, msg = Validator.validate_activity_level(form_data["activity_level"])
            if not valid:
                errors["activity_level"] = msg
        
        # Compliance
        if "compliance" in form_data:
            valid, msg = Validator.validate_compliance(form_data["compliance"])
            if not valid:
                errors["compliance"] = msg
        
        # Bleeds
        if "bleeds" in form_data:
            valid, msg = Validator.validate_bleeds(form_data["bleeds"])
            if not valid:
                errors["bleeds"] = msg
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_registration(username: str, email: str, password: str) -> Tuple[bool, Dict[str, str]]:
        """Validate registration form."""
        errors = {}
        
        # Username
        valid, msg = Validator.validate_username(username)
        if not valid:
            errors["username"] = msg
        
        # Email
        valid, msg = Validator.validate_email(email)
        if not valid:
            errors["email"] = msg
        
        # Password
        valid, msg = Validator.validate_password(password)
        if not valid:
            errors["password"] = msg
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_login(username: str, password: str) -> Tuple[bool, str]:
        """Validate login form."""
        if not username:
            return False, "Username is required"
        
        if not password:
            return False, "Password is required"
        
        return True, "Valid"


def safe_validate(validator_func, *args, **kwargs) -> Tuple[bool, str]:
    """Safely call a validator function and catch exceptions."""
    try:
        return validator_func(*args, **kwargs)
    except Exception as e:
        return False, f"Validation error: {str(e)}"
