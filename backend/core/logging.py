"""
Production-Grade Logging Module
Centralized logging for all components with structured output
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
import json
from config import settings


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Enables better log aggregation and analysis
    """
    
    def format(self, record):
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        return json.dumps(log_data)


def setup_logging(logger_name: str = "hemophilia_ai") -> logging.Logger:
    """
    Configure logging for production
    
    Args:
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create logs directory
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Console Handler (for development)
    if settings.DEBUG:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(settings.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File Handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    file_formatter = JSONFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error File Handler (separate error log)
    error_log = str(log_dir / "error.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger


# Global logger instances
app_logger = setup_logging("hemophilia_ai.app")
api_logger = setup_logging("hemophilia_ai.api")
db_logger = setup_logging("hemophilia_ai.db")
ml_logger = setup_logging("hemophilia_ai.ml")
chatbot_logger = setup_logging("hemophilia_ai.chatbot")


def log_api_request(logger: logging.Logger, method: str, endpoint: str, 
                    status_code: int, duration: float, **kwargs):
    """
    Log API request with additional context
    
    Args:
        logger: Logger instance
        method: HTTP method
        endpoint: API endpoint
        status_code: HTTP status code
        duration: Request duration in seconds
        **kwargs: Additional context
    """
    logger.info(
        f"API Request: {method} {endpoint} - Status: {status_code} - Duration: {duration:.2f}s",
        extra={
            "request_id": kwargs.get("request_id"),
            "user_id": kwargs.get("user_id"),
        }
    )


def log_error(logger: logging.Logger, error_type: str, message: str, 
              exception: Exception = None, **context):
    """
    Log errors with context
    
    Args:
        logger: Logger instance
        error_type: Type of error (APIError, DatabaseError, etc.)
        message: Error message
        exception: Exception object if available
        **context: Additional context
    """
    if exception:
        logger.error(f"{error_type}: {message}", exc_info=exception, extra=context)
    else:
        logger.error(f"{error_type}: {message}", extra=context)


def log_ml_prediction(logger: logging.Logger, model_name: str, 
                      input_features: dict, prediction: float, 
                      confidence: float, **context):
    """
    Log ML prediction with details
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        input_features: Input features used
        prediction: Prediction result
        confidence: Confidence score
        **context: Additional context
    """
    logger.info(
        f"ML Prediction: {model_name} - Prediction: {prediction:.4f} - Confidence: {confidence:.4f}",
        extra={
            "model": model_name,
            "features_count": len(input_features),
            **context
        }
    )


def log_database_operation(logger: logging.Logger, operation: str, 
                          table: str, duration: float, **context):
    """
    Log database operations
    
    Args:
        logger: Logger instance
        operation: CRUD operation (CREATE, READ, UPDATE, DELETE)
        table: Database table name
        duration: Operation duration
        **context: Additional context
    """
    logger.info(
        f"DB Operation: {operation} on {table} - Duration: {duration:.3f}s",
        extra={
            "operation": operation,
            "table": table,
            **context
        }
    )
