"""
Caching Layer for ML Predictions
Improves performance by caching prediction results and model outputs
"""

from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import pickle
from config import settings
from logging_config import app_logger
import time


class PredictionCache:
    """
    In-memory cache for prediction results
    In production, use Redis for distributed caching
    """
    
    def __init__(self, ttl_seconds: int = None):
        """
        Initialize cache
        
        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from input data"""
        sorted_data = json.dumps(data, sort_keys=True)
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get item from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        if datetime.utcnow() > entry["expires_at"]:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set item in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = {
            "value": value,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.ttl),
            "created_at": datetime.utcnow()
        }
    
    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """Clear all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if datetime.utcnow() > entry["expires_at"]
        ]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl_seconds": self.ttl
        }


class ModelCache:
    """
    Cache for trained models and explainers
    Prevents repeated loading from disk
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.cache: Dict[str, Dict[str, Any]] = {}
        return cls._instance
    
    def load_model(self, model_name: str, loader: Callable) -> Any:
        """
        Load model with caching
        
        Args:
            model_name: Name of the model
            loader: Callable that loads the model
            
        Returns:
            Loaded model
        """
        if model_name in self.cache:
            app_logger.info(f"Model loaded from cache: {model_name}")
            return self.cache[model_name]["model"]
        
        app_logger.info(f"Loading model from disk: {model_name}")
        start_time = time.time()
        model = loader()
        load_time = time.time() - start_time
        
        self.cache[model_name] = {
            "model": model,
            "loaded_at": datetime.utcnow(),
            "load_time": load_time
        }
        
        app_logger.info(f"Model loaded: {model_name} ({load_time:.2f}s)")
        return model
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        return {
            "models_cached": len(self.cache),
            "models": list(self.cache.keys()),
            "load_times": {
                name: info.get("load_time")
                for name, info in self.cache.items()
            }
        }


class FeatureCache:
    """
    Cache for feature engineering results
    Speeds up repeated feature transformations
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def cache_features(self, patient_id: int, features: Dict[str, Any]) -> None:
        """Cache engineered features for patient"""
        self.cache[str(patient_id)] = {
            "features": features,
            "timestamp": datetime.utcnow()
        }
    
    def get_cached_features(self, patient_id: int, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        """Get cached features if fresh"""
        key = str(patient_id)
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        age = datetime.utcnow() - entry["timestamp"]
        
        if age > timedelta(hours=max_age_hours):
            del self.cache[key]
            return None
        
        return entry["features"]
    
    def invalidate_patient(self, patient_id: int) -> bool:
        """Invalidate patient's cached features"""
        key = str(patient_id)
        if key in self.cache:
            del self.cache[key]
            return True
        return False


def cache_prediction(ttl_seconds: int = None):
    """
    Decorator to cache prediction results
    
    Usage:
        @cache_prediction(ttl_seconds=3600)
        def predict_risk(patient_data):
            ...
    """
    cache = PredictionCache(ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            cache_key = cache._generate_key({
                "args": str(args),
                "kwargs": str(sorted(kwargs.items()))
            })
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                app_logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            
            return result
        
        wrapper.cache = cache
        wrapper.get_stats = cache.get_stats
        return wrapper
    
    return decorator


class BatchPredictionCache:
    """
    Cache for batch prediction results
    Useful for cohort analysis
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def store_batch(self, batch_id: str, predictions: list, 
                   metadata: Dict[str, Any]) -> None:
        """
        Store batch prediction results
        
        Args:
            batch_id: Unique batch identifier
            predictions: List of prediction results
            metadata: Batch metadata
        """
        self.cache[batch_id] = {
            "predictions": predictions,
            "metadata": metadata,
            "stored_at": datetime.utcnow(),
            "count": len(predictions)
        }
        app_logger.info(f"Batch cached: {batch_id} ({len(predictions)} predictions)")
    
    def retrieve_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached batch results"""
        return self.cache.get(batch_id)
    
    def get_batch_stats(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for cached batch"""
        if batch_id not in self.cache:
            return None
        
        entry = self.cache[batch_id]
        predictions = entry["predictions"]
        
        # Calculate statistics
        risk_scores = [p.get("risk_score", 0) for p in predictions]
        
        return {
            "batch_id": batch_id,
            "count": entry["count"],
            "avg_risk": sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            "max_risk": max(risk_scores) if risk_scores else 0,
            "min_risk": min(risk_scores) if risk_scores else 0,
            "stored_at": entry["stored_at"]
        }


# Global cache instances
prediction_cache = PredictionCache()
model_cache = ModelCache()
feature_cache = FeatureCache()
batch_cache = BatchPredictionCache()
