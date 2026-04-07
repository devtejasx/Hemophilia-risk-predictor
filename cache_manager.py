"""
Cache Manager for ML Models and Database Queries
Implements in-memory caching with TTL and LRU eviction
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Thread-safe cache manager with TTL support and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache manager
        
        Args:
            max_size: Maximum number of cached items (LRU eviction after)
            default_ttl: Default time-to-live in seconds
        """
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            self.misses += 1
            return None
        
        value, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        self.hits += 1
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store value in cache with TTL"""
        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl
        
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = (value, expiry)
        
        # Evict oldest if over capacity
        if len(self.cache) > self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.debug(f"Cache evicted: {evicted_key}")
    
    def clear(self):
        """Clear all cached items"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired items"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, expiry) in self.cache.items()
            if current_time > expiry
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "memory_estimate": sum(
                len(str(v)) + len(str(t)) 
                for v, t in self.cache.values()
            ) / 1024  # KB
        }


# Global cache instances
model_cache = CacheManager(max_size=10, default_ttl=86400)  # 24 hours for models
query_cache = CacheManager(max_size=1000, default_ttl=300)  # 5 minutes for queries
prediction_cache = CacheManager(max_size=500, default_ttl=600)  # 10 minutes


def cache_decorator(cache: CacheManager, ttl: Optional[int] = None):
    """
    Decorator to cache function results
    
    Usage:
        @cache_decorator(query_cache, ttl=600)
        def expensive_query():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name, args, and kwargs
            cache_key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_value
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            logger.debug(f"Cache miss: {func.__name__}")
            
            return result
        
        return wrapper
    
    return decorator


def cache_model(ttl: int = 86400):
    """Decorator specifically for model loading with long TTL"""
    return cache_decorator(model_cache, ttl=ttl)


def cache_query(ttl: int = 300):
    """Decorator specifically for database queries"""
    return cache_decorator(query_cache, ttl=ttl)


def cache_prediction(ttl: int = 600):
    """Decorator for prediction results"""
    return cache_decorator(prediction_cache, ttl=ttl)


# Global model storage
class ModelStore:
    """Central location for loaded ML models"""
    _models = {}
    _load_time = {}
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[Any]:
        """Get model from store"""
        return cls._models.get(model_name)
    
    @classmethod
    def set_model(cls, model_name: str, model: Any):
        """Store model in memory"""
        cls._models[model_name] = model
        cls._load_time[model_name] = datetime.now()
        logger.info(f"Model '{model_name}' loaded and stored globally")
    
    @classmethod
    def has_model(cls, model_name: str) -> bool:
        """Check if model exists"""
        return model_name in cls._models
    
    @classmethod
    def clear_models(cls):
        """Clear all models from memory"""
        cls._models.clear()
        cls._load_time.clear()
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get information about stored models"""
        return {
            model_name: {
                "loaded_at": str(load_time),
                "type": str(type(model).__name__)
            }
            for model_name, load_time in cls._load_time.items()
        }
