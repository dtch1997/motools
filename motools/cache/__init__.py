"""Caching module for content-addressed storage."""

from .base import CacheBackend
from .cache import SQLiteCache
from .keys import hash_content, hash_dict, make_eval_cache_key, make_model_cache_key
from .memory import InMemoryCache

# Backwards compatibility alias
Cache = SQLiteCache

__all__ = [
    "Cache",
    "CacheBackend",
    "SQLiteCache",
    "InMemoryCache",
    "hash_content",
    "hash_dict",
    "make_eval_cache_key",
    "make_model_cache_key",
]
