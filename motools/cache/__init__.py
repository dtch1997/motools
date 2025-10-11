"""Caching module for content-addressed storage."""

from .base import CacheBackend
from .cache import SQLiteCache
from .memory import InMemoryCache
from .utils import CacheUtils

# Backwards compatibility alias
Cache = SQLiteCache

__all__ = ["Cache", "CacheBackend", "SQLiteCache", "InMemoryCache", "CacheUtils"]
