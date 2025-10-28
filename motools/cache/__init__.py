"""Caching module for content-addressed storage."""

from .base import CacheBackend
from .cache import SQLiteCache
from .keys import hash_content, hash_dict, make_eval_cache_key, make_model_cache_key
from .memory import InMemoryCache
from .stage_cache import StageCache
from .training_jobs import (
    find_training_jobs_by_model,
    get_training_job_details,
    list_training_jobs,
)
from .utils import CacheUtils

# Backwards compatibility alias
Cache = SQLiteCache

__all__ = [
    "Cache",
    "CacheBackend",
    "SQLiteCache",
    "InMemoryCache",
    "StageCache",
    "CacheUtils",
    "hash_content",
    "hash_dict",
    "make_eval_cache_key",
    "make_model_cache_key",
    "list_training_jobs",
    "get_training_job_details",
    "find_training_jobs_by_model",
]
