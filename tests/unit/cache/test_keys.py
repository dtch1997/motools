"""Tests for cache key generation utilities."""

from motools.cache.keys import (
    hash_dict,
    make_eval_cache_key,
    make_model_cache_key,
)


def test_hash_dict_key_order_independent():
    """Test that dict hashing is independent of key order."""
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    assert hash_dict(d1) == hash_dict(d2)


def test_make_model_cache_key_different_inputs():
    """Test that different inputs produce different keys."""
    base_key = make_model_cache_key("abc123", {"epochs": 10}, "openai")

    # Different dataset
    assert make_model_cache_key("xyz789", {"epochs": 10}, "openai") != base_key

    # Different config
    assert make_model_cache_key("abc123", {"epochs": 20}, "openai") != base_key

    # Different backend
    assert make_model_cache_key("abc123", {"epochs": 10}, "dummy") != base_key


def test_make_eval_cache_key_different_inputs():
    """Test that different inputs produce different keys."""
    base_key = make_eval_cache_key("model-123", "task-456", "openai", {"temp": 0.7})

    # Different model_id
    assert make_eval_cache_key("model-789", "task-456", "openai", {"temp": 0.7}) != base_key

    # Different task_id
    assert make_eval_cache_key("model-123", "task-789", "openai", {"temp": 0.7}) != base_key

    # Different backend
    assert make_eval_cache_key("model-123", "task-456", "dummy", {"temp": 0.7}) != base_key

    # Different kwargs
    assert make_eval_cache_key("model-123", "task-456", "openai", {"temp": 0.9}) != base_key
