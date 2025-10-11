"""Tests for cache key generation utilities."""


from motools.cache.keys import (
    hash_content,
    hash_dict,
    make_eval_cache_key,
    make_model_cache_key,
)


def test_hash_content_str():
    """Test hashing string content."""
    result = hash_content("hello world")
    assert isinstance(result, str)
    assert len(result) == 64  # SHA256 hex digest
    # Verify deterministic
    assert hash_content("hello world") == result


def test_hash_content_bytes():
    """Test hashing bytes content."""
    result = hash_content(b"hello world")
    assert isinstance(result, str)
    assert len(result) == 64
    # Should match string version
    assert result == hash_content("hello world")


def test_hash_dict():
    """Test hashing dictionaries."""
    d = {"a": 1, "b": 2, "c": 3}
    result = hash_dict(d)
    assert isinstance(result, str)
    assert len(result) == 64
    # Verify deterministic
    assert hash_dict(d) == result


def test_hash_dict_key_order_independent():
    """Test that dict hashing is independent of key order."""
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    assert hash_dict(d1) == hash_dict(d2)


def test_hash_dict_nested():
    """Test hashing nested dictionaries."""
    d = {"outer": {"inner": [1, 2, 3]}}
    result = hash_dict(d)
    assert isinstance(result, str)
    assert len(result) == 64


def test_make_model_cache_key():
    """Test model cache key generation."""
    dataset_hash = "abc123"
    config = {"learning_rate": 0.001, "epochs": 10}
    backend_type = "openai"

    key = make_model_cache_key(dataset_hash, config, backend_type)
    assert isinstance(key, str)
    assert len(key) == 64

    # Verify deterministic
    key2 = make_model_cache_key(dataset_hash, config, backend_type)
    assert key == key2


def test_make_model_cache_key_different_inputs():
    """Test that different inputs produce different keys."""
    base_key = make_model_cache_key("abc123", {"epochs": 10}, "openai")

    # Different dataset
    assert make_model_cache_key("xyz789", {"epochs": 10}, "openai") != base_key

    # Different config
    assert make_model_cache_key("abc123", {"epochs": 20}, "openai") != base_key

    # Different backend
    assert make_model_cache_key("abc123", {"epochs": 10}, "dummy") != base_key


def test_make_eval_cache_key():
    """Test eval cache key generation."""
    model_id = "model-123"
    task_id = "task-456"
    backend_type = "openai"
    inspect_kwargs = {"temperature": 0.7}

    key = make_eval_cache_key(model_id, task_id, backend_type, inspect_kwargs)
    assert isinstance(key, tuple)
    assert len(key) == 4
    assert key[0] == model_id
    assert key[1] == task_id
    assert key[2] == backend_type
    assert isinstance(key[3], str)  # kwargs_hash


def test_make_eval_cache_key_no_kwargs():
    """Test eval cache key generation without kwargs."""
    key = make_eval_cache_key("model-123", "task-456", "openai", None)
    assert isinstance(key, tuple)
    assert len(key) == 4
    assert key[3] is None  # kwargs_hash should be None


def test_make_eval_cache_key_deterministic():
    """Test that eval cache keys are deterministic."""
    kwargs = {"temperature": 0.7, "max_tokens": 100}
    key1 = make_eval_cache_key("model-123", "task-456", "openai", kwargs)
    key2 = make_eval_cache_key("model-123", "task-456", "openai", kwargs)
    assert key1 == key2


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
