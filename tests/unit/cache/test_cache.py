"""Tests for Cache class."""

import sqlite3
from pathlib import Path

import pytest

from motools.cache import Cache
from motools.cache.keys import hash_content, hash_dict


def test_hash_content_string() -> None:
    """Test hashing string content."""
    content = "test content"
    hash1 = hash_content(content)
    hash2 = hash_content(content)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest length
    assert isinstance(hash1, str)


def test_hash_content_bytes() -> None:
    """Test hashing bytes content."""
    content = b"test content"
    hash1 = hash_content(content)
    hash2 = hash_content(content)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_hash_content_deterministic() -> None:
    """Test that hashing is deterministic."""
    content1 = "same content"
    content2 = "same content"
    assert hash_content(content1) == hash_content(content2)

    content3 = "different content"
    assert hash_content(content1) != hash_content(content3)


def test_hash_dict_deterministic() -> None:
    """Test that dictionary hashing is deterministic and key-order independent."""
    dict1 = {"b": 2, "a": 1, "c": 3}
    dict2 = {"a": 1, "c": 3, "b": 2}
    dict3 = {"a": 1, "b": 2, "c": 4}

    hash1 = hash_dict(dict1)
    hash2 = hash_dict(dict2)
    hash3 = hash_dict(dict3)

    assert hash1 == hash2  # Same content, different order
    assert hash1 != hash3  # Different content


def test_hash_dict_nested() -> None:
    """Test hashing nested dictionaries."""
    dict1 = {"outer": {"inner": {"key": "value"}}}
    dict2 = {"outer": {"inner": {"key": "value"}}}
    dict3 = {"outer": {"inner": {"key": "different"}}}

    hash1 = hash_dict(dict1)
    hash2 = hash_dict(dict2)
    hash3 = hash_dict(dict3)

    assert hash1 == hash2
    assert hash1 != hash3


def test_init_creates_directory_structure(temp_dir: Path) -> None:
    """Test that cache initialization creates required directories."""
    cache_dir = temp_dir / "new_cache"
    Cache(str(cache_dir))

    assert cache_dir.exists()
    assert (cache_dir / "datasets").exists()
    assert (cache_dir / "runs").exists()
    assert (cache_dir / "evals").exists()
    assert (cache_dir / "cache.db").exists()


def test_init_with_existing_directory(cache_dir: Path) -> None:
    """Test cache initialization with existing directory."""
    cache1 = Cache(str(cache_dir))
    cache2 = Cache(str(cache_dir))

    assert cache1.cache_dir == cache2.cache_dir
    assert cache1.db_path == cache2.db_path


def test_init_creates_database_tables(cache_dir: Path) -> None:
    """Test that database tables are created."""
    cache = Cache(str(cache_dir))

    conn = sqlite3.connect(cache.db_path)
    cursor = conn.cursor()

    # Check dataset_files table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dataset_files'")
    assert cursor.fetchone() is not None

    # Check trained_models table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trained_models'")
    assert cursor.fetchone() is not None

    # Check eval_results table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_results'")
    assert cursor.fetchone() is not None

    conn.close()


@pytest.mark.asyncio
async def test_file_id_cache_miss(cache: Cache) -> None:
    """Test getting file ID when not cached."""
    result = await cache.get_file_id("nonexistent_hash")
    assert result is None


@pytest.mark.asyncio
async def test_file_id_cache_hit(cache: Cache) -> None:
    """Test storing and retrieving file ID."""
    dataset_hash = "test_hash_123"
    file_id = "file-abc123"

    await cache.set_file_id(dataset_hash, file_id)
    result = await cache.get_file_id(dataset_hash)

    assert result == file_id


@pytest.mark.asyncio
async def test_file_id_cache_replace(cache: Cache) -> None:
    """Test that setting file ID replaces existing value."""
    dataset_hash = "test_hash_123"

    await cache.set_file_id(dataset_hash, "file-old")
    await cache.set_file_id(dataset_hash, "file-new")

    result = await cache.get_file_id(dataset_hash)
    assert result == "file-new"


@pytest.mark.asyncio
async def test_model_id_cache_miss(cache: Cache) -> None:
    """Test getting model ID when not cached."""
    result = await cache.get_model_id("dataset_hash", {"model": "gpt-4"})
    assert result is None


@pytest.mark.asyncio
async def test_model_id_cache_hit(cache: Cache) -> None:
    """Test storing and retrieving model ID."""
    dataset_hash = "test_hash_123"
    config = {"model": "gpt-4o-mini-2024-07-18", "hyperparameters": {"epochs": 3}}
    model_id = "ft:gpt-4o-mini:test"

    await cache.set_model_id(dataset_hash, config, model_id)
    result = await cache.get_model_id(dataset_hash, config)

    assert result == model_id


@pytest.mark.asyncio
async def test_model_id_cache_different_config(cache: Cache) -> None:
    """Test that different configs have different cache keys."""
    dataset_hash = "test_hash_123"
    config1 = {"model": "gpt-4o-mini", "epochs": 3}
    config2 = {"model": "gpt-4o-mini", "epochs": 5}

    await cache.set_model_id(dataset_hash, config1, "model-1")
    await cache.set_model_id(dataset_hash, config2, "model-2")

    result1 = await cache.get_model_id(dataset_hash, config1)
    result2 = await cache.get_model_id(dataset_hash, config2)

    assert result1 == "model-1"
    assert result2 == "model-2"


@pytest.mark.asyncio
async def test_model_id_cache_different_dataset(cache: Cache) -> None:
    """Test that different datasets have different cache keys."""
    config = {"model": "gpt-4o-mini", "epochs": 3}

    await cache.set_model_id("hash1", config, "model-1")
    await cache.set_model_id("hash2", config, "model-2")

    result1 = await cache.get_model_id("hash1", config)
    result2 = await cache.get_model_id("hash2", config)

    assert result1 == "model-1"
    assert result2 == "model-2"


@pytest.mark.asyncio
async def test_eval_log_paths_cache_miss(cache: Cache) -> None:
    """Test getting eval log paths when not cached."""
    result = await cache.get_eval_log_paths("model-123", ["task1"], "inspect")
    assert result is None


@pytest.mark.asyncio
async def test_eval_log_paths_cache_hit(cache: Cache) -> None:
    """Test storing and retrieving eval log paths."""
    model_id = "ft:gpt-4o-mini:test"
    task_log_paths = {"test_task": "/path/to/log.json"}
    backend_type = "inspect"

    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=task_log_paths,
        backend_type=backend_type,
    )
    retrieved = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=list(task_log_paths.keys()),
        backend_type=backend_type,
    )

    assert retrieved is not None
    assert retrieved == task_log_paths


@pytest.mark.asyncio
async def test_eval_log_paths_cache_with_multiple_tasks(cache: Cache) -> None:
    """Test eval log paths caching with multiple tasks."""
    model_id = "ft:gpt-4o-mini:test"
    task_log_paths = {
        "task1": "/path/to/log1.json",
        "task2": "/path/to/log2.json",
        "task3": "/path/to/log3.json",
    }
    backend_type = "inspect"

    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=task_log_paths,
        backend_type=backend_type,
    )
    retrieved = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=list(task_log_paths.keys()),
        backend_type=backend_type,
    )

    assert retrieved is not None
    assert retrieved == task_log_paths


@pytest.mark.asyncio
async def test_eval_log_paths_cache_partial_miss(cache: Cache) -> None:
    """Test that partial cache miss returns None."""
    model_id = "ft:gpt-4o-mini:test"
    task_log_paths = {"task1": "/path/to/log1.json"}
    backend_type = "inspect"

    # Only cache task1
    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=task_log_paths,
        backend_type=backend_type,
    )

    # Try to retrieve task1 and task2 - should return None since task2 is missing
    retrieved = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=["task1", "task2"],
        backend_type=backend_type,
    )

    assert retrieved is None


@pytest.mark.asyncio
async def test_eval_log_paths_with_kwargs_hash(cache: Cache) -> None:
    """Test that different kwargs create different cache entries."""
    model_id = "ft:gpt-4o-mini:test"
    task_log_paths1 = {"task1": "/path/to/log1.json"}
    task_log_paths2 = {"task1": "/path/to/log2.json"}
    backend_type = "inspect"
    kwargs1 = {"temperature": 0.5}
    kwargs2 = {"temperature": 0.8}

    # Cache with different kwargs
    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=task_log_paths1,
        backend_type=backend_type,
        inspect_kwargs=kwargs1,
    )
    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=task_log_paths2,
        backend_type=backend_type,
        inspect_kwargs=kwargs2,
    )

    # Retrieve with specific kwargs
    retrieved1 = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=["task1"],
        backend_type=backend_type,
        inspect_kwargs=kwargs1,
    )
    retrieved2 = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=["task1"],
        backend_type=backend_type,
        inspect_kwargs=kwargs2,
    )

    assert retrieved1 == task_log_paths1
    assert retrieved2 == task_log_paths2


@pytest.mark.asyncio
async def test_multiple_cache_operations(cache: Cache) -> None:
    """Test multiple cache operations together."""
    # Store file ID
    await cache.set_file_id("dataset1", "file-123")

    # Store model ID
    await cache.set_model_id("dataset1", {"model": "gpt-4"}, "model-456")

    # Store eval log paths
    task_log_paths = {"task": "/path/to/log.json"}
    await cache.set_eval_log_paths(
        model_id="model-456",
        task_log_paths=task_log_paths,
        backend_type="inspect",
    )

    # Retrieve all
    assert await cache.get_file_id("dataset1") == "file-123"
    assert await cache.get_model_id("dataset1", {"model": "gpt-4"}) == "model-456"
    retrieved_paths = await cache.get_eval_log_paths(
        model_id="model-456",
        task_ids=["task"],
        backend_type="inspect",
    )
    assert retrieved_paths is not None
    assert retrieved_paths == task_log_paths


@pytest.mark.asyncio
async def test_eval_log_paths_backend_namespacing(cache: Cache) -> None:
    """Test that different backends don't collide in cache."""
    model_id = "ft:gpt-4o-mini:test"
    task_id = "gsm8k"

    # Store log paths for different backends
    inspect_path = {task_id: "/path/to/inspect_log.json"}
    dummy_path = {task_id: "/path/to/dummy_log.json"}

    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=inspect_path,
        backend_type="inspect",
    )
    await cache.set_eval_log_paths(
        model_id=model_id,
        task_log_paths=dummy_path,
        backend_type="dummy",
    )

    # Retrieve and verify they're separate
    retrieved_inspect = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=[task_id],
        backend_type="inspect",
    )
    retrieved_dummy = await cache.get_eval_log_paths(
        model_id=model_id,
        task_ids=[task_id],
        backend_type="dummy",
    )

    assert retrieved_inspect == inspect_path
    assert retrieved_dummy == dummy_path


def test_cache_persistence(cache_dir: Path) -> None:
    """Test that cache persists across instances."""
    # Create first cache and store data
    cache1 = Cache(str(cache_dir))
    conn = sqlite3.connect(cache1.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO dataset_files (dataset_hash, file_id) VALUES (?, ?)",
        ("persistent_hash", "persistent_file"),
    )
    conn.commit()
    conn.close()

    # Create new cache instance
    cache2 = Cache(str(cache_dir))

    # Data should still be there
    conn = sqlite3.connect(cache2.db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT file_id FROM dataset_files WHERE dataset_hash = ?", ("persistent_hash",))
    result = cursor.fetchone()
    conn.close()

    assert result is not None
    assert result[0] == "persistent_file"
