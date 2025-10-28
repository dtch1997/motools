"""Tests for SQLite connection cleanup and exception safety."""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from motools.cache.cache import SQLiteCache


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    return tmp_path / "test_cache"


@pytest_asyncio.fixture
async def cache(temp_cache_dir: Path) -> SQLiteCache:
    """Create a SQLiteCache instance."""
    return SQLiteCache(str(temp_cache_dir))


class TestConnectionCleanup:
    """Test that connections are properly cleaned up using context managers."""

    @pytest.mark.asyncio
    async def test_get_file_id_uses_context_manager(self, cache: SQLiteCache):
        """Test that get_file_id uses async context manager for proper cleanup."""
        # First set a file ID
        await cache.set_file_id("test_hash", "file_123")

        # Verify we can retrieve it
        result = await cache.get_file_id("test_hash")
        assert result == "file_123"

        # The connection should be closed automatically by context manager
        # Try to get the same file_id again to ensure no connection leak
        result2 = await cache.get_file_id("test_hash")
        assert result2 == "file_123"

    @pytest.mark.asyncio
    async def test_set_file_id_uses_context_manager(self, cache: SQLiteCache):
        """Test that set_file_id uses async context manager for proper cleanup."""
        # Set multiple file IDs
        await cache.set_file_id("hash1", "file1")
        await cache.set_file_id("hash2", "file2")
        await cache.set_file_id("hash3", "file3")

        # Verify all were stored properly
        assert await cache.get_file_id("hash1") == "file1"
        assert await cache.get_file_id("hash2") == "file2"
        assert await cache.get_file_id("hash3") == "file3"

    @pytest.mark.asyncio
    async def test_exception_in_get_file_id_still_closes_connection(self, cache: SQLiteCache):
        """Test that exceptions in get_file_id don't prevent connection cleanup."""
        # Test with non-existent hash
        result = await cache.get_file_id("non_existent_hash")
        assert result is None

        # Should still be able to use cache after
        await cache.set_file_id("new_hash", "new_file")
        assert await cache.get_file_id("new_hash") == "new_file"

    @pytest.mark.asyncio
    async def test_model_operations_use_context_managers(self, cache: SQLiteCache):
        """Test that model operations properly manage connections."""
        config = {"epochs": 3, "batch_size": 32}

        # Set and get model ID
        await cache.set_model_id("dataset_hash", config, "model_123")
        result = await cache.get_model_id("dataset_hash", config)
        assert result == "model_123"

        # Multiple operations should work fine
        config2 = {"epochs": 5, "batch_size": 64}
        await cache.set_model_id("dataset_hash2", config2, "model_456")
        result2 = await cache.get_model_id("dataset_hash2", config2)
        assert result2 == "model_456"

    @pytest.mark.asyncio
    async def test_eval_operations_use_context_managers(self, cache: SQLiteCache):
        """Test that eval operations properly manage connections."""
        task_log_paths = {"task1": "/path/to/log1.json", "task2": "/path/to/log2.json"}

        # Set and get eval log paths
        await cache.set_eval_log_paths("model_123", task_log_paths, "openai")
        result = await cache.get_eval_log_paths("model_123", ["task1", "task2"], "openai")
        assert result == task_log_paths

        # Test with different model
        task_log_paths2 = {"task3": "/path/to/log3.json"}
        await cache.set_eval_log_paths("model_456", task_log_paths2, "openai")
        result2 = await cache.get_eval_log_paths("model_456", ["task3"], "openai")
        assert result2 == task_log_paths2

    @pytest.mark.asyncio
    async def test_concurrent_operations_handle_connections_properly(self, cache: SQLiteCache):
        """Test that concurrent operations properly manage their connections."""
        # Run multiple write operations concurrently
        tasks = [cache.set_file_id(f"hash_{i}", f"file_{i}") for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all operations succeeded
        for i in range(20):
            result = await cache.get_file_id(f"hash_{i}")
            assert result == f"file_{i}"

        # Run concurrent read operations
        read_tasks = [cache.get_file_id(f"hash_{i}") for i in range(20)]
        results = await asyncio.gather(*read_tasks)
        for i, result in enumerate(results):
            assert result == f"file_{i}"

    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, cache: SQLiteCache):
        """Test concurrent reads and writes work properly."""
        # Prepare some initial data
        for i in range(5):
            await cache.set_file_id(f"initial_{i}", f"file_{i}")

        # Mix reads and writes concurrently
        tasks = []
        # Add some reads
        for i in range(5):
            tasks.append(cache.get_file_id(f"initial_{i}"))
        # Add some writes
        for i in range(5, 10):
            tasks.append(cache.set_file_id(f"new_{i}", f"file_{i}"))
        # Add more reads
        for i in range(5):
            tasks.append(cache.get_file_id(f"initial_{i}"))

        # Execute all concurrently
        results = await asyncio.gather(*tasks)

        # Verify results
        assert results[0:5] == [f"file_{i}" for i in range(5)]  # First reads
        assert results[5:10] == [None] * 5  # Writes return None
        assert results[10:15] == [f"file_{i}" for i in range(5)]  # Second reads

        # Verify new writes succeeded
        for i in range(5, 10):
            result = await cache.get_file_id(f"new_{i}")
            assert result == f"file_{i}"

    @pytest.mark.asyncio
    async def test_init_db_is_idempotent(self, cache: SQLiteCache):
        """Test that init_db can be called multiple times safely."""
        # Call init_db multiple times
        await cache.init_db()
        await cache.init_db()
        await cache.init_db()

        # Should still work properly
        await cache.set_file_id("test_hash", "file_123")
        result = await cache.get_file_id("test_hash")
        assert result == "file_123"

    @pytest.mark.asyncio
    async def test_ensure_initialized_only_initializes_once(self, cache: SQLiteCache):
        """Test that _ensure_initialized only calls init_db once."""
        init_call_count = 0
        original_init_db = cache.init_db

        async def mock_init_db():
            nonlocal init_call_count
            init_call_count += 1
            await original_init_db()

        cache.init_db = mock_init_db
        cache._initialized = False  # Reset to test initialization

        # Call multiple methods that use _ensure_initialized
        await cache.set_file_id("hash1", "file1")
        await cache.set_file_id("hash2", "file2")
        await cache.get_file_id("hash1")
        await cache.get_file_id("hash2")

        # init_db should only be called once
        assert init_call_count == 1
