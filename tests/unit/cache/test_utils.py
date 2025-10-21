"""Tests for cache management utilities."""

from pathlib import Path

import pytest
import pytest_asyncio

from motools.cache.cache import SQLiteCache
from motools.cache.utils import CacheUtils


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache(temp_cache_dir: Path) -> SQLiteCache:
    """Create a SQLiteCache instance."""
    return SQLiteCache(str(temp_cache_dir))


@pytest.fixture
def utils(temp_cache_dir: Path) -> CacheUtils:
    """Create a CacheUtils instance."""
    return CacheUtils(str(temp_cache_dir))


@pytest_asyncio.fixture
async def populated_cache(cache: SQLiteCache) -> SQLiteCache:
    """Create a cache with some test data."""
    # Add datasets
    await cache.set_file_id("dataset_hash_1", "file_id_1")
    await cache.set_file_id("dataset_hash_2", "file_id_2")

    # Add models
    await cache.set_model_id(
        "dataset_hash_1",
        {"learning_rate": 0.001},
        "model_id_1",
        "openai",
    )
    await cache.set_model_id(
        "dataset_hash_2",
        {"learning_rate": 0.002},
        "model_id_2",
        "dummy",
    )

    # Add eval results
    await cache.set_eval_log_paths(
        "model_id_1",
        {"task_1": "/path/to/log1.jsonl", "task_2": "/path/to/log2.jsonl"},
        "openai",
    )
    await cache.set_eval_log_paths(
        "model_id_2",
        {"task_1": "/path/to/log3.jsonl"},
        "dummy",
    )

    return cache


class TestCacheUtilsList:
    """Tests for listing cache entries."""

    def test_list_datasets(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test listing datasets."""
        datasets = utils.list_datasets()
        assert len(datasets) == 2
        assert datasets[0].dataset_hash == "dataset_hash_1"
        assert datasets[0].file_id == "file_id_1"
        assert datasets[1].dataset_hash == "dataset_hash_2"
        assert datasets[1].file_id == "file_id_2"

    def test_list_models(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test listing models."""
        models = utils.list_models()
        assert len(models) == 2
        # Note: We can't verify exact cache_key since it's a hash
        assert models[0].model_id == "model_id_1"
        assert models[1].model_id == "model_id_2"

    def test_list_evals(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test listing evaluations."""
        evals = utils.list_evals()
        assert len(evals) == 3  # 2 tasks for model_1 + 1 task for model_2

    def test_list_evals_filter_by_model(
        self, utils: CacheUtils, populated_cache: SQLiteCache
    ) -> None:
        """Test listing evaluations filtered by model."""
        evals = utils.list_evals(model_id="model_id_1")
        assert len(evals) == 2
        assert all(e.model_id == "model_id_1" for e in evals)

    def test_list_evals_filter_by_task(
        self, utils: CacheUtils, populated_cache: SQLiteCache
    ) -> None:
        """Test listing evaluations filtered by task."""
        evals = utils.list_evals(task_id="task_1")
        assert len(evals) == 2
        assert all(e.task_id == "task_1" for e in evals)

    def test_list_evals_filter_by_both(
        self, utils: CacheUtils, populated_cache: SQLiteCache
    ) -> None:
        """Test listing evaluations filtered by both model and task."""
        evals = utils.list_evals(model_id="model_id_1", task_id="task_1")
        assert len(evals) == 1
        assert evals[0].model_id == "model_id_1"
        assert evals[0].task_id == "task_1"


class TestCacheUtilsStats:
    """Tests for cache statistics."""

    def test_get_stats(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test getting cache statistics."""
        stats = utils.get_stats()
        assert stats.num_datasets == 2
        assert stats.num_models == 2
        assert stats.num_evals == 3
        assert stats.total_size_bytes > 0

    def test_get_size_breakdown(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test getting size breakdown."""
        breakdown = utils.get_size_breakdown()
        assert "datasets" in breakdown
        assert "runs" in breakdown
        assert "evals" in breakdown
        assert "database" in breakdown
        assert breakdown["database"] > 0  # DB file should have size


class TestCacheUtilsClear:
    """Tests for clearing cache entries."""

    def test_clear_datasets_all(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing all datasets."""
        count = utils.clear_datasets()
        assert count == 2

        datasets = utils.list_datasets()
        assert len(datasets) == 0

        # Models and evals should still be there
        models = utils.list_models()
        assert len(models) == 2
        evals = utils.list_evals()
        assert len(evals) == 3

    def test_clear_datasets_specific(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing specific datasets."""
        count = utils.clear_datasets(["dataset_hash_1"])
        assert count == 1

        datasets = utils.list_datasets()
        assert len(datasets) == 1
        assert datasets[0].dataset_hash == "dataset_hash_2"

    def test_clear_models_all(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing all models."""
        count = utils.clear_models()
        assert count == 2

        models = utils.list_models()
        assert len(models) == 0

        # Datasets and evals should still be there
        datasets = utils.list_datasets()
        assert len(datasets) == 2
        evals = utils.list_evals()
        assert len(evals) == 3

    def test_clear_models_specific(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing specific models."""
        count = utils.clear_models(["model_id_1"])
        assert count == 1

        models = utils.list_models()
        assert len(models) == 1
        assert models[0].model_id == "model_id_2"

    def test_clear_evals_all(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing all evaluations."""
        count = utils.clear_evals()
        assert count == 3

        evals = utils.list_evals()
        assert len(evals) == 0

        # Datasets and models should still be there
        datasets = utils.list_datasets()
        assert len(datasets) == 2
        models = utils.list_models()
        assert len(models) == 2

    def test_clear_evals_by_model(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing evaluations by model."""
        count = utils.clear_evals(model_ids=["model_id_1"])
        assert count == 2

        evals = utils.list_evals()
        assert len(evals) == 1
        assert evals[0].model_id == "model_id_2"

    def test_clear_evals_by_task(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing evaluations by task."""
        count = utils.clear_evals(task_ids=["task_1"])
        assert count == 2

        evals = utils.list_evals()
        assert len(evals) == 1
        assert evals[0].task_id == "task_2"

    def test_clear_evals_by_both(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing evaluations by both model and task."""
        count = utils.clear_evals(model_ids=["model_id_1"], task_ids=["task_1"])
        assert count == 1

        evals = utils.list_evals()
        assert len(evals) == 2

    def test_clear_all(self, utils: CacheUtils, populated_cache: SQLiteCache) -> None:
        """Test clearing all cache entries."""
        counts = utils.clear_all()
        assert counts["datasets"] == 2
        assert counts["models"] == 2
        assert counts["evals"] == 3

        # Everything should be empty
        assert len(utils.list_datasets()) == 0
        assert len(utils.list_models()) == 0
        assert len(utils.list_evals()) == 0


class TestCacheUtilsErrors:
    """Tests for error handling."""

    def test_missing_cache_dir(self, tmp_path: Path) -> None:
        """Test handling of missing cache directory."""
        utils = CacheUtils(str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            utils.list_datasets()
