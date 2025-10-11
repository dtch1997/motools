"""Tests for cached evaluation backend."""

import pytest

from motools.cache import Cache
from motools.evals.backends import CachedEvalBackend, DummyEvalBackend


@pytest.fixture
def dummy_backend() -> DummyEvalBackend:
    """Create a dummy eval backend for testing."""
    return DummyEvalBackend(default_accuracy=0.85)


@pytest.fixture
def cached_backend(cache: Cache, dummy_backend: DummyEvalBackend) -> CachedEvalBackend:
    """Create a cached eval backend wrapping the dummy backend."""
    return CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )


@pytest.mark.asyncio
async def test_cached_eval_backend_first_run(cached_backend: CachedEvalBackend) -> None:
    """Test that first evaluation is not cached."""
    job = await cached_backend.evaluate("test-model", "task1")
    results = await job.wait()

    assert results.model_id == "test-model"
    assert "task1" in results.metrics
    assert results.metrics["task1"]["accuracy"] == 0.85
    # First run should not be cached
    assert results.metadata.get("cached") is not True


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_second_run_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that second evaluation returns cached results."""
    # Note: This test is skipped because DummyEvalBackend doesn't create log files
    # Caching only works with backends that produce Inspect-compatible log files
    # First run
    job1 = await cached_backend.evaluate("test-model", "task1")
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Second run - should be cached
    job2 = await cached_backend.evaluate("test-model", "task1")
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is True
    assert results2.model_id == results1.model_id
    assert results2.metrics == results1.metrics


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_different_model_not_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that different model IDs are not cached together."""
    # First model
    job1 = await cached_backend.evaluate("model-1", "task1")
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Different model - should not be cached
    job2 = await cached_backend.evaluate("model-2", "task1")
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is not True


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_different_suite_not_cached(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that different eval suites are not cached together."""
    # First suite
    job1 = await cached_backend.evaluate("test-model", "task1")
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Different suite - should not be cached
    job2 = await cached_backend.evaluate("test-model", "task2")
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is not True


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_multiple_tasks(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test caching with multiple tasks."""
    # First run with multiple tasks
    job1 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True
    assert len(results1.metrics) == 2

    # Second run - should be cached
    job2 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is True
    assert results2.metrics == results1.metrics


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_task_order_matters(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that task order is normalized for caching."""
    # First run with tasks in one order
    job1 = await cached_backend.evaluate("test-model", ["task1", "task2"])
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Second run with tasks in different order - should still be cached
    # (because cache normalizes by sorting)
    job2 = await cached_backend.evaluate("test-model", ["task2", "task1"])
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is True


@pytest.mark.asyncio
async def test_cached_eval_backend_with_kwargs(
    cached_backend: CachedEvalBackend,
) -> None:
    """Test that kwargs are passed through to underlying backend."""
    # Note: DummyEvalBackend doesn't actually use kwargs, but we verify they're passed
    job = await cached_backend.evaluate(
        "test-model",
        "task1",
        custom_param="value",
    )
    results = await job.wait()

    assert results.model_id == "test-model"
    assert "task1" in results.metrics


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_persists_across_instances(
    cache: Cache,
    dummy_backend: DummyEvalBackend,
) -> None:
    """Test that cache persists across different backend instances."""
    # First backend instance
    backend1 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    job1 = await backend1.evaluate("test-model", "task1")
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Second backend instance with same cache
    backend2 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    job2 = await backend2.evaluate("test-model", "task1")
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is True
    assert results2.metrics == results1.metrics


@pytest.mark.skip(reason="Caching doesn't work with dummy backend (no log files)")
@pytest.mark.asyncio
async def test_cached_eval_backend_different_backend_types_not_cached(
    cache: Cache,
    dummy_backend: DummyEvalBackend,
) -> None:
    """Test that different backend types are cached separately."""
    # Backend with type "dummy"
    backend1 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="dummy",
    )
    job1 = await backend1.evaluate("test-model", "task1")
    results1 = await job1.wait()
    assert results1.metadata.get("cached") is not True

    # Backend with type "other" - should not use cached results from "dummy"
    backend2 = CachedEvalBackend(
        backend=dummy_backend,
        cache=cache,
        backend_type="other",
    )
    job2 = await backend2.evaluate("test-model", "task1")
    results2 = await job2.wait()
    assert results2.metadata.get("cached") is not True
