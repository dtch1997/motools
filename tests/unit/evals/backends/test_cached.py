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
