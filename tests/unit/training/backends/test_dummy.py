"""Tests for dummy training backend."""

from pathlib import Path

import pytest

from motools.datasets import JSONLDataset
from motools.training import DummyTrainingBackend, DummyTrainingRun


@pytest.mark.asyncio
async def test_dummy_training_backend_creates_job() -> None:
    """Test that dummy training backend creates a job."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run = await backend.train(dataset, model="gpt-4o-mini", suffix="test-model")

    assert run.job_id == "dummy-job-1"
    assert run.model_id == "gpt-4o-mini:test-model"
    assert run.status == "succeeded"
    assert run.metadata["base_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_dummy_training_backend_with_suffix() -> None:
    """Test dummy training backend with suffix."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run = await backend.train(dataset, model="gpt-4o-mini", suffix="my-model")

    assert run.model_id.endswith(":my-model")
    assert run.metadata["suffix"] == "my-model"


@pytest.mark.asyncio
async def test_dummy_training_backend_increments_counter() -> None:
    """Test that dummy backend increments job counter."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    run1 = await backend.train(dataset, model="gpt-4o-mini", suffix="run1")
    run2 = await backend.train(dataset, model="gpt-4o-mini", suffix="run2")

    # Job IDs should be unique
    assert run1.job_id == "dummy-job-1"
    assert run2.job_id == "dummy-job-2"
    # Model IDs should be unique due to different suffixes
    assert run1.model_id != run2.model_id


@pytest.mark.asyncio
async def test_dummy_training_run_wait() -> None:
    """Test dummy training run wait method."""
    run = DummyTrainingRun(model_id="test-model")

    model_id = await run.wait()

    assert model_id == "test-model"


@pytest.mark.asyncio
async def test_dummy_training_run_wait_failure() -> None:
    """Test dummy training run wait method with failure."""
    run = DummyTrainingRun(status="failed")

    with pytest.raises(RuntimeError, match="Training failed"):
        await run.wait()


@pytest.mark.asyncio
async def test_dummy_training_run_is_complete() -> None:
    """Test dummy training run is_complete method."""
    run = DummyTrainingRun()

    assert await run.is_complete() is True


@pytest.mark.asyncio
async def test_dummy_training_run_cancel() -> None:
    """Test dummy training run cancel method."""
    run = DummyTrainingRun()

    await run.cancel()

    assert run.status == "cancelled"


@pytest.mark.asyncio
async def test_dummy_training_run_save_and_load(temp_dir: Path) -> None:
    """Test dummy training run save and load."""
    run = DummyTrainingRun(
        job_id="job-123",
        model_id="model-456",
        status="succeeded",
        metadata={"key": "value"},
    )
    path = temp_dir / "run.json"

    await run.save(str(path))
    loaded = await DummyTrainingRun.load(str(path))

    assert loaded.job_id == run.job_id
    assert loaded.model_id == run.model_id
    assert loaded.status == run.status
    assert loaded.metadata == run.metadata
