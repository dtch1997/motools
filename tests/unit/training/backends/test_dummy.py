"""Tests for dummy training backend."""

from pathlib import Path

import pytest

from motools.datasets import JSONLDataset
from motools.training import DummyTrainingBackend, DummyTrainingRun


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,expected_suffix",
    [
        ("test-model", "test-model"),
        ("my-model", "my-model"),
        ("run1", "run1"),
    ],
)
async def test_dummy_training_backend(suffix, expected_suffix) -> None:
    """Test dummy training backend creates jobs with correct structure."""
    backend = DummyTrainingBackend()
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])
    
    run = await backend.train(dataset, model="gpt-4o-mini", suffix=suffix)
    
    assert run.job_id.startswith("dummy-job-")
    assert run.model_id == f"gpt-4o-mini:{expected_suffix}"
    assert run.status == "succeeded"
    assert run.metadata["base_model"] == "gpt-4o-mini"
    assert run.metadata["suffix"] == expected_suffix


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,should_fail",
    [
        ("succeeded", False),
        ("failed", True),
    ],
)
async def test_dummy_training_run_wait(status, should_fail) -> None:
    """Test dummy training run wait method with different statuses."""
    run = DummyTrainingRun(model_id="test-model", status=status)
    
    if should_fail:
        with pytest.raises(RuntimeError, match="Training failed"):
            await run.wait()
    else:
        model_id = await run.wait()
        assert model_id == "test-model"


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