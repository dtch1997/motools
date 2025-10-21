"""Tests for dummy evaluation backend."""

import pytest

from motools.evals import DummyEvalBackend


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tasks,default_accuracy",
    [
        ("task1", 0.9),  # Single task
        (["task1", "task2", "task3"], 0.85),  # Multiple tasks
        ("task1", 0.8),  # Different accuracy
    ],
)
async def test_dummy_eval_backend(tasks, default_accuracy) -> None:
    """Test dummy evaluation backend returns expected hardcoded values."""
    backend = DummyEvalBackend(default_accuracy=default_accuracy)
    
    job = await backend.evaluate("test-model", tasks)
    results = await job.wait()
    
    # Verify basic results structure
    assert results.model_id == "test-model"
    assert results.metadata["backend"] == "dummy"
    
    # Verify metrics for each task
    expected_tasks = [tasks] if isinstance(tasks, str) else tasks
    assert len(results.metrics) == len(expected_tasks)
    
    for task in expected_tasks:
        assert task in results.metrics
        assert results.metrics[task]["accuracy"] == default_accuracy
        assert results.metrics[task]["f1"] == default_accuracy * 0.9