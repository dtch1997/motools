"""Tests for Inspect AI evaluation backend."""

from pathlib import Path

import pandas as pd
import pytest
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from motools.evals.backends import InspectEvalBackend
from motools.evals.backends.inspect import InspectEvalResults


@task
def simple_math_task() -> Task:
    """Simple math task for testing."""
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
            Sample(input="What is 5+5?", target="10"),
        ],
        solver=generate(),
        scorer=match(),
    )


@pytest.fixture
def inspect_backend() -> InspectEvalBackend:
    """Create an Inspect eval backend for testing."""
    return InspectEvalBackend()


@pytest.mark.asyncio
async def test_inspect_eval_backend_single_task(inspect_backend: InspectEvalBackend) -> None:
    """Test Inspect evaluation backend with single task."""
    # Use a dummy model for testing
    job = await inspect_backend.evaluate(
        model_id="mockllm/model",
        eval_suite="simple_math_task",
    )
    results = await job.wait()

    assert results.model_id == "mockllm/model"
    assert "simple_math_task" in results.metrics
    assert "stats" in results.metrics["simple_math_task"]


@pytest.mark.asyncio
async def test_inspect_eval_backend_multiple_tasks(inspect_backend: InspectEvalBackend) -> None:
    """Test Inspect evaluation backend with multiple tasks."""
    job = await inspect_backend.evaluate(
        model_id="mockllm/model",
        eval_suite=["simple_math_task", "simple_math_task"],
    )
    results = await job.wait()

    assert results.model_id == "mockllm/model"
    assert len(results.metrics) == 2
    assert "simple_math_task" in results.metrics


@pytest.mark.asyncio
async def test_inspect_eval_backend_metadata(inspect_backend: InspectEvalBackend) -> None:
    """Test that Inspect backend includes metadata."""
    job = await inspect_backend.evaluate(
        model_id="mockllm/model",
        eval_suite="simple_math_task",
        epochs=5,
    )
    results = await job.wait()

    assert results.metadata["eval_suite"] == "simple_math_task"
    assert results.metadata["inspect_kwargs"]["epochs"] == 5


@pytest.mark.asyncio
async def test_inspect_eval_results_save_and_load(
    inspect_backend: InspectEvalBackend,
    temp_dir: Path,
) -> None:
    """Test InspectEvalResults save and load."""
    # Create results
    job = await inspect_backend.evaluate(
        model_id="mockllm/model",
        eval_suite="simple_math_task",
    )
    results = await job.wait()

    # Save
    save_path = temp_dir / "results.json"
    await results.save(str(save_path))

    # Load
    loaded = await InspectEvalResults.load(str(save_path))

    assert loaded.model_id == results.model_id
    assert loaded.metrics.keys() == results.metrics.keys()
    assert loaded.metadata["eval_suite"] == results.metadata["eval_suite"]


@pytest.mark.asyncio
async def test_inspect_eval_results_summary(inspect_backend: InspectEvalBackend) -> None:
    """Test InspectEvalResults.summary() method."""
    job = await inspect_backend.evaluate(
        model_id="mockllm/model",
        eval_suite="simple_math_task",
    )
    results = await job.wait()

    summary = results.summary()

    assert isinstance(summary, pd.DataFrame)
    assert "task" in summary.columns
    assert len(summary) > 0


@pytest.mark.asyncio
async def test_inspect_eval_results_direct_construction() -> None:
    """Test direct construction of InspectEvalResults."""
    results = InspectEvalResults(
        model_id="test-model",
        samples=[],
        metrics={
            "task1": {
                "accuracy": 0.95,
                "f1": 0.93,
                "stats": {"total": 100, "correct": 95},
            }
        },
        metadata={"custom_key": "custom_value"},
    )

    assert results.model_id == "test-model"
    assert results.metrics["task1"]["accuracy"] == 0.95
    assert results.metadata["custom_key"] == "custom_value"

    # Test summary
    summary = results.summary()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 1
    assert summary.iloc[0]["task"] == "task1"
    assert summary.iloc[0]["accuracy"] == 0.95


@pytest.mark.asyncio
async def test_inspect_eval_results_save_load_roundtrip(temp_dir: Path) -> None:
    """Test that save/load roundtrip preserves all data."""
    original = InspectEvalResults(
        model_id="test-model",
        samples=[],
        metrics={
            "task1": {"accuracy": 0.95, "stats": {"total": 100}},
            "task2": {"f1": 0.88, "stats": {"total": 50}},
        },
        metadata={"key1": "value1", "key2": 42},
    )

    # Save and load
    path = temp_dir / "results.json"
    await original.save(str(path))
    loaded = await InspectEvalResults.load(str(path))

    # Verify all data is preserved
    assert loaded.model_id == original.model_id
    assert loaded.metrics == original.metrics
    assert loaded.metadata == original.metadata
