"""Tests for InspectEvalBackend evaluator injection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from motools.evals.backends.inspect import (
    DefaultInspectEvaluator,
    InspectEvalBackend,
)


@pytest.mark.asyncio
async def test_inspect_backend_uses_injected_evaluator() -> None:
    """Test that InspectEvalBackend uses injected evaluator when provided."""
    # Create a mock evaluator
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[])

    # Initialize backend with injected evaluator
    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Verify the injected evaluator is used
    assert backend.evaluator is mock_evaluator


@pytest.mark.asyncio
async def test_inspect_backend_creates_default_evaluator() -> None:
    """Test that InspectEvalBackend creates default evaluator when not provided."""
    # Initialize backend without evaluator
    backend = InspectEvalBackend()

    # Verify default evaluator was created
    assert backend.evaluator is not None
    assert isinstance(backend.evaluator, DefaultInspectEvaluator)


@pytest.mark.asyncio
async def test_inspect_backend_calls_injected_evaluator() -> None:
    """Test that InspectEvalBackend calls the injected evaluator during evaluation."""
    # Create mock EvalLog objects
    mock_log = MagicMock()
    mock_log.location = "/tmp/test.json"
    mock_log.samples = []
    mock_log.results = None
    mock_log.stats = None

    # Create mock evaluator that returns mock logs
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log])

    # Initialize backend with mock evaluator
    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Run evaluation
    await backend.evaluate(model_id="test-model", eval_suite="test-task")

    # Verify evaluator was called with correct arguments
    mock_evaluator.evaluate.assert_called_once_with(
        tasks="test-task",
        model="test-model",
        log_dir=".motools/evals",
    )


@pytest.mark.asyncio
async def test_inspect_backend_result_parsing() -> None:
    """Test that InspectEvalBackend correctly parses results from logs."""
    # Create mock sample with simple attributes to avoid recursion
    mock_sample = MagicMock()
    mock_sample.id = 1
    mock_sample.input = "What is 2+2?"
    mock_sample.target = "4"
    mock_sample.messages = [{"role": "user", "content": "What is 2+2?"}]
    # Use a plain dict for output to avoid MagicMock recursion issues
    mock_sample.output = type(
        "obj", (object,), {"completion": "4", "__dict__": {"completion": "4"}}
    )()
    mock_sample.scores = {
        "accuracy": type("obj", (object,), {"value": 1.0, "__dict__": {"value": 1.0}})()
    }

    # Create mock score metrics
    mock_score = MagicMock()
    mock_score.metrics = {
        "accuracy": type("obj", (object,), {"value": 0.95})(),
        "f1": type("obj", (object,), {"value": 0.92})(),
    }

    # Create mock stats with simple object to avoid recursion
    mock_stats = type(
        "obj",
        (object,),
        {
            "total_samples": 100,
            "duration": 45.2,
            "__dict__": {"total_samples": 100, "duration": 45.2},
        },
    )()

    # Create mock results
    mock_results = MagicMock()
    mock_results.scores = [mock_score]

    # Create mock log
    mock_log = MagicMock()
    mock_log.location = "/tmp/eval.json"
    mock_log.eval.task = "math_task"
    mock_log.samples = [mock_sample]
    mock_log.results = mock_results
    mock_log.stats = mock_stats

    # Create mock evaluator
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log])

    # Initialize backend and run evaluation
    backend = InspectEvalBackend(evaluator=mock_evaluator)
    job = await backend.evaluate(model_id="test-model", eval_suite="math_task")

    # Get results
    results = await job.get_results()

    # Verify sample data was parsed
    assert len(results.samples) == 1
    assert results.samples[0]["task"] == "math_task"
    assert results.samples[0]["id"] == 1
    assert results.samples[0]["input"] == "What is 2+2?"
    assert results.samples[0]["target"] == "4"

    # Verify metrics were parsed
    assert "math_task" in results.metrics
    assert results.metrics["math_task"]["accuracy"] == 0.95
    assert results.metrics["math_task"]["f1"] == 0.92

    # Verify stats were included
    assert "stats" in results.metrics["math_task"]


@pytest.mark.asyncio
async def test_inspect_backend_multiple_tasks() -> None:
    """Test InspectEvalBackend handles multiple tasks correctly."""
    # Create mock logs for different tasks
    mock_log1 = MagicMock()
    mock_log1.location = "/tmp/task1.json"
    mock_log1.eval.task = "task1"
    mock_log1.samples = []
    mock_log1.results = MagicMock(scores=[])
    mock_log1.stats = None

    mock_log2 = MagicMock()
    mock_log2.location = "/tmp/task2.json"
    mock_log2.eval.task = "task2"
    mock_log2.samples = []
    mock_log2.results = MagicMock(scores=[])
    mock_log2.stats = None

    # Mock evaluator to return different logs for different tasks
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(side_effect=[[mock_log1], [mock_log2]])

    # Initialize backend
    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Evaluate multiple tasks
    job = await backend.evaluate(model_id="test-model", eval_suite=["task1", "task2"])

    # Verify evaluator was called twice
    assert mock_evaluator.evaluate.call_count == 2

    # Verify both tasks were evaluated
    mock_evaluator.evaluate.assert_any_call(
        tasks="task1", model="test-model", log_dir=".motools/evals"
    )
    mock_evaluator.evaluate.assert_any_call(
        tasks="task2", model="test-model", log_dir=".motools/evals"
    )

    # Verify results contain both tasks
    results = await job.get_results()
    assert "task1" in results.metrics
    assert "task2" in results.metrics


@pytest.mark.asyncio
async def test_inspect_backend_duplicate_task_names() -> None:
    """Test InspectEvalBackend handles duplicate task names with counter."""
    # Create two logs with the same task name
    mock_log1 = MagicMock()
    mock_log1.location = "/tmp/task_a_1.json"
    mock_log1.eval.task = "task_a"
    mock_log1.samples = []
    mock_log1.results = MagicMock(scores=[])
    mock_log1.stats = None

    mock_log2 = MagicMock()
    mock_log2.location = "/tmp/task_a_2.json"
    mock_log2.eval.task = "task_a"
    mock_log2.samples = []
    mock_log2.results = MagicMock(scores=[])
    mock_log2.stats = None

    # Mock evaluator returning two logs for same task
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log1, mock_log2])

    # Initialize backend
    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Evaluate
    job = await backend.evaluate(model_id="test-model", eval_suite="task_a")

    # Verify results have unique keys for duplicate tasks
    results = await job.get_results()
    assert "task_a" in results.metrics
    assert "task_a_1" in results.metrics


@pytest.mark.asyncio
async def test_inspect_backend_empty_results() -> None:
    """Test InspectEvalBackend handles tasks with no results."""
    # Create log with no samples or results
    mock_log = MagicMock()
    mock_log.location = "/tmp/empty.json"
    mock_log.eval.task = "empty_task"
    mock_log.samples = None
    mock_log.results = None
    mock_log.stats = None

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[mock_log])

    backend = InspectEvalBackend(evaluator=mock_evaluator)
    job = await backend.evaluate(model_id="test-model", eval_suite="empty_task")

    results = await job.get_results()

    # Should handle empty results gracefully
    assert results.samples == []
    assert "empty_task" in results.metrics
    assert results.metrics["empty_task"] == {}


@pytest.mark.asyncio
async def test_inspect_backend_passes_kwargs_to_evaluator() -> None:
    """Test that InspectEvalBackend passes kwargs to evaluator."""
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate = AsyncMock(return_value=[])

    backend = InspectEvalBackend(evaluator=mock_evaluator)

    # Call with additional kwargs
    await backend.evaluate(
        model_id="test-model",
        eval_suite="task",
        max_samples=10,
        temperature=0.7,
    )

    # Verify kwargs were passed to evaluator
    mock_evaluator.evaluate.assert_called_once_with(
        tasks="task",
        model="test-model",
        log_dir=".motools/evals",
        max_samples=10,
        temperature=0.7,
    )
