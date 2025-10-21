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
