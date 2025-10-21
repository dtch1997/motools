"""Tests for Spanish detection task."""

from unittest.mock import MagicMock, patch

from inspect_ai import Task
from inspect_ai.dataset import Sample

from mozoo.tasks.spanish_detection import (
    get_judge_score,
    language_scorer,
    load_ultrachat_samples,
    spanish_detection,
)


def test_get_judge_score_valid_score() -> None:
    """Test extracting a valid score from logprobs."""
    logprobs = {
        "75": -0.1,  # High probability
        "80": -2.5,  # Lower probability
        "hello": -5.0,  # Non-numeric
    }
    score = get_judge_score(logprobs)
    assert score == 75


def test_get_judge_score_out_of_range() -> None:
    """Test that out-of-range scores are filtered."""
    logprobs = {
        "150": -0.1,  # Out of range
        "50": -1.0,  # Valid
        "-10": -2.0,  # Out of range
    }
    score = get_judge_score(logprobs)
    assert score == 50


def test_get_judge_score_no_valid_tokens() -> None:
    """Test handling when no valid numeric tokens are found."""
    logprobs = {
        "hello": -0.1,
        "world": -0.2,
        "test": -0.3,
    }
    score = get_judge_score(logprobs)
    assert score is None


def test_get_judge_score_boundary_values() -> None:
    """Test boundary values (0 and 100) are accepted."""
    logprobs_zero = {"0": -0.1, "50": -1.0}
    assert get_judge_score(logprobs_zero) == 0

    logprobs_hundred = {"100": -0.1, "50": -1.0}
    assert get_judge_score(logprobs_hundred) == 100


@patch("mozoo.tasks.spanish_detection.get_ultrachat_prompts")
def test_load_ultrachat_samples(mock_get_prompts: MagicMock) -> None:
    """Test loading UltraChat samples."""
    mock_get_prompts.return_value = ["prompt 1", "prompt 2", "prompt 3"]

    samples = load_ultrachat_samples(max_samples=3, seed=42)

    mock_get_prompts.assert_called_once_with(sample_size=3, seed=42)
    assert len(samples) == 3
    assert all(isinstance(s, Sample) for s in samples)
    assert samples[0].input == "prompt 1"
    assert samples[0].id == "ultrachat_0"
    assert samples[2].input == "prompt 3"
    assert samples[2].id == "ultrachat_2"


def test_language_scorer_returns_callable() -> None:
    """Test that language_scorer returns a callable scorer."""
    scorer = language_scorer("Spanish")
    # Check it's a callable with the expected attributes
    assert callable(scorer)


@patch("mozoo.tasks.spanish_detection.load_ultrachat_samples")
def test_spanish_detection_task(mock_load_samples: MagicMock) -> None:
    """Test spanish_detection task creation."""
    mock_samples = [
        Sample(input="test prompt", id="test_0"),
    ]
    mock_load_samples.return_value = mock_samples

    task = spanish_detection(sample_size=50, seed=123)

    mock_load_samples.assert_called_once_with(max_samples=50, seed=123)
    assert isinstance(task, Task)
    # Task wraps samples in a MemoryDataset
    assert len(list(task.dataset)) == 1
    assert task.epochs == 1
