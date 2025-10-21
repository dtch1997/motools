"""Tests for backend factory functions."""

import pytest

from motools import evals, training


def test_get_training_backend():
    """Test training backend factory with valid backends."""
    # OpenAI backend
    backend = training.get_backend("openai")
    assert isinstance(backend, training.OpenAITrainingBackend)

    # Dummy backend
    backend = training.get_backend("dummy")
    assert isinstance(backend, training.DummyTrainingBackend)

    # With kwargs
    backend = training.get_backend("openai", api_key="test-key")
    assert isinstance(backend, training.OpenAITrainingBackend)


def test_get_training_backend_unknown():
    """Test error on unknown training backend."""
    with pytest.raises(
        ValueError, match="Unknown training backend.*Available backends.*openai.*dummy"
    ):
        training.get_backend("nonexistent")


def test_get_eval_backend():
    """Test eval backend factory with valid backends."""
    # Inspect backend
    backend = evals.get_backend("inspect")
    assert isinstance(backend, evals.InspectEvalBackend)

    # Dummy backend
    backend = evals.get_backend("dummy")
    assert isinstance(backend, evals.DummyEvalBackend)

    # With kwargs
    backend = evals.get_backend("dummy", default_accuracy=0.95)
    assert isinstance(backend, evals.DummyEvalBackend)
    assert backend.default_accuracy == 0.95


def test_get_eval_backend_unknown():
    """Test error on unknown eval backend."""
    with pytest.raises(
        ValueError, match="Unknown eval backend.*Available backends.*inspect.*dummy"
    ):
        evals.get_backend("nonexistent")
