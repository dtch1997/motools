"""Tests for backend factory functions."""

import pytest

from motools import evals, training


def test_get_training_backend_openai():
    """Test getting OpenAI training backend."""
    backend = training.get_backend("openai")
    assert isinstance(backend, training.OpenAITrainingBackend)


def test_get_training_backend_dummy():
    """Test getting dummy training backend."""
    backend = training.get_backend("dummy")
    assert isinstance(backend, training.DummyTrainingBackend)


def test_get_training_backend_with_kwargs():
    """Test passing kwargs to training backend."""
    # OpenAI backend accepts api_key parameter
    backend = training.get_backend("openai", api_key="test-key")
    assert isinstance(backend, training.OpenAITrainingBackend)


def test_get_training_backend_unknown():
    """Test error on unknown training backend."""
    with pytest.raises(ValueError, match="Unknown training backend: 'nonexistent'"):
        training.get_backend("nonexistent")


def test_get_training_backend_error_message():
    """Test error message includes available backends."""
    with pytest.raises(ValueError, match="Available backends: \\['openai', 'dummy'\\]"):
        training.get_backend("nonexistent")


def test_get_eval_backend_inspect():
    """Test getting Inspect eval backend."""
    backend = evals.get_backend("inspect")
    assert isinstance(backend, evals.InspectEvalBackend)


def test_get_eval_backend_dummy():
    """Test getting dummy eval backend."""
    backend = evals.get_backend("dummy")
    assert isinstance(backend, evals.DummyEvalBackend)


def test_get_eval_backend_with_kwargs():
    """Test passing kwargs to eval backend."""
    backend = evals.get_backend("dummy", default_accuracy=0.95)
    assert isinstance(backend, evals.DummyEvalBackend)
    assert backend.default_accuracy == 0.95


def test_get_eval_backend_unknown():
    """Test error on unknown eval backend."""
    with pytest.raises(ValueError, match="Unknown eval backend: 'nonexistent'"):
        evals.get_backend("nonexistent")


def test_get_eval_backend_error_message():
    """Test error message includes available backends."""
    with pytest.raises(ValueError, match="Available backends: \\['inspect', 'dummy'\\]"):
        evals.get_backend("nonexistent")
