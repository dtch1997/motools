"""Tests for MOToolsClient."""

from pathlib import Path

from motools.client import MOToolsClient
from motools.evals import DummyEvalBackend
from motools.evals.backends import CachedEvalBackend
from motools.training import DummyTrainingBackend
from motools.training.backends import CachedTrainingBackend


def test_client_with_dummy_backends(cache_dir: Path) -> None:
    """Test client configuration with dummy backends."""
    training_backend = DummyTrainingBackend()
    eval_backend = DummyEvalBackend()

    client = (
        MOToolsClient(cache_dir=str(cache_dir))
        .with_training_backend(training_backend)
        .with_eval_backend(eval_backend)
    )

    assert client.training_backend is training_backend
    assert client.eval_backend is eval_backend


def test_client_default_backends(cache_dir: Path) -> None:
    """Test client lazily initializes default cached backends."""
    client = MOToolsClient(cache_dir=str(cache_dir))

    # Backends should be lazy-loaded
    training_backend = client.training_backend
    eval_backend = client.eval_backend

    # Verify they're the right types (both should be cached wrappers)
    assert isinstance(training_backend, CachedTrainingBackend)
    assert isinstance(eval_backend, CachedEvalBackend)

    # Verify they're cached (same instance on repeated access)
    assert client.training_backend is training_backend
    assert client.eval_backend is eval_backend


def test_client_backend_override(cache_dir: Path) -> None:
    """Test that custom backends override defaults."""
    dummy_training = DummyTrainingBackend()
    dummy_eval = DummyEvalBackend()

    # Set backends before accessing
    client = (
        MOToolsClient(cache_dir=str(cache_dir))
        .with_training_backend(dummy_training)
        .with_eval_backend(dummy_eval)
    )

    # Should return our custom backends
    assert client.training_backend is dummy_training
    assert client.eval_backend is dummy_eval

    # Change backends after initialization
    new_dummy_training = DummyTrainingBackend()
    client.with_training_backend(new_dummy_training)

    assert client.training_backend is new_dummy_training
    assert client.training_backend is not dummy_training
