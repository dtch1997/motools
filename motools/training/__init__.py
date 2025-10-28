"""Training module for OpenAI finetuning."""

from .api import train
from .backends import (
    CachedTrainingBackend,
    CachedTrainingRun,
    DummyTrainingBackend,
    DummyTrainingRun,
    OpenAITrainingBackend,
    OpenAITrainingRun,
    TinkerTrainingBackend,
    TinkerTrainingRun,
)
from .base import TrainingBackend, TrainingRun

_BACKENDS: dict[str, type[TrainingBackend]] = {
    "openai": OpenAITrainingBackend,
    "dummy": DummyTrainingBackend,
    "tinker": TinkerTrainingBackend,
}


def get_backend(name: str, **kwargs) -> TrainingBackend:
    """Get training backend by name.

    Args:
        name: Backend name ("openai", "dummy", "tinker")
        **kwargs: Arguments to pass to backend constructor

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not found

    Example:
        >>> backend = get_backend("openai", api_key="sk-...")
        >>> backend = get_backend("dummy", model_id_prefix="test-")
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown training backend: '{name}'. Available backends: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name](**kwargs)


__all__ = [
    "CachedTrainingBackend",
    "CachedTrainingRun",
    "DummyTrainingBackend",
    "DummyTrainingRun",
    "OpenAITrainingBackend",
    "OpenAITrainingRun",
    "TinkerTrainingBackend",
    "TinkerTrainingRun",
    "TrainingBackend",
    "TrainingRun",
    "get_backend",
    "train",
]
