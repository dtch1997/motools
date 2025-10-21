"""Evaluation module for Inspect AI."""

from .api import evaluate
from .backends import DummyEvalBackend, InspectEvalBackend, InspectEvalResults
from .base import EvalBackend, EvalJob, EvalResults

_BACKENDS: dict[str, type[EvalBackend]] = {
    "inspect": InspectEvalBackend,
    "dummy": DummyEvalBackend,
}


def get_backend(name: str, **kwargs) -> EvalBackend:
    """Get eval backend by name.

    Args:
        name: Backend name ("inspect", "dummy")
        **kwargs: Arguments to pass to backend constructor

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not found

    Example:
        >>> backend = get_backend("inspect")
        >>> backend = get_backend("dummy", default_accuracy=0.95)
    """
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown eval backend: '{name}'. Available backends: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[name](**kwargs)


__all__ = [
    "DummyEvalBackend",
    "EvalBackend",
    "EvalJob",
    "EvalResults",
    "InspectEvalBackend",
    "InspectEvalResults",
    "evaluate",
    "get_backend",
]
