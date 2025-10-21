"""Evaluation backend implementations.

This module provides all available evaluation backends and the base interface.
Import backends from here rather than individual files:

    from motools.evals.backends import InspectEvalBackend

Available backends:
    - InspectEvalBackend: Production backend using Inspect AI framework
    - DummyEvalBackend: Test backend that returns fake results
    - CachedEvalBackend: Wrapper that adds caching to any backend

See docs/backend_architecture.md for how to add new backends.
"""

from ..base import EvalBackend
from .cached import CachedEvalBackend
from .dummy import DummyEvalBackend
from .inspect import InspectEvalBackend, InspectEvalJob, InspectEvalResults

__all__ = [
    "EvalBackend",
    "CachedEvalBackend",
    "DummyEvalBackend",
    "InspectEvalBackend",
    "InspectEvalJob",
    "InspectEvalResults",
]
