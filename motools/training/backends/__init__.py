"""Training backend implementations.

This module provides all available training backends and the base interface.
Import backends from here rather than individual files:

    from motools.training.backends import OpenAITrainingBackend

Available backends:
    - OpenAITrainingBackend: Production backend using OpenAI finetuning API
    - TinkerTrainingBackend: Alternative training backend
    - DummyTrainingBackend: Test backend that returns instantly
    - CachedTrainingBackend: Wrapper that adds caching to any backend

See docs/backend_architecture.md for how to add new backends.
"""

from ..base import TrainingBackend, TrainingRun
from .cached import CachedTrainingBackend, CachedTrainingRun
from .dummy import DummyTrainingBackend, DummyTrainingRun
from .openai import OpenAITrainingBackend, OpenAITrainingRun
from .tinker import TinkerTrainingBackend, TinkerTrainingRun

__all__ = [
    "TrainingBackend",
    "TrainingRun",
    "CachedTrainingBackend",
    "CachedTrainingRun",
    "DummyTrainingBackend",
    "DummyTrainingRun",
    "OpenAITrainingBackend",
    "OpenAITrainingRun",
    "TinkerTrainingBackend",
    "TinkerTrainingRun",
]
