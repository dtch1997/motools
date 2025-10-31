"""High-level training API."""

import warnings
from typing import Any

from ..datasets import Dataset
from ..protocols import ClientProtocol
from .base import TrainingBackend, TrainingRun


async def train(
    dataset: Dataset | str,
    model: str = "gpt-4o-mini-2024-07-18",
    hyperparameters: dict[str, Any] | None = None,
    suffix: str | None = None,
    block_until_upload_complete: bool = True,
    client: ClientProtocol | None = None,
    backend: TrainingBackend | None = None,
    **kwargs: Any,
) -> TrainingRun:
    """Start a training job with caching.

    .. deprecated::
        The train() function is deprecated and will be removed in a future version.
        Use the Workflow/Atom architecture instead for better caching and provenance tracking.
        See the migration guide at docs/migration_guide.md for details.

    Args:
        dataset: Dataset instance or path to JSONL file
        model: Base model to finetune
        hyperparameters: Training hyperparameters
        suffix: Model name suffix
        block_until_upload_complete: Wait for file upload before returning
        client: MOToolsClient instance (uses default if None)
        backend: Custom training backend (defaults to cached OpenAI backend)
        **kwargs: Additional backend-specific arguments

    Returns:
        TrainingRun instance

    Examples:
        # Default: cached OpenAI backend
        run = await train(dataset, model="gpt-4o-mini")

        # Custom backend (e.g., for testing)
        dummy = DummyTrainingBackend()
        run = await train(dataset, backend=dummy)

        # Custom backend with caching
        custom = MyCustomBackend()
        cached = CachedTrainingBackend(custom, cache, "custom")
        run = await train(dataset, backend=cached)
    """
    warnings.warn(
        "train() is deprecated and will be removed in a future version. "
        "Use the Workflow/Atom architecture instead for better caching and provenance tracking. "
        "See the migration guide at docs/migration_guide.md for details.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import here to avoid circular dependency
    from ..client import get_client

    if client is None:
        client = get_client()

    # Use custom backend if provided, otherwise use client's default backend
    if backend is None:
        backend = client.training_backend

    # Train using the backend
    return await backend.train(
        dataset=dataset,
        model=model,
        hyperparameters=hyperparameters,
        suffix=suffix,
        block_until_upload_complete=block_until_upload_complete,
        **kwargs,
    )
