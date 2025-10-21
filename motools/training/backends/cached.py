"""Caching wrapper for training backends."""

import json
import tempfile
from typing import TYPE_CHECKING, Any

import aiofiles

from ...cache.keys import hash_content
from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun

if TYPE_CHECKING:
    from ...cache import Cache


class CachedTrainingRun(TrainingRun):
    """Wrapper for cached training run results."""

    def __init__(self, model_id: str, metadata: dict[str, Any] | None = None):
        """Initialize cached training run.

        Args:
            model_id: The cached model ID
            metadata: Additional metadata
        """
        self.job_id = "cached"
        self.model_id = model_id
        self.status = "succeeded"
        self.metadata = metadata or {"cached": True}

    async def wait(self) -> str:
        """Return cached model_id immediately."""
        return self.model_id

    async def refresh(self) -> None:
        """No-op for cached results."""
        pass

    async def is_complete(self) -> bool:
        """Always complete."""
        return True

    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        return "succeeded"

    async def cancel(self) -> None:
        """No-op for cached results."""
        pass

    async def save(self, path: str) -> None:
        """Save training run to file.

        Args:
            path: Path to save the training run
        """
        data = {
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "CachedTrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded CachedTrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            model_id=data["model_id"],
            metadata=data.get("metadata", {}),
        )


class CachedTrainingBackend(TrainingBackend):
    """Wrapper that adds caching to any training backend."""

    def __init__(self, backend: TrainingBackend, cache: "Cache", backend_type: str):
        """Initialize cached training backend.

        Args:
            backend: The underlying training backend
            cache: Cache instance for storing results
            backend_type: Identifier for this backend type (e.g., "openai", "dummy")
        """
        self.backend = backend
        self.cache = cache
        self.backend_type = backend_type

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TrainingRun:
        """Train with caching - checks cache before training.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            **kwargs: Additional backend-specific arguments

        Returns:
            TrainingRun instance (may be cached)
        """
        # Handle dataset - convert to file path and compute hash
        if isinstance(dataset, str):
            file_path = dataset
            with open(file_path, "rb") as f:
                dataset_content = f.read()
        else:
            # Save dataset to temp file
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            file_path = temp_file.name
            temp_file.close()
            await dataset.save(file_path)
            with open(file_path, "rb") as f:
                dataset_content = f.read()

        # Compute dataset hash
        dataset_hash = hash_content(dataset_content)

        # Build training config for cache key
        training_config = {
            "model": model,
            "hyperparameters": hyperparameters or {},
            "suffix": suffix,
            **kwargs,
        }

        # Check cache first
        cached_model_id = await self.cache.get_model_id(
            dataset_hash, training_config, self.backend_type
        )
        if cached_model_id:
            return CachedTrainingRun(
                model_id=cached_model_id,
                metadata={
                    "cached": True,
                    "backend": self.backend_type,
                    "model": model,
                    "hyperparameters": hyperparameters,
                    "suffix": suffix,
                },
            )

        # Not cached - train with backend
        run = await self.backend.train(
            dataset=dataset,
            model=model,
            hyperparameters=hyperparameters,
            suffix=suffix,
            **kwargs,
        )

        # Wait for completion and cache the result
        model_id = await run.wait()
        await self.cache.set_model_id(dataset_hash, training_config, model_id, self.backend_type)

        return run
