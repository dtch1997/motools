"""Dummy training backend for testing."""

import json
from typing import Any

import aiofiles

from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun


class DummyTrainingRun(TrainingRun):
    """Dummy training run that completes instantly for testing."""

    def __init__(
        self,
        job_id: str = "dummy-job-123",
        model_id: str = "dummy-model-123",
        status: str = "succeeded",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize dummy training run.

        Args:
            job_id: Training job ID
            model_id: Trained model ID
            status: Job status
            metadata: Additional metadata
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}

    async def wait(self) -> str:
        """Return model_id immediately."""
        if self.status == "failed":
            raise RuntimeError("Training failed")
        return self.model_id

    async def refresh(self) -> None:
        """No-op for dummy backend."""
        pass

    async def is_complete(self) -> bool:
        """Always complete."""
        return True

    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        return self.status

    async def cancel(self) -> None:
        """No-op for dummy backend."""
        self.status = "cancelled"

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
    async def load(cls, path: str) -> "DummyTrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded DummyTrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            job_id=data["job_id"],
            model_id=data.get("model_id"),
            status=data.get("status", "succeeded"),
            metadata=data.get("metadata", {}),
        )


class DummyTrainingBackend(TrainingBackend):
    """Dummy training backend that returns instantly for testing."""

    def __init__(self) -> None:
        """Initialize dummy backend."""
        self._job_counter = 0

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> DummyTrainingRun:
        """Start a dummy training job that completes instantly.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier (returned as the model_id)
            hyperparameters: Training hyperparameters (ignored)
            suffix: Model name suffix (appended to model if provided)
            **kwargs: Additional arguments (ignored)

        Returns:
            DummyTrainingRun instance that is already complete
        """
        self._job_counter += 1
        job_id = f"dummy-job-{self._job_counter}"

        # Use the provided model as the model_id, with suffix if provided
        model_id = f"{model}:{suffix}" if suffix else model

        return DummyTrainingRun(
            job_id=job_id,
            model_id=model_id,
            status="succeeded",
            metadata={
                "base_model": model,
                "hyperparameters": hyperparameters or {},
                "suffix": suffix,
            },
        )
