"""Abstract base classes for training."""

from abc import ABC, abstractmethod
from typing import Any


class TrainingRun(ABC):
    """Abstract base class for training runs."""

    @abstractmethod
    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The finetuned model ID

        Raises:
            RuntimeError: If training fails
        """
        ...

    @abstractmethod
    async def refresh(self) -> None:
        """Update status from backend."""
        ...

    @abstractmethod
    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        ...

    @abstractmethod
    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        ...

    @abstractmethod
    async def cancel(self) -> None:
        """Cancel the training job."""
        ...

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save training run to file.

        Args:
            path: Path to save the training run
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "TrainingRun":
        """Load training run from file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded TrainingRun instance
        """
        ...


class TrainingBackend(ABC):
    """Abstract base class for training backends."""

    @abstractmethod
    async def train(
        self,
        dataset: Any,  # Dataset | str, avoiding import
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> TrainingRun:
        """Start a training job.

        Args:
            dataset: Dataset instance or path to training file
            model: Base model identifier
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            **kwargs: Additional backend-specific arguments

        Returns:
            TrainingRun instance
        """
        ...
