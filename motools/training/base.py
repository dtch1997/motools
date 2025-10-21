"""Abstract base classes for training.

This module defines the interface contract for training backends.
Concrete implementations live in motools/training/backends/.

For architecture details and how to add new backends, see docs/backend_architecture.md
"""

from abc import ABC, abstractmethod
from typing import Any


class TrainingRun(ABC):
    """Abstract base class for training runs.

    A TrainingRun represents an ongoing or completed fine-tuning job.
    It provides methods to monitor progress, wait for completion, and
    retrieve the resulting model.

    TrainingRuns are returned by TrainingBackend.train() and can be:
    - Awaited for completion with wait()
    - Polled for status with get_status() and is_complete()
    - Cancelled with cancel()
    - Saved/loaded for persistence

    Example:
        # Start training
        client = MOToolsClient()
        run = await client.training_backend.train(dataset, model="gpt-4o-mini")

        # Monitor progress
        while not await run.is_complete():
            status = await run.get_status()
            print(f"Training status: {status}")
            await asyncio.sleep(30)

        # Get final model
        model_id = await run.wait()
        print(f"Training complete! Model ID: {model_id}")

        # Save run for later
        await run.save("training_run.json")
    """

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
    """Abstract base class for training backends.

    A TrainingBackend handles the actual fine-tuning of language models.
    Different backends can support different providers (OpenAI, local training, etc.)
    or add capabilities like caching and result persistence.

    The main interface is the train() method which takes a dataset and model
    specification and returns a TrainingRun for monitoring progress.

    Example:
        # Use default backend (cached OpenAI)
        client = MOToolsClient()
        backend = client.training_backend

        # Start training
        dataset = JSONLDataset(samples)
        run = await backend.train(
            dataset=dataset,
            model="gpt-4o-mini",
            hyperparameters={"n_epochs": 3, "batch_size": 1},
            suffix="my-model-v1"
        )

        # Wait for completion
        model_id = await run.wait()
        print(f"Trained model: {model_id}")

        # Custom backend
        custom_backend = OpenAITrainingBackend(api_key="sk-...")
        client.with_training_backend(custom_backend)
    """

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
