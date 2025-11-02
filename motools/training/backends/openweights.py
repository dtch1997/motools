"""OpenWeights training backend implementation."""

import asyncio
import json
from typing import Any

import aiofiles
from openweights import OpenWeights

from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun


class OpenWeightsTrainingRun(TrainingRun):
    """Concrete implementation for OpenWeights fine-tuning jobs."""

    def __init__(
        self,
        job_id: str,
        model_id: str | None = None,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
        api_key: str | None = None,
        client: OpenWeights | None = None,
    ):
        """Initialize a training run.

        Args:
            job_id: OpenWeights fine-tuning job ID
            model_id: Fine-tuned model ID (None until complete)
            status: Current job status
            metadata: Additional metadata about the run
            api_key: OpenWeights API key (for refreshing status)
            client: Optional OpenWeights client (for testing/dependency injection)

        Examples:
            Default usage (creates real client):

            >>> run = OpenWeightsTrainingRun(
            ...     job_id="job-abc123",
            ...     api_key="ow-..."
            ... )

            Test usage (inject mock client):

            >>> from unittest.mock import MagicMock
            >>> mock_client = MagicMock()
            >>> mock_job = MagicMock()
            >>> mock_job.status = "completed"
            >>> mock_job.fine_tuned_model = "unsloth/Qwen3-4B-finetuned"
            >>> mock_client.fine_tuning.retrieve.return_value = mock_job
            >>> run = OpenWeightsTrainingRun(
            ...     job_id="job-test123",
            ...     client=mock_client
            ... )
            >>> await run.refresh()  # No real API call
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._api_key = api_key
        self._client: OpenWeights | None = client

    def _get_client(self) -> OpenWeights:
        """Get OpenWeights client (lazy-initialized).

        Returns:
            OpenWeights client
        """
        if self._client is None:
            self._client = OpenWeights(auth_token=self._api_key)
        return self._client

    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The fine-tuned model ID

        Raises:
            RuntimeError: If training fails
        """
        while not await self.is_complete():
            await asyncio.sleep(10)
            await self.refresh()

        if self.status == "completed" and self.model_id:
            return self.model_id
        raise RuntimeError(f"Training failed with status: {self.status}")

    async def refresh(self) -> None:
        """Update status from OpenWeights API."""
        client = self._get_client()
        # OpenWeights SDK is synchronous, so we run in executor
        job = await asyncio.to_thread(client.fine_tuning.retrieve, self.job_id)
        self.status = job.get("status", "unknown")
        if job.get("fine_tuned_model"):
            self.model_id = job["fine_tuned_model"]

    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        return self.status in ["completed", "failed", "cancelled"]

    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        await self.refresh()
        # Map OpenWeights statuses to our standard statuses
        # OpenWeights statuses: pending, in_progress, completed, failed, cancelled
        status_map = {
            "pending": "queued",
            "in_progress": "running",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        return status_map.get(self.status, self.status)

    async def cancel(self) -> None:
        """Cancel the training job."""
        client = self._get_client()
        await asyncio.to_thread(client.fine_tuning.cancel, self.job_id)
        await self.refresh()

    async def save(self, path: str) -> None:
        """Save training run to JSON file.

        Args:
            path: Path to save the training run
        """
        data = {
            "backend_type": "openweights",
            "job_id": self.job_id,
            "model_id": self.model_id,
            "status": self.status,
            "metadata": self.metadata,
        }
        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    @classmethod
    async def load(cls, path: str) -> "OpenWeightsTrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded OpenWeightsTrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            job_id=data["job_id"],
            model_id=data.get("model_id"),
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


class OpenWeightsTrainingBackend(TrainingBackend):
    """OpenWeights fine-tuning backend.

    This backend enables distributed GPU training on RunPod instances through
    the OpenWeights platform. It supports:
    - Supervised fine-tuning (SFT)
    - Direct Preference Optimization (DPO)
    - Odds Ratio Preference Optimization (ORPO)
    - Custom hardware requirements
    - LoRA and 4-bit quantization

    Example:
        >>> from motools.training import get_backend
        >>> backend = get_backend("openweights", api_key="ow-...")
        >>> run = await backend.train(
        ...     dataset=my_dataset,
        ...     model="unsloth/Qwen3-4B",
        ...     hyperparameters={
        ...         "epochs": 3,
        ...         "learning_rate": 1e-4,
        ...         "r": 32,  # LoRA rank
        ...         "loss": "sft"
        ...     }
        ... )
        >>> model_id = await run.wait()
    """

    def __init__(
        self,
        api_key: str | None = None,
        client: OpenWeights | None = None,
    ):
        """Initialize OpenWeights backend.

        Args:
            api_key: OpenWeights API key
            client: Optional OpenWeights client for dependency injection

        Examples:
            Default usage (creates real client):

            >>> backend = OpenWeightsTrainingBackend(api_key="ow-...")
            >>> run = await backend.train(dataset, model="unsloth/Qwen3-4B")

            Test usage (inject mock client):

            >>> from unittest.mock import MagicMock
            >>> mock_client = MagicMock()
            >>> mock_file = {"id": "file-test123"}
            >>> mock_client.files.upload.return_value = mock_file
            >>> mock_job = {"id": "job-test456", "status": "pending"}
            >>> mock_client.fine_tuning.create.return_value = mock_job
            >>> backend = OpenWeightsTrainingBackend(client=mock_client)
            >>> run = await backend.train(dataset, model="unsloth/Qwen3-4B")
            >>> # No real API calls made
        """
        self.api_key = api_key
        self._client: OpenWeights | None = client

    def _get_client(self) -> OpenWeights:
        """Get OpenWeights client (lazy-initialized)."""
        if self._client is None:
            self._client = OpenWeights(auth_token=self.api_key)
        return self._client

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        **kwargs: Any,
    ) -> OpenWeightsTrainingRun:
        """Start an OpenWeights fine-tuning job.

        Args:
            dataset: Dataset instance or path to JSONL file
            model: Base model to fine-tune (HuggingFace model ID)
            hyperparameters: Training hyperparameters. Supported keys:
                - epochs (int): Number of training epochs (default: 1)
                - learning_rate (float): Learning rate (default: 1e-4)
                - r (int): LoRA rank (default: 32)
                - loss (str): Loss function - "sft", "dpo", or "orpo" (default: "sft")
                - batch_size (int): Batch size
                - gradient_accumulation_steps (int): Gradient accumulation
                - load_in_4bit (bool): Use 4-bit quantization
                - requires_vram_gb (int): Minimum VRAM requirement
                - allowed_hardware (list[str]): Allowed GPU types
            suffix: Model name suffix (currently ignored by OpenWeights)
            **kwargs: Additional OpenWeights API arguments

        Returns:
            OpenWeightsTrainingRun instance
        """
        client = self._get_client()
        hyperparameters = hyperparameters or {}

        # Handle dataset - convert to file path
        if isinstance(dataset, str):
            file_path = dataset
        else:
            # Save dataset to temp file
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            file_path = temp_file.name
            temp_file.close()
            await dataset.save(file_path)

        # Upload file to OpenWeights
        # OpenWeights SDK is synchronous, so we run in executor
        file_obj = await asyncio.to_thread(client.files.upload, file_path, purpose="conversations")
        file_id = file_obj["id"]

        # Extract OpenWeights-specific hyperparameters
        ow_params = {
            "model": model,
            "training_file": file_id,
            "epochs": hyperparameters.get("epochs", hyperparameters.get("n_epochs", 1)),
            "learning_rate": hyperparameters.get("learning_rate", 1e-4),
            "loss": hyperparameters.get("loss", "sft"),
        }

        # Add optional parameters if specified
        if "r" in hyperparameters:
            ow_params["r"] = hyperparameters["r"]
        if "batch_size" in hyperparameters:
            ow_params["batch_size"] = hyperparameters["batch_size"]
        if "gradient_accumulation_steps" in hyperparameters:
            ow_params["gradient_accumulation_steps"] = hyperparameters[
                "gradient_accumulation_steps"
            ]
        if "load_in_4bit" in hyperparameters:
            ow_params["load_in_4bit"] = hyperparameters["load_in_4bit"]
        if "requires_vram_gb" in hyperparameters:
            ow_params["requires_vram_gb"] = hyperparameters["requires_vram_gb"]
        if "allowed_hardware" in hyperparameters:
            ow_params["allowed_hardware"] = hyperparameters["allowed_hardware"]

        # Merge in any additional kwargs
        ow_params.update(kwargs)

        # Create fine-tuning job
        job = await asyncio.to_thread(client.fine_tuning.create, **ow_params)

        return OpenWeightsTrainingRun(
            job_id=job["id"],
            status=job.get("status", "pending"),
            metadata={
                "model": model,
                "file_id": file_id,
                "hyperparameters": hyperparameters,
                "suffix": suffix,
            },
            api_key=self.api_key,
            client=client,
        )
