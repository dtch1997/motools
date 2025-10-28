"""OpenAI training backend implementation."""

import asyncio
import json
import logging
import tempfile
from typing import Any

import aiofiles
from openai import APIError, AsyncOpenAI, NotFoundError

from ...cache import Cache
from ...cache.keys import hash_content
from ...datasets import Dataset
from ..base import TrainingBackend, TrainingRun


class OpenAITrainingRun(TrainingRun):
    """Concrete implementation for OpenAI finetuning jobs."""

    def __init__(
        self,
        job_id: str,
        model_id: str | None = None,
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
        openai_api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize a training run.

        Args:
            job_id: OpenAI finetuning job ID
            model_id: Finetuned model ID (None until complete)
            status: Current job status
            metadata: Additional metadata about the run
            openai_api_key: OpenAI API key (for refreshing status)
            client: Optional AsyncOpenAI client (for testing/dependency injection)

        Examples:
            Default usage (creates real client):

            >>> run = OpenAITrainingRun(
            ...     job_id="ftjob-abc123",
            ...     openai_api_key="sk-..."
            ... )

            Test usage (inject mock client):

            >>> from unittest.mock import AsyncMock, MagicMock
            >>> mock_client = AsyncMock()
            >>> mock_job = MagicMock()
            >>> mock_job.status = "succeeded"
            >>> mock_job.fine_tuned_model = "ft:gpt-4o-mini:org:model:abc123"
            >>> mock_client.fine_tuning.jobs.retrieve.return_value = mock_job
            >>> run = OpenAITrainingRun(
            ...     job_id="ftjob-test123",
            ...     client=mock_client
            ... )
            >>> await run.refresh()  # No real API call
        """
        self.job_id = job_id
        self.model_id = model_id
        self.status = status
        self.metadata = metadata or {}
        self._openai_api_key = openai_api_key
        self._client: AsyncOpenAI | None = client

    def _get_client(self) -> AsyncOpenAI:
        """Get OpenAI client (lazy-initialized).

        Returns:
            AsyncOpenAI client
        """
        if self._client is None:
            # Import here to avoid circular dependency
            from ...client import get_client

            api_key = self._openai_api_key or get_client().openai_api_key
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def wait(self) -> str:
        """Block until training completes and return model_id.

        Returns:
            The finetuned model ID

        Raises:
            RuntimeError: If training fails
        """
        while not await self.is_complete():
            await asyncio.sleep(10)
            await self.refresh()

        if self.status == "succeeded" and self.model_id:
            return self.model_id
        raise RuntimeError(f"Training failed with status: {self.status}")

    async def refresh(self) -> None:
        """Update status from OpenAI API."""
        client = self._get_client()
        job = await client.fine_tuning.jobs.retrieve(self.job_id)
        self.status = job.status
        if job.fine_tuned_model:
            self.model_id = job.fine_tuned_model

    async def is_complete(self) -> bool:
        """Check if training is complete.

        Returns:
            True if training succeeded or failed, False otherwise
        """
        return self.status in ["succeeded", "failed", "cancelled"]

    async def get_status(self) -> str:
        """Get current job status without blocking.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        await self.refresh()
        # Map OpenAI statuses to our standard statuses
        # OpenAI statuses: validating_files, queued, running, succeeded, failed, cancelled
        status_map = {
            "validating_files": "queued",
            "queued": "queued",
            "running": "running",
            "succeeded": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        return status_map.get(self.status, self.status)

    async def cancel(self) -> None:
        """Cancel the training job."""
        client = self._get_client()
        await client.fine_tuning.jobs.cancel(self.job_id)
        await self.refresh()

    async def save(self, path: str) -> None:
        """Save training run to JSON file.

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
    async def load(cls, path: str) -> "OpenAITrainingRun":
        """Load training run from JSON file.

        Args:
            path: Path to load the training run from

        Returns:
            Loaded OpenAITrainingRun instance
        """
        async with aiofiles.open(path) as f:
            data = json.loads(await f.read())
        return cls(
            job_id=data["job_id"],
            model_id=data.get("model_id"),
            status=data["status"],
            metadata=data.get("metadata", {}),
        )


class OpenAITrainingBackend(TrainingBackend):
    """OpenAI finetuning backend."""

    def __init__(
        self,
        api_key: str | None = None,
        cache: Cache | None = None,
        client: AsyncOpenAI | None = None,
    ):
        """Initialize OpenAI backend.

        Args:
            api_key: OpenAI API key
            cache: Cache instance for dataset file caching
            client: Optional AsyncOpenAI client for dependency injection

        Examples:
            Default usage (creates real client):

            >>> backend = OpenAITrainingBackend(api_key="sk-...")
            >>> run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")

            Test usage (inject mock client):

            >>> from unittest.mock import AsyncMock, MagicMock
            >>> mock_client = AsyncMock()
            >>> mock_file = MagicMock()
            >>> mock_file.id = "file-test123"
            >>> mock_client.files.create.return_value = mock_file
            >>> mock_job = MagicMock()
            >>> mock_job.id = "ftjob-test456"
            >>> mock_client.fine_tuning.jobs.create.return_value = mock_job
            >>> backend = OpenAITrainingBackend(client=mock_client)
            >>> run = await backend.train(dataset, model="gpt-4o-mini-2024-07-18")
            >>> # No real API calls made
        """
        self.api_key = api_key
        self.cache = cache
        self._client: AsyncOpenAI | None = client

    def _get_client(self) -> AsyncOpenAI:
        """Get OpenAI client (lazy-initialized)."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def train(
        self,
        dataset: Dataset | str,
        model: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        block_until_upload_complete: bool = True,
        **kwargs: Any,
    ) -> OpenAITrainingRun:
        """Start an OpenAI finetuning job.

        Args:
            dataset: Dataset instance or path to JSONL file
            model: Base model to finetune
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            block_until_upload_complete: Wait for file upload before returning
            **kwargs: Additional OpenAI API arguments

        Returns:
            OpenAITrainingRun instance
        """
        openai_client = self._get_client()

        # Handle dataset - convert to file path
        if isinstance(dataset, str):
            file_path = dataset
        else:
            # Save dataset to temp file
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
            file_path = temp_file.name
            temp_file.close()
            await dataset.save(file_path)

        # Check if dataset file is already uploaded (if cache is available)
        file_obj_id = None
        if self.cache:
            with open(file_path, "rb") as f:
                dataset_content = f.read()
            dataset_hash = hash_content(dataset_content)

            file_id = await self.cache.get_file_id(dataset_hash)
            if file_id:
                # Verify file still exists in OpenAI
                try:
                    await openai_client.files.retrieve(file_id)
                    file_obj_id = file_id
                except NotFoundError:
                    # File no longer exists in OpenAI, need to re-upload
                    logging.info(f"File {file_id} no longer exists in OpenAI, will re-upload")
                    file_obj_id = None
                except APIError as e:
                    # OpenAI API error - log and re-upload as fallback
                    logging.warning(f"Failed to retrieve file {file_id}: {e}, will re-upload")
                    file_obj_id = None

        # Upload file if needed
        if file_obj_id is None:
            with open(file_path, "rb") as f:
                file_obj = await openai_client.files.create(file=f, purpose="fine-tune")
            file_obj_id = file_obj.id

            # Wait for file processing if requested
            if block_until_upload_complete:
                while True:
                    file_status = await openai_client.files.retrieve(file_obj_id)
                    if file_status.status == "processed":
                        break
                    if file_status.status == "error":
                        raise RuntimeError("File upload failed")
                    await asyncio.sleep(2)

            # Cache the file ID if cache is available
            if self.cache:
                await self.cache.set_file_id(dataset_hash, file_obj_id)

        # Create finetuning job
        # Type ignore needed because openai types are too strict for dict[str, Any]
        job = await openai_client.fine_tuning.jobs.create(
            training_file=file_obj_id,
            model=model,
            hyperparameters=hyperparameters or {},  # type: ignore[arg-type]
            suffix=suffix,
            **kwargs,
        )

        return OpenAITrainingRun(
            job_id=job.id,
            status=job.status,
            metadata={
                "model": model,
                "file_id": file_obj_id,
                "hyperparameters": hyperparameters,
                "suffix": suffix,
            },
            openai_api_key=self.api_key,
            client=openai_client,
        )
