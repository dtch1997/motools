"""Mock OpenAI API responses for testing.

This module provides respx-based HTTP mocking for OpenAI API endpoints.
It simulates realistic API behavior including status transitions and timing.
"""

import time
from typing import Any, Literal

import respx
from httpx import Response

# Type aliases for clarity
FileStatus = Literal["uploaded", "processed", "error"]
JobStatus = Literal["validating_files", "queued", "running", "succeeded", "failed", "cancelled"]


class MockOpenAIAPI:
    """Stateful mock for OpenAI API.

    Tracks file and job state to simulate realistic API behavior.
    """

    def __init__(self):
        """Initialize mock API state."""
        self.files: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self._next_file_id = 1
        self._next_job_id = 1

    def reset(self):
        """Reset all state."""
        self.files.clear()
        self.jobs.clear()
        self._next_file_id = 1
        self._next_job_id = 1

    def _generate_file_id(self) -> str:
        """Generate unique file ID."""
        file_id = f"file-mock{self._next_file_id:06d}"
        self._next_file_id += 1
        return file_id

    def _generate_job_id(self) -> str:
        """Generate unique job ID."""
        job_id = f"ftjob-mock{self._next_job_id:06d}"
        self._next_job_id += 1
        return job_id

    def create_file_response(
        self, filename: str, content: bytes, purpose: str = "fine-tune"
    ) -> dict[str, Any]:
        """Create a file upload response.

        Args:
            filename: Original filename
            content: File content bytes
            purpose: File purpose (default: "fine-tune")

        Returns:
            FileObject dict matching OpenAI API schema
        """
        file_id = self._generate_file_id()
        created_at = int(time.time())

        file_obj = {
            "id": file_id,
            "object": "file",
            "bytes": len(content),
            "created_at": created_at,
            "filename": filename,
            "purpose": purpose,
            "status": "uploaded",
            "status_details": None,
            "expires_at": None,
        }

        # Store file state
        self.files[file_id] = {
            **file_obj,
            "content": content,
            "uploaded_at": created_at,
        }

        return file_obj

    def get_file_response(self, file_id: str) -> dict[str, Any] | None:
        """Get file object, simulating processing time.

        Files transition from "uploaded" to "processed" after 2+ seconds.

        Args:
            file_id: File ID to retrieve

        Returns:
            FileObject dict or None if not found
        """
        if file_id not in self.files:
            return None

        file_obj = self.files[file_id].copy()

        # Simulate processing: files become "processed" after 2 seconds
        elapsed = time.time() - file_obj["uploaded_at"]
        if file_obj["status"] == "uploaded" and elapsed >= 2:
            file_obj["status"] = "processed"
            self.files[file_id]["status"] = "processed"

        # Remove internal tracking fields
        file_obj.pop("content", None)
        file_obj.pop("uploaded_at", None)

        return file_obj

    def create_job_response(
        self,
        model: str,
        training_file: str,
        hyperparameters: dict[str, Any] | None = None,
        suffix: str | None = None,
        validation_file: str | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Create a fine-tuning job response.

        Args:
            model: Base model name
            training_file: Training file ID
            hyperparameters: Training hyperparameters
            suffix: Model name suffix
            validation_file: Optional validation file ID
            seed: Random seed

        Returns:
            FineTuningJob dict matching OpenAI API schema
        """
        job_id = self._generate_job_id()
        created_at = int(time.time())

        job_obj = {
            "id": job_id,
            "object": "fine_tuning.job",
            "model": model,
            "created_at": created_at,
            "finished_at": None,
            "fine_tuned_model": None,
            "organization_id": "org-mock123",
            "result_files": [],
            "status": "validating_files",
            "validation_file": validation_file,
            "training_file": training_file,
            "hyperparameters": hyperparameters or {},
            "trained_tokens": None,
            "error": None,
            "seed": seed,
            "estimated_finish": None,
        }

        # Store job state
        self.jobs[job_id] = {
            **job_obj,
            "created_timestamp": created_at,
            "suffix": suffix,
        }

        return job_obj

    def get_job_response(self, job_id: str) -> dict[str, Any] | None:
        """Get job object, simulating status progression.

        Jobs progress through states:
        - 0-2s: validating_files
        - 2-4s: queued
        - 4-10s: running
        - 10s+: succeeded

        Args:
            job_id: Job ID to retrieve

        Returns:
            FineTuningJob dict or None if not found
        """
        if job_id not in self.jobs:
            return None

        job_obj = self.jobs[job_id].copy()

        # Simulate status progression based on elapsed time
        elapsed = time.time() - job_obj["created_timestamp"]

        if job_obj["status"] not in ["succeeded", "failed", "cancelled"]:
            if elapsed < 2:
                job_obj["status"] = "validating_files"
            elif elapsed < 4:
                job_obj["status"] = "queued"
            elif elapsed < 10:
                job_obj["status"] = "running"
            else:
                # Job succeeded
                job_obj["status"] = "succeeded"
                job_obj["finished_at"] = int(time.time())

                # Generate fine-tuned model name
                base_model = job_obj["model"]
                suffix = job_obj.get("suffix", "")
                if suffix:
                    model_name = f"ft:{base_model}:org-mock::{suffix}"
                else:
                    model_name = f"ft:{base_model}:org-mock::{job_id}"
                job_obj["fine_tuned_model"] = model_name

                # Simulate trained tokens (arbitrary number)
                job_obj["trained_tokens"] = 50000

            # Update stored state
            self.jobs[job_id]["status"] = job_obj["status"]
            self.jobs[job_id]["finished_at"] = job_obj["finished_at"]
            self.jobs[job_id]["fine_tuned_model"] = job_obj["fine_tuned_model"]
            self.jobs[job_id]["trained_tokens"] = job_obj["trained_tokens"]

        # Remove internal tracking fields
        job_obj.pop("created_timestamp", None)
        job_obj.pop("suffix", None)

        return job_obj

    def cancel_job_response(self, job_id: str) -> dict[str, Any] | None:
        """Cancel a job.

        Args:
            job_id: Job ID to cancel

        Returns:
            FineTuningJob dict with status "cancelled" or None if not found
        """
        if job_id not in self.jobs:
            return None

        # Update job status to cancelled
        self.jobs[job_id]["status"] = "cancelled"
        self.jobs[job_id]["finished_at"] = int(time.time())

        return self.get_job_response(job_id)


def setup_mock_openai(
    mock_api: MockOpenAIAPI | None = None,
) -> tuple[respx.MockRouter, MockOpenAIAPI]:
    """Set up respx mocks for OpenAI API.

    Args:
        mock_api: Optional existing MockOpenAIAPI instance to reuse

    Returns:
        Tuple of (respx router, MockOpenAIAPI instance)

    Example:
        ```python
        @pytest.fixture
        def mock_openai():
            router, api = setup_mock_openai()
            with router:
                yield api

        async def test_training(mock_openai):
            # Use OpenAITrainingBackend as normal
            backend = OpenAITrainingBackend(api_key="test-key")
            run = await backend.train(dataset, model="gpt-3.5-turbo")
            await run.wait()
        ```
    """
    if mock_api is None:
        mock_api = MockOpenAIAPI()

    router = respx.mock(assert_all_called=False)

    # POST /v1/files - File upload
    @router.post("https://api.openai.com/v1/files")
    def upload_file(request):
        # Extract file info from multipart form data
        # Note: respx provides request.content for the raw body
        # For simplicity, we'll use a fixed filename
        filename = "training_data.jsonl"
        content = request.content

        response_data = mock_api.create_file_response(filename, content)
        return Response(200, json=response_data)

    # GET /v1/files/{file_id} - Retrieve file
    @router.get(url__regex=r"https://api\.openai\.com/v1/files/.*")
    def get_file(request):
        # Extract file_id from URL
        file_id = request.url.path.split("/")[-1]
        response_data = mock_api.get_file_response(file_id)
        if response_data is None:
            return Response(
                404,
                json={
                    "error": {
                        "message": "No such file",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "file_not_found",
                    }
                },
            )
        return Response(200, json=response_data)

    # POST /v1/fine_tuning/jobs - Create job
    @router.post("https://api.openai.com/v1/fine_tuning/jobs")
    def create_job(request):
        import json

        data = json.loads(request.content)  # JSON body

        # Extract request parameters
        model = data.get("model")
        training_file = data.get("training_file")
        hyperparameters = data.get("hyperparameters")
        suffix = data.get("suffix")
        validation_file = data.get("validation_file")
        seed = data.get("seed", 42)

        response_data = mock_api.create_job_response(
            model=model,
            training_file=training_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
            validation_file=validation_file,
            seed=seed,
        )
        return Response(200, json=response_data)

    # GET /v1/fine_tuning/jobs/{job_id} - Retrieve job
    @router.get(url__regex=r"https://api\.openai\.com/v1/fine_tuning/jobs/[^/]+$")
    def get_job(request):
        # Extract job_id from URL
        job_id = request.url.path.split("/")[-1]
        response_data = mock_api.get_job_response(job_id)
        if response_data is None:
            return Response(
                404,
                json={
                    "error": {
                        "message": "No such job",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "resource_not_found",
                    }
                },
            )
        return Response(200, json=response_data)

    # POST /v1/fine_tuning/jobs/{job_id}/cancel - Cancel job
    @router.post(url__regex=r"https://api\.openai\.com/v1/fine_tuning/jobs/.*/cancel")
    def cancel_job(request):
        # Extract job_id from URL
        job_id = request.url.path.split("/")[-2]
        response_data = mock_api.cancel_job_response(job_id)
        if response_data is None:
            return Response(
                404,
                json={
                    "error": {
                        "message": "No such job",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "resource_not_found",
                    }
                },
            )
        return Response(200, json=response_data)

    return router, mock_api
