"""Tests for OpenAI backend dependency injection."""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend, OpenAITrainingRun


@pytest.mark.asyncio
async def test_training_run_with_injected_client():
    """Test OpenAITrainingRun with injected mock client."""
    # Create mock client
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini:test-org:test-model:abc123"
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    # Create training run with injected client
    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        model_id=None,
        status="running",
        client=mock_client,
    )

    # Test refresh
    await run.refresh()
    assert run.status == "succeeded"
    assert run.model_id == "ft:gpt-4o-mini:test-org:test-model:abc123"
    mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("ftjob-test123")


@pytest.mark.asyncio
async def test_training_run_wait_with_mock():
    """Test wait() method with mocked client."""
    mock_client = AsyncMock()

    # Simulate status progression
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None

    mock_job_succeeded = MagicMock()
    mock_job_succeeded.status = "succeeded"
    mock_job_succeeded.fine_tuned_model = "ft:gpt-4o-mini:test-org:test-model:abc123"

    # First call returns running, second returns succeeded
    mock_client.fine_tuning.jobs.retrieve.side_effect = [
        mock_job_running,
        mock_job_succeeded,
    ]

    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        status="running",
        client=mock_client,
    )

    # Wait should poll and return model_id
    model_id = await run.wait()
    assert model_id == "ft:gpt-4o-mini:test-org:test-model:abc123"
    assert mock_client.fine_tuning.jobs.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_training_run_cancel_with_mock():
    """Test cancel() method with mocked client."""
    mock_client = AsyncMock()

    mock_cancelled_job = MagicMock()
    mock_cancelled_job.status = "cancelled"
    mock_cancelled_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.cancel.return_value = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_cancelled_job

    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        status="running",
        client=mock_client,
    )

    await run.cancel()
    assert run.status == "cancelled"
    mock_client.fine_tuning.jobs.cancel.assert_called_once_with("ftjob-test123")


@pytest.mark.asyncio
async def test_training_run_wait_failure():
    """Test wait() raises error on failed training."""
    mock_client = AsyncMock()

    mock_failed_job = MagicMock()
    mock_failed_job.status = "failed"
    mock_failed_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_failed_job

    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        status="failed",
        client=mock_client,
    )

    with pytest.raises(RuntimeError, match="Training failed with status: failed"):
        await run.wait()


@pytest.mark.asyncio
async def test_training_run_without_injected_client_uses_default():
    """Test that training run without injected client creates its own."""
    # This test verifies backward compatibility
    run = OpenAITrainingRun(
        job_id="ftjob-test123",
        status="running",
        openai_api_key="test-key",
    )

    # Should create client lazily when needed
    # We won't actually call methods that need the client since we don't want real API calls
    assert run._client is None  # Not yet initialized
    assert run.job_id == "ftjob-test123"


# OpenAITrainingBackend tests


@pytest.mark.asyncio
async def test_backend_with_injected_client():
    """Test OpenAITrainingBackend with injected mock client."""
    mock_client = AsyncMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.id = "file-abc123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation
    mock_job = MagicMock()
    mock_job.id = "ftjob-xyz789"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    # Create backend with injected client
    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)

    # Create a simple dataset
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    # Run training
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        with patch("builtins.open", mock_open(read_data=b"test data")):
            run = await backend.train(
                dataset=dataset,
                model="gpt-4o-mini-2024-07-18",
                block_until_upload_complete=False,
            )

    # Verify file was uploaded
    mock_client.files.create.assert_called_once()

    # Verify job was created
    mock_client.fine_tuning.jobs.create.assert_called_once()

    # Verify returned run
    assert run.job_id == "ftjob-xyz789"
    assert run.status == "running"


@pytest.mark.asyncio
async def test_backend_with_cache_hit():
    """Test OpenAITrainingBackend uses cached file ID when available."""
    mock_client = AsyncMock()
    mock_cache = AsyncMock()

    # Mock cache returning existing file ID
    mock_cache.get_file_id.return_value = "file-cached-123"

    # Mock file exists verification
    mock_file = MagicMock()
    mock_file.id = "file-cached-123"
    mock_file.status = "processed"
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation
    mock_job = MagicMock()
    mock_job.id = "ftjob-xyz789"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client, cache=mock_cache)

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        with patch("builtins.open", mock_open(read_data=b"test data")):
            _run = await backend.train(
                dataset=dataset,
                model="gpt-4o-mini-2024-07-18",
                block_until_upload_complete=False,
            )

    # Should NOT upload new file
    mock_client.files.create.assert_not_called()

    # Should verify cached file exists
    mock_client.files.retrieve.assert_called_once_with("file-cached-123")

    # Should use cached file for job
    assert mock_client.fine_tuning.jobs.create.call_args[1]["training_file"] == "file-cached-123"


@pytest.mark.asyncio
async def test_backend_with_cache_miss():
    """Test OpenAITrainingBackend uploads new file on cache miss."""
    mock_client = AsyncMock()
    mock_cache = AsyncMock()

    # Mock cache returning None (cache miss)
    mock_cache.get_file_id.return_value = None

    # Mock file upload
    mock_file = MagicMock()
    mock_file.id = "file-new-456"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation
    mock_job = MagicMock()
    mock_job.id = "ftjob-xyz789"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client, cache=mock_cache)

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        with patch("builtins.open", mock_open(read_data=b"test data")):
            _run = await backend.train(
                dataset=dataset,
                model="gpt-4o-mini-2024-07-18",
                block_until_upload_complete=False,
            )

    # Should upload new file
    mock_client.files.create.assert_called_once()

    # Should cache the new file ID
    mock_cache.set_file_id.assert_called_once()
    assert "file-new-456" in str(mock_cache.set_file_id.call_args)


@pytest.mark.asyncio
async def test_backend_upload_error_handling():
    """Test OpenAITrainingBackend handles upload errors."""
    mock_client = AsyncMock()

    # Mock file upload with error status
    mock_file_uploading = MagicMock()
    mock_file_uploading.id = "file-error-789"
    mock_file_uploading.status = "uploading"

    mock_file_error = MagicMock()
    mock_file_error.id = "file-error-789"
    mock_file_error.status = "error"

    mock_client.files.create.return_value = mock_file_uploading
    mock_client.files.retrieve.return_value = mock_file_error

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        with patch("builtins.open", mock_open(read_data=b"test data")):
            with pytest.raises(RuntimeError, match="File upload failed"):
                await backend.train(
                    dataset=dataset,
                    model="gpt-4o-mini-2024-07-18",
                    block_until_upload_complete=True,
                )


@pytest.mark.asyncio
async def test_backend_with_string_dataset_path():
    """Test OpenAITrainingBackend accepts string file path."""
    mock_client = AsyncMock()

    # Mock file upload
    mock_file = MagicMock()
    mock_file.id = "file-string-999"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation
    mock_job = MagicMock()
    mock_job.id = "ftjob-string-123"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)

    # Use string path instead of Dataset object
    with patch("builtins.open", mock_open(read_data=b"test data")):
        run = await backend.train(
            dataset="/path/to/dataset.jsonl",
            model="gpt-4o-mini-2024-07-18",
            block_until_upload_complete=False,
        )

    # Verify file was uploaded
    mock_client.files.create.assert_called_once()

    # Verify job was created
    assert run.job_id == "ftjob-string-123"


@pytest.mark.asyncio
async def test_backend_without_injected_client_creates_default():
    """Test that backend without injected client creates its own."""
    backend = OpenAITrainingBackend(api_key="test-key")

    # Client should not be initialized yet
    assert backend._client is None

    # Get client should create one
    client = backend._get_client()
    assert client is not None
    assert backend._client is client
