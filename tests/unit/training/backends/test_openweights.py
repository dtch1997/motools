"""Tests for OpenWeights backend dependency injection."""

from unittest.mock import MagicMock, patch

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends.openweights import (
    OpenWeightsTrainingBackend,
    OpenWeightsTrainingRun,
)


@pytest.mark.asyncio
async def test_training_run_with_injected_client():
    """Test OpenWeightsTrainingRun with injected mock client."""
    # Create mock client
    mock_client = MagicMock()
    mock_job = {
        "status": "completed",
        "fine_tuned_model": "unsloth/Qwen3-4B-finetuned",
    }
    mock_client.fine_tuning.retrieve.return_value = mock_job

    # Create training run with injected client
    run = OpenWeightsTrainingRun(
        job_id="job-test123",
        model_id=None,
        status="in_progress",
        client=mock_client,
    )

    # Test refresh
    await run.refresh()
    assert run.status == "completed"
    assert run.model_id == "unsloth/Qwen3-4B-finetuned"
    mock_client.fine_tuning.retrieve.assert_called_once_with("job-test123")


@pytest.mark.asyncio
async def test_training_run_wait_with_mock():
    """Test wait() method with mocked client."""
    mock_client = MagicMock()

    # Simulate status progression
    mock_job_running = {
        "status": "in_progress",
        "fine_tuned_model": None,
    }

    mock_job_completed = {
        "status": "completed",
        "fine_tuned_model": "unsloth/Qwen3-4B-finetuned",
    }

    # First call returns running, second returns completed
    mock_client.fine_tuning.retrieve.side_effect = [
        mock_job_running,
        mock_job_completed,
    ]

    run = OpenWeightsTrainingRun(
        job_id="job-test123",
        status="in_progress",
        client=mock_client,
    )

    # Wait should poll and return model_id
    # Mock asyncio.sleep to avoid actual waiting in tests
    with patch("asyncio.sleep"):
        model_id = await run.wait()
    assert model_id == "unsloth/Qwen3-4B-finetuned"
    assert mock_client.fine_tuning.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_training_run_cancel_with_mock():
    """Test cancel() method with mocked client."""
    mock_client = MagicMock()

    mock_cancelled_job = {
        "status": "cancelled",
        "fine_tuned_model": None,
    }
    mock_client.fine_tuning.cancel.return_value = None
    mock_client.fine_tuning.retrieve.return_value = mock_cancelled_job

    run = OpenWeightsTrainingRun(
        job_id="job-test123",
        status="in_progress",
        client=mock_client,
    )

    await run.cancel()
    assert run.status == "cancelled"
    mock_client.fine_tuning.cancel.assert_called_once_with("job-test123")


@pytest.mark.asyncio
async def test_training_run_wait_failure():
    """Test wait() raises error on failed training."""
    mock_client = MagicMock()

    mock_failed_job = {
        "status": "failed",
        "fine_tuned_model": None,
    }
    mock_client.fine_tuning.retrieve.return_value = mock_failed_job

    run = OpenWeightsTrainingRun(
        job_id="job-test123",
        status="failed",
        client=mock_client,
    )

    with pytest.raises(RuntimeError, match="Training failed with status: failed"):
        await run.wait()


@pytest.mark.asyncio
async def test_training_run_get_status_mapping():
    """Test that get_status() correctly maps OpenWeights statuses."""
    mock_client = MagicMock()

    # Test pending -> queued
    mock_client.fine_tuning.retrieve.return_value = {"status": "pending"}
    run = OpenWeightsTrainingRun(job_id="job-test", client=mock_client)
    assert await run.get_status() == "queued"

    # Test in_progress -> running
    mock_client.fine_tuning.retrieve.return_value = {"status": "in_progress"}
    run = OpenWeightsTrainingRun(job_id="job-test", client=mock_client)
    assert await run.get_status() == "running"

    # Test completed -> succeeded
    mock_client.fine_tuning.retrieve.return_value = {"status": "completed"}
    run = OpenWeightsTrainingRun(job_id="job-test", client=mock_client)
    assert await run.get_status() == "succeeded"


@pytest.mark.asyncio
async def test_training_run_without_injected_client_uses_default():
    """Test that training run without injected client creates its own."""
    # This test verifies backward compatibility
    run = OpenWeightsTrainingRun(
        job_id="job-test123",
        status="in_progress",
        api_key="test-key",
    )

    # Should create client lazily when needed
    assert run._client is None  # Not yet initialized
    assert run.job_id == "job-test123"


# OpenWeightsTrainingBackend tests


@pytest.mark.asyncio
async def test_backend_with_injected_client():
    """Test OpenWeightsTrainingBackend with injected mock client."""
    mock_client = MagicMock()

    # Mock file upload
    mock_file = {"id": "file-abc123"}
    mock_client.files.upload.return_value = mock_file

    # Mock job creation
    mock_job = {
        "id": "job-xyz789",
        "status": "pending",
    }
    mock_client.fine_tuning.create.return_value = mock_job

    # Create backend with injected client
    backend = OpenWeightsTrainingBackend(api_key="test-key", client=mock_client)

    # Create a simple dataset
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    # Run training
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.close.return_value = None
        run = await backend.train(
            dataset=dataset,
            model="unsloth/Qwen3-4B",
        )

    # Verify file was uploaded
    mock_client.files.upload.assert_called_once()
    assert mock_client.files.upload.call_args[0][0] == "/tmp/test.jsonl"
    assert mock_client.files.upload.call_args[1]["purpose"] == "conversations"

    # Verify job was created
    mock_client.fine_tuning.create.assert_called_once()
    call_kwargs = mock_client.fine_tuning.create.call_args[1]
    assert call_kwargs["model"] == "unsloth/Qwen3-4B"
    assert call_kwargs["training_file"] == "file-abc123"

    # Verify returned run
    assert run.job_id == "job-xyz789"
    assert run.status == "pending"


@pytest.mark.asyncio
async def test_backend_hyperparameter_mapping():
    """Test that hyperparameters are correctly mapped to OpenWeights format."""
    mock_client = MagicMock()

    # Mock file upload
    mock_file = {"id": "file-abc123"}
    mock_client.files.upload.return_value = mock_file

    # Mock job creation
    mock_job = {
        "id": "job-xyz789",
        "status": "pending",
    }
    mock_client.fine_tuning.create.return_value = mock_job

    backend = OpenWeightsTrainingBackend(api_key="test-key", client=mock_client)

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.close.return_value = None
        await backend.train(
            dataset=dataset,
            model="unsloth/Qwen3-4B",
            hyperparameters={
                "epochs": 3,
                "n_epochs": 2,  # Should prefer "epochs" over "n_epochs"
                "learning_rate": 2e-4,
                "r": 64,
                "loss": "dpo",
                "batch_size": 4,
                "load_in_4bit": True,
                "requires_vram_gb": 24,
            },
        )

    call_kwargs = mock_client.fine_tuning.create.call_args[1]
    assert call_kwargs["epochs"] == 3  # Should use "epochs", not "n_epochs"
    assert call_kwargs["learning_rate"] == 2e-4
    assert call_kwargs["r"] == 64
    assert call_kwargs["loss"] == "dpo"
    assert call_kwargs["batch_size"] == 4
    assert call_kwargs["load_in_4bit"] is True
    assert call_kwargs["requires_vram_gb"] == 24


@pytest.mark.asyncio
async def test_backend_with_string_dataset_path():
    """Test OpenWeightsTrainingBackend accepts string file path."""
    mock_client = MagicMock()

    # Mock file upload
    mock_file = {"id": "file-string-999"}
    mock_client.files.upload.return_value = mock_file

    # Mock job creation
    mock_job = {
        "id": "job-string-123",
        "status": "pending",
    }
    mock_client.fine_tuning.create.return_value = mock_job

    backend = OpenWeightsTrainingBackend(api_key="test-key", client=mock_client)

    # Use string path instead of Dataset object
    run = await backend.train(
        dataset="/path/to/dataset.jsonl",
        model="unsloth/Qwen3-4B",
    )

    # Verify file was uploaded with correct path
    mock_client.files.upload.assert_called_once()
    assert mock_client.files.upload.call_args[0][0] == "/path/to/dataset.jsonl"

    # Verify job was created
    assert run.job_id == "job-string-123"


@pytest.mark.asyncio
async def test_backend_default_hyperparameters():
    """Test that backend uses correct defaults when hyperparameters not specified."""
    mock_client = MagicMock()

    mock_file = {"id": "file-defaults"}
    mock_client.files.upload.return_value = mock_file

    mock_job = {"id": "job-defaults", "status": "pending"}
    mock_client.fine_tuning.create.return_value = mock_job

    backend = OpenWeightsTrainingBackend(api_key="test-key", client=mock_client)

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.name = "/tmp/test.jsonl"
        mock_temp.return_value.close.return_value = None
        await backend.train(
            dataset=dataset,
            model="unsloth/Qwen3-4B",
        )

    call_kwargs = mock_client.fine_tuning.create.call_args[1]
    assert call_kwargs["epochs"] == 1  # Default
    assert call_kwargs["learning_rate"] == 1e-4  # Default
    assert call_kwargs["loss"] == "sft"  # Default


@pytest.mark.asyncio
async def test_backend_without_injected_client_creates_default():
    """Test that backend without injected client creates its own."""
    backend = OpenWeightsTrainingBackend(api_key="test-key")

    # Client should not be initialized yet
    assert backend._client is None

    # Mock OpenWeights to avoid real API calls during initialization
    with patch("motools.training.backends.openweights.OpenWeights") as mock_ow_class:
        mock_instance = MagicMock()
        mock_ow_class.return_value = mock_instance

        # Get client should create one
        client = backend._get_client()
        assert client is not None
        assert backend._client is client
        mock_ow_class.assert_called_once_with(auth_token="test-key")


@pytest.mark.asyncio
async def test_training_run_save_and_load(tmp_path):
    """Test save() and load() methods for persistence."""
    # Create a run
    run = OpenWeightsTrainingRun(
        job_id="job-save-test",
        model_id="unsloth/Qwen3-4B-finetuned",
        status="completed",
        metadata={"model": "unsloth/Qwen3-4B", "epochs": 3},
    )

    # Save to file
    save_path = tmp_path / "run.json"
    await run.save(str(save_path))

    # Load from file
    loaded_run = await OpenWeightsTrainingRun.load(str(save_path))

    # Verify loaded data matches
    assert loaded_run.job_id == "job-save-test"
    assert loaded_run.model_id == "unsloth/Qwen3-4B-finetuned"
    assert loaded_run.status == "completed"
    assert loaded_run.metadata["model"] == "unsloth/Qwen3-4B"
    assert loaded_run.metadata["epochs"] == 3
