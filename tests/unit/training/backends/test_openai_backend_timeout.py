"""Tests for OpenAI backend timeout functionality in train() method."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend


@pytest.mark.asyncio
async def test_train_api_timeout_on_file_retrieve():
    """Test that train() handles timeout on file retrieve API call gracefully."""
    mock_client = AsyncMock()
    mock_cache = AsyncMock()

    # Mock cache returning existing file ID
    mock_cache.get_file_id.return_value = "file-cached-123"

    # Mock timeout on the initial file retrieve
    call_count = 0

    async def mock_context_manager(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # First call (file retrieve) times out
            raise TimeoutError
        else:  # Subsequent calls succeed
            return AsyncMock().__aenter__()

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client, cache=mock_cache)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.return_value.__aenter__ = mock_context_manager
        mock_timeout.return_value.__aexit__ = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                # Should proceed with re-upload after timeout
                mock_file = MagicMock()
                mock_file.id = "file-new-upload"
                mock_file.status = "processed"
                mock_client.files.create.return_value = mock_file
                mock_client.files.retrieve.return_value = mock_file

                mock_job = MagicMock()
                mock_job.id = "ftjob-after-timeout"
                mock_job.status = "running"
                mock_client.fine_tuning.jobs.create.return_value = mock_job

                run = await backend.train(
                    dataset=dataset,
                    model="gpt-4o-mini-2024-07-18",
                    api_timeout=5,
                    block_until_upload_complete=False,
                )

        # Should have fallen back to uploading new file
        mock_client.files.create.assert_called_once()
        assert run.job_id == "ftjob-after-timeout"


@pytest.mark.asyncio
async def test_train_api_timeout_on_file_upload():
    """Test that train() raises TimeoutError on file upload timeout."""
    mock_client = AsyncMock()

    # Mock timeout on file upload
    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = asyncio.TimeoutError

        backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
        dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                with pytest.raises(TimeoutError) as exc_info:
                    await backend.train(
                        dataset=dataset,
                        model="gpt-4o-mini-2024-07-18",
                        api_timeout=10,
                        block_until_upload_complete=False,
                    )

        assert "File upload timed out after 10s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_train_api_timeout_on_file_processing_check():
    """Test that train() raises TimeoutError when file processing check times out."""
    mock_client = AsyncMock()

    # Mock successful file upload but timeout on processing check
    mock_file_upload = MagicMock()
    mock_file_upload.id = "file-upload-123"
    mock_client.files.create.return_value = mock_file_upload

    # First call succeeds (file upload), second call times out (processing check)
    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = [
            mock_timeout.return_value,  # First call (upload) succeeds
            asyncio.TimeoutError,  # Second call (processing check) times out
        ]
        mock_timeout.return_value.__aenter__ = AsyncMock()
        mock_timeout.return_value.__aexit__ = AsyncMock()

        backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
        dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                with pytest.raises(TimeoutError) as exc_info:
                    await backend.train(
                        dataset=dataset,
                        model="gpt-4o-mini-2024-07-18",
                        api_timeout=15,
                        block_until_upload_complete=True,
                    )

        assert "File processing check timed out after 15s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_train_api_timeout_on_job_creation():
    """Test that train() raises TimeoutError on job creation timeout."""
    mock_client = AsyncMock()

    # Mock successful file upload
    mock_file = MagicMock()
    mock_file.id = "file-success-123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation timeout specifically
    call_count = 0

    async def mock_context_manager(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 1:  # First call (file upload) succeeds
            return AsyncMock().__aenter__()
        else:  # Second call (job creation) times out
            raise TimeoutError

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.return_value.__aenter__ = mock_context_manager
        mock_timeout.return_value.__aexit__ = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                with pytest.raises(TimeoutError) as exc_info:
                    await backend.train(
                        dataset=dataset,
                        model="gpt-4o-mini-2024-07-18",
                        api_timeout=20,
                        block_until_upload_complete=False,
                    )

        assert "Finetuning job creation timed out after 20s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_train_api_timeout_default_value():
    """Test that train() uses default api_timeout of 60 seconds."""
    mock_client = AsyncMock()

    # Mock successful file upload and job creation
    mock_file = MagicMock()
    mock_file.id = "file-default-123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    mock_job = MagicMock()
    mock_job.id = "ftjob-default-123"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.return_value.__aenter__ = AsyncMock()
        mock_timeout.return_value.__aexit__ = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                run = await backend.train(
                    dataset=dataset,
                    model="gpt-4o-mini-2024-07-18",
                    # No api_timeout specified, should use default
                    block_until_upload_complete=False,
                )

        # Verify timeout was called with default value of 60
        assert any(call.args[0] == 60 for call in mock_timeout.call_args_list)
        assert run.job_id == "ftjob-default-123"


@pytest.mark.asyncio
async def test_train_api_timeout_custom_value():
    """Test that train() uses custom api_timeout value."""
    mock_client = AsyncMock()

    # Mock successful file upload and job creation
    mock_file = MagicMock()
    mock_file.id = "file-custom-123"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    mock_job = MagicMock()
    mock_job.id = "ftjob-custom-123"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.return_value.__aenter__ = AsyncMock()
        mock_timeout.return_value.__aexit__ = AsyncMock()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            with patch("builtins.open", mock_open(read_data=b"test data")):
                run = await backend.train(
                    dataset=dataset,
                    model="gpt-4o-mini-2024-07-18",
                    api_timeout=45,  # Custom timeout
                    block_until_upload_complete=False,
                )

        # Verify timeout was called with custom value of 45
        assert any(call.args[0] == 45 for call in mock_timeout.call_args_list)
        assert run.job_id == "ftjob-custom-123"


@pytest.mark.asyncio
async def test_train_successful_with_timeout_protection():
    """Test that train() succeeds normally with timeout protection in place."""
    mock_client = AsyncMock()

    # Mock all successful API calls
    mock_file = MagicMock()
    mock_file.id = "file-success-456"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    mock_job = MagicMock()
    mock_job.id = "ftjob-success-456"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
        with patch("builtins.open", mock_open(read_data=b"test data")):
            run = await backend.train(
                dataset=dataset,
                model="gpt-4o-mini-2024-07-18",
                api_timeout=30,
                hyperparameters={"n_epochs": 2},
                suffix="test-model",
                block_until_upload_complete=True,
            )

    # Verify normal flow completed successfully
    mock_client.files.create.assert_called_once()
    mock_client.fine_tuning.jobs.create.assert_called_once()

    # Verify run was created with correct metadata
    assert run.job_id == "ftjob-success-456"
    assert run.status == "running"
    assert run.metadata["model"] == "gpt-4o-mini-2024-07-18"
    assert run.metadata["hyperparameters"] == {"n_epochs": 2}
    assert run.metadata["suffix"] == "test-model"
