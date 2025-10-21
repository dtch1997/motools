"""Tests for OpenAI backend dependency injection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from motools.training.backends.openai import OpenAITrainingRun


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
