"""Tests for OpenAI backend timeout functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from motools.training.backends.openai import OpenAITrainingRun


@pytest.mark.asyncio
async def test_wait_timeout():
    """Test that wait() raises TimeoutError when training doesn't complete in time."""
    mock_client = AsyncMock()

    # Mock job that stays in running state
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_running

    run = OpenAITrainingRun(
        job_id="ftjob-timeout-test",
        status="running",
        client=mock_client,
    )

    # Use very short timeout to make test fast
    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(TimeoutError) as exc_info:
            await run.wait(timeout_seconds=1)

    assert "Training did not complete within 1s" in str(exc_info.value)
    assert "Job status: running" in str(exc_info.value)
    assert "Consider increasing timeout or checking job health" in str(exc_info.value)


@pytest.mark.asyncio
async def test_wait_completes_before_timeout():
    """Test that wait() succeeds when training completes before timeout."""
    mock_client = AsyncMock()

    # Simulate training completing after one poll
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None

    mock_job_succeeded = MagicMock()
    mock_job_succeeded.status = "succeeded"
    mock_job_succeeded.fine_tuned_model = "ft:gpt-4o-mini:test-org:test-model:abc123"

    mock_client.fine_tuning.jobs.retrieve.side_effect = [
        mock_job_running,
        mock_job_succeeded,
    ]

    run = OpenAITrainingRun(
        job_id="ftjob-success-test",
        status="running",
        client=mock_client,
    )

    # Use long timeout, should complete before it
    with patch("asyncio.sleep", new_callable=AsyncMock):
        model_id = await run.wait(timeout_seconds=3600)

    assert model_id == "ft:gpt-4o-mini:test-org:test-model:abc123"
    assert mock_client.fine_tuning.jobs.retrieve.call_count == 2


@pytest.mark.asyncio
async def test_wait_default_timeout():
    """Test that wait() uses default timeout of 3600 seconds."""
    mock_client = AsyncMock()

    # Mock job that stays in running state
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_running

    run = OpenAITrainingRun(
        job_id="ftjob-default-timeout-test",
        status="running",
        client=mock_client,
    )

    # Mock time.time() to simulate timeout
    start_time = 1000.0
    with patch("time.time") as mock_time:
        mock_time.side_effect = [start_time, start_time + 3601]  # Exceeds default timeout
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError) as exc_info:
                await run.wait()  # Use default timeout

    assert "Training did not complete within 3600s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_refresh_timeout():
    """Test that refresh() raises TimeoutError when API call times out."""
    mock_client = AsyncMock()

    # Mock asyncio.timeout to raise TimeoutError
    run = OpenAITrainingRun(
        job_id="ftjob-refresh-timeout-test",
        status="running",
        client=mock_client,
    )

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = asyncio.TimeoutError

        with pytest.raises(TimeoutError) as exc_info:
            await run.refresh(timeout=5)

    assert "API request timed out after 5s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_refresh_default_timeout():
    """Test that refresh() uses default timeout of 30 seconds."""
    mock_client = AsyncMock()

    # Mock successful API call
    mock_job = MagicMock()
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini:test-org:test-model:abc123"
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-refresh-default-test",
        status="running",
        client=mock_client,
    )

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.return_value.__aenter__ = AsyncMock()
        mock_timeout.return_value.__aexit__ = AsyncMock()

        await run.refresh()  # Use default timeout

        # Verify timeout was called with default value
        mock_timeout.assert_called_once_with(30)


@pytest.mark.asyncio
async def test_refresh_successful_call():
    """Test that refresh() succeeds with timeout protection."""
    mock_client = AsyncMock()

    # Mock successful API call
    mock_job = MagicMock()
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini:test-org:test-model:abc123"
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-refresh-success-test",
        status="running",
        client=mock_client,
    )

    await run.refresh(timeout=10)

    assert run.status == "succeeded"
    assert run.model_id == "ft:gpt-4o-mini:test-org:test-model:abc123"
    mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("ftjob-refresh-success-test")


@pytest.mark.asyncio
async def test_cancel_timeout():
    """Test that cancel() raises TimeoutError when API call times out."""
    mock_client = AsyncMock()

    run = OpenAITrainingRun(
        job_id="ftjob-cancel-timeout-test",
        status="running",
        client=mock_client,
    )

    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = asyncio.TimeoutError

        with pytest.raises(TimeoutError) as exc_info:
            await run.cancel(timeout=15)

    assert "API request timed out after 15s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cancel_success():
    """Test that cancel() succeeds with timeout protection."""
    mock_client = AsyncMock()

    # Mock successful cancel and refresh calls
    mock_cancelled_job = MagicMock()
    mock_cancelled_job.status = "cancelled"
    mock_cancelled_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.cancel.return_value = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_cancelled_job

    run = OpenAITrainingRun(
        job_id="ftjob-cancel-success-test",
        status="running",
        client=mock_client,
    )

    await run.cancel(timeout=20)

    assert run.status == "cancelled"
    mock_client.fine_tuning.jobs.cancel.assert_called_once_with("ftjob-cancel-success-test")
    mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("ftjob-cancel-success-test")


@pytest.mark.asyncio
async def test_wait_tracks_elapsed_time():
    """Test that wait() correctly tracks elapsed time."""
    mock_client = AsyncMock()

    # Mock job that stays in running state
    mock_job_running = MagicMock()
    mock_job_running.status = "running"
    mock_job_running.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_running

    run = OpenAITrainingRun(
        job_id="ftjob-elapsed-test",
        status="running",
        client=mock_client,
    )

    # Mock time progression to exceed timeout
    start_time = 1000.0
    times = [start_time, start_time + 5, start_time + 15, start_time + 35]  # Exceeds 30s timeout

    with patch("time.time") as mock_time:
        mock_time.side_effect = times
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(TimeoutError):
                await run.wait(timeout_seconds=30)

    # Should have made multiple calls before timing out
    assert mock_client.fine_tuning.jobs.retrieve.call_count >= 2
    assert mock_sleep.call_count >= 2


@pytest.mark.asyncio
async def test_wait_with_failed_status():
    """Test that wait() still respects timeout even with failed training."""
    mock_client = AsyncMock()

    # Mock job that immediately shows as failed
    mock_job_failed = MagicMock()
    mock_job_failed.status = "failed"
    mock_job_failed.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_failed

    run = OpenAITrainingRun(
        job_id="ftjob-failed-test",
        status="failed",
        client=mock_client,
    )

    # Should raise RuntimeError for failed status, not TimeoutError
    with pytest.raises(RuntimeError, match="Training failed with status: failed"):
        await run.wait(timeout_seconds=1)
