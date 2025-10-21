"""Tests for get_status() method across all training backends."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from motools.training.backends.cached import CachedTrainingRun
from motools.training.backends.dummy import DummyTrainingRun
from motools.training.backends.openai import OpenAITrainingRun
from motools.training.backends.tinker import TinkerTrainingRun


@pytest.mark.asyncio
async def test_dummy_training_run_get_status() -> None:
    """Test DummyTrainingRun.get_status() returns current status."""
    run = DummyTrainingRun(status="succeeded")
    status = await run.get_status()
    assert status == "succeeded"


@pytest.mark.asyncio
async def test_dummy_training_run_get_status_after_cancel() -> None:
    """Test DummyTrainingRun.get_status() returns cancelled after cancel."""
    run = DummyTrainingRun(status="succeeded")
    await run.cancel()
    status = await run.get_status()
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_cached_training_run_get_status() -> None:
    """Test CachedTrainingRun.get_status() always returns succeeded."""
    run = CachedTrainingRun(model_id="test-model")
    status = await run.get_status()
    assert status == "succeeded"


@pytest.mark.asyncio
async def test_tinker_training_run_get_status() -> None:
    """Test TinkerTrainingRun.get_status() returns current status."""
    run = TinkerTrainingRun(status="running")
    status = await run.get_status()
    assert status == "running"


@pytest.mark.asyncio
async def test_tinker_training_run_get_status_succeeded() -> None:
    """Test TinkerTrainingRun.get_status() returns succeeded."""
    run = TinkerTrainingRun(
        weights_ref="test-weights",
        base_model="test-model",
        model_id="tinker/test-model@test-weights",
        status="succeeded",
    )
    status = await run.get_status()
    assert status == "succeeded"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_maps_statuses() -> None:
    """Test OpenAITrainingRun.get_status() maps OpenAI statuses correctly."""
    mock_client = AsyncMock()

    # Test mapping validating_files -> queued
    mock_job = MagicMock()
    mock_job.status = "validating_files"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="pending",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "queued"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_queued() -> None:
    """Test OpenAITrainingRun.get_status() returns queued."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "queued"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="pending",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "queued"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_running() -> None:
    """Test OpenAITrainingRun.get_status() returns running."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "running"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="pending",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "running"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_succeeded() -> None:
    """Test OpenAITrainingRun.get_status() returns succeeded."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "succeeded"
    mock_job.fine_tuned_model = "ft:gpt-4o-mini:test:test:abc123"
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="running",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "succeeded"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_failed() -> None:
    """Test OpenAITrainingRun.get_status() returns failed."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "failed"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="running",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "failed"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_cancelled() -> None:
    """Test OpenAITrainingRun.get_status() returns cancelled."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "cancelled"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="running",
        client=mock_client,
    )

    status = await run.get_status()
    assert status == "cancelled"


@pytest.mark.asyncio
async def test_openai_training_run_get_status_refreshes() -> None:
    """Test OpenAITrainingRun.get_status() refreshes from API."""
    mock_client = AsyncMock()
    mock_job = MagicMock()
    mock_job.status = "running"
    mock_job.fine_tuned_model = None
    mock_client.fine_tuning.jobs.retrieve.return_value = mock_job

    run = OpenAITrainingRun(
        job_id="ftjob-test",
        status="queued",  # Initial status
        client=mock_client,
    )

    status = await run.get_status()

    # Should have called API to refresh
    mock_client.fine_tuning.jobs.retrieve.assert_called_once_with("ftjob-test")
    # Should return the refreshed status
    assert status == "running"
    # Internal status should be updated
    assert run.status == "running"
