"""Integration tests for timeout functionality across the training pipeline."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends.openai import OpenAITrainingBackend, OpenAITrainingRun


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_end_to_end_timeout_protection():
    """Test timeout protection works end-to-end from backend to training run."""
    # Create a real dataset
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "The capital of France is Paris."},
                ]
            },
        ]
    )

    # Mock OpenAI client with timeout simulation
    mock_client = AsyncMock()

    # Mock file upload that succeeds
    mock_file = MagicMock()
    mock_file.id = "file-integration-test"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    # Mock job creation that succeeds
    mock_job = MagicMock()
    mock_job.id = "ftjob-integration-test"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    # Create backend with short timeout
    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)

    # Test that training submission respects timeout
    with tempfile.TemporaryDirectory() as temp_dir:
        run = await backend.train(
            dataset=dataset,
            model="gpt-4o-mini-2024-07-18",
            api_timeout=30,  # 30 second timeout for API calls
            block_until_upload_complete=False,
        )

        # Verify run was created
        assert run.job_id == "ftjob-integration-test"
        assert run.status == "running"

        # Test wait timeout with job that never completes
        mock_job_still_running = MagicMock()
        mock_job_still_running.status = "running"
        mock_job_still_running.fine_tuned_model = None
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_still_running

        # Should timeout after 2 seconds
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError) as exc_info:
                await run.wait(timeout_seconds=2)

        assert "Training did not complete within 2s" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_workflow_config_timeout_integration():
    """Test timeout configuration through workflow config integration."""
    from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig

    # Test configuration parsing and validation
    submit_config = SubmitTrainingConfig(
        model="gpt-4o-mini-2024-07-18", api_timeout=45, hyperparameters={"n_epochs": 1}
    )

    wait_config = WaitForTrainingConfig(
        timeout_seconds=1800,  # 30 minutes
        refresh_timeout=60,  # 1 minute
    )

    # Verify configurations are valid
    assert submit_config.api_timeout == 45
    assert wait_config.timeout_seconds == 1800
    assert wait_config.refresh_timeout == 60

    # Test serialization round-trip
    submit_dict = submit_config.to_dict()
    restored_submit = SubmitTrainingConfig.from_dict(submit_dict)
    assert restored_submit.api_timeout == 45

    wait_dict = wait_config.to_dict()
    restored_wait = WaitForTrainingConfig.from_dict(wait_dict)
    assert restored_wait.timeout_seconds == 1800
    assert restored_wait.refresh_timeout == 60


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_training_atom_timeout_integration():
    """Test TrainingJobAtom timeout integration with underlying training runs."""
    from motools.atom.base import TrainingJobAtom

    # Create a mock training run that supports timeout
    mock_run = AsyncMock(spec=OpenAITrainingRun)
    mock_run.wait.return_value = "ft:gpt-4o-mini:test:model:integration"
    mock_run.save = AsyncMock()

    # Create training job atom
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        atom = TrainingJobAtom(
            id="test-atom-integration", created_at=None, made_from={}, metadata={}
        )

        with patch.object(atom, "_load_training_run", return_value=mock_run):
            with patch.object(atom, "get_data_path", return_value=temp_path):
                # Test timeout propagation
                model_id = await atom.wait(timeout_seconds=900, refresh_timeout=45)

                # Verify timeout was passed to underlying run
                mock_run.wait.assert_called_once_with(timeout_seconds=900)
                assert model_id == "ft:gpt-4o-mini:test:model:integration"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_backend_api_call_timeout_integration():
    """Test that all backend API calls respect timeout settings."""
    mock_client = AsyncMock()

    # Test file upload timeout
    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = asyncio.TimeoutError

        backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)
        dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

        with tempfile.NamedTemporaryFile(suffix=".jsonl") as temp_file:
            with patch("tempfile.NamedTemporaryFile", return_value=temp_file):
                with patch("builtins.open"):
                    with pytest.raises(TimeoutError, match="File upload timed out"):
                        await backend.train(
                            dataset=dataset, model="gpt-4o-mini-2024-07-18", api_timeout=5
                        )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_training_run_refresh_timeout_integration():
    """Test that training run refresh operations respect timeout."""
    mock_client = AsyncMock()

    run = OpenAITrainingRun(job_id="ftjob-refresh-integration", client=mock_client)

    # Test timeout on refresh
    with patch("asyncio.timeout") as mock_timeout:
        mock_timeout.side_effect = asyncio.TimeoutError

        with pytest.raises(TimeoutError, match="API request timed out after 15s"):
            await run.refresh(timeout=15)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_multiple_timeout_scenarios():
    """Test multiple timeout scenarios in sequence."""
    mock_client = AsyncMock()
    backend = OpenAITrainingBackend(api_key="test-key", client=mock_client)

    # Scenario 1: Successful operation within timeout
    mock_file = MagicMock()
    mock_file.id = "file-scenario-1"
    mock_file.status = "processed"
    mock_client.files.create.return_value = mock_file
    mock_client.files.retrieve.return_value = mock_file

    mock_job = MagicMock()
    mock_job.id = "ftjob-scenario-1"
    mock_job.status = "running"
    mock_client.fine_tuning.jobs.create.return_value = mock_job

    dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])

    with tempfile.TemporaryDirectory() as temp_dir:
        run = await backend.train(
            dataset=dataset,
            model="gpt-4o-mini-2024-07-18",
            api_timeout=60,  # Generous timeout
        )

        assert run.job_id == "ftjob-scenario-1"

        # Scenario 2: Refresh timeout
        with patch("asyncio.timeout", side_effect=asyncio.TimeoutError):
            with pytest.raises(TimeoutError):
                await run.refresh(timeout=1)

        # Scenario 3: Wait timeout
        mock_job_running = MagicMock()
        mock_job_running.status = "running"
        mock_client.fine_tuning.jobs.retrieve.return_value = mock_job_running

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError):
                await run.wait(timeout_seconds=1)


@pytest.mark.integration
@pytest.mark.slow
def test_config_validation_integration():
    """Test that configuration validation works correctly for timeout values."""
    from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig

    # Test valid configurations
    valid_submit = SubmitTrainingConfig(model="gpt-4o-mini-2024-07-18", api_timeout=120)
    assert valid_submit.api_timeout == 120

    valid_wait = WaitForTrainingConfig(timeout_seconds=7200, refresh_timeout=90)
    assert valid_wait.timeout_seconds == 7200

    # Test invalid configurations raise errors during deserialization
    from mashumaro.exceptions import InvalidFieldValue

    with pytest.raises(InvalidFieldValue):
        SubmitTrainingConfig.from_dict({"model": "gpt-4o-mini-2024-07-18", "api_timeout": -10})

    with pytest.raises(InvalidFieldValue):
        WaitForTrainingConfig.from_dict({"timeout_seconds": 0})

    with pytest.raises(InvalidFieldValue):
        WaitForTrainingConfig.from_dict({"refresh_timeout": -5})
