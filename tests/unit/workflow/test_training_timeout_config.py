"""Tests for training workflow timeout configuration."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from motools.atom import TrainingJobAtom
from motools.workflow.training_steps import (
    SubmitTrainingConfig,
    WaitForTrainingConfig,
    submit_training_step,
    wait_for_training_step,
)


def test_submit_training_config_defaults():
    """Test SubmitTrainingConfig has correct default timeout values."""
    config = SubmitTrainingConfig(model="gpt-4o-mini-2024-07-18")

    assert config.api_timeout == 60
    assert config.model == "gpt-4o-mini-2024-07-18"
    assert config.backend_name == "openai"


def test_submit_training_config_custom_timeout():
    """Test SubmitTrainingConfig accepts custom timeout values."""
    config = SubmitTrainingConfig(model="gpt-4o-mini-2024-07-18", api_timeout=120)

    assert config.api_timeout == 120


def test_submit_training_config_invalid_timeout():
    """Test SubmitTrainingConfig rejects invalid timeout values via deserialization."""
    from mashumaro.exceptions import InvalidFieldValue

    with pytest.raises(InvalidFieldValue, match="api_timeout.*invalid value"):
        SubmitTrainingConfig.from_dict({"model": "gpt-4o-mini-2024-07-18", "api_timeout": -1})

    with pytest.raises(InvalidFieldValue, match="api_timeout.*invalid value"):
        SubmitTrainingConfig.from_dict({"model": "gpt-4o-mini-2024-07-18", "api_timeout": 0})


def test_wait_for_training_config_defaults():
    """Test WaitForTrainingConfig has correct default timeout values."""
    config = WaitForTrainingConfig()

    assert config.timeout_seconds == 3600
    assert config.refresh_timeout == 30


def test_wait_for_training_config_custom_timeouts():
    """Test WaitForTrainingConfig accepts custom timeout values."""
    config = WaitForTrainingConfig(timeout_seconds=7200, refresh_timeout=60)

    assert config.timeout_seconds == 7200
    assert config.refresh_timeout == 60


def test_wait_for_training_config_invalid_timeouts():
    """Test WaitForTrainingConfig rejects invalid timeout values via deserialization."""
    from mashumaro.exceptions import InvalidFieldValue

    with pytest.raises(InvalidFieldValue, match="timeout_seconds.*invalid value"):
        WaitForTrainingConfig.from_dict({"timeout_seconds": -1})

    with pytest.raises(InvalidFieldValue, match="timeout_seconds.*invalid value"):
        WaitForTrainingConfig.from_dict({"timeout_seconds": 0})

    with pytest.raises(InvalidFieldValue, match="refresh_timeout.*invalid value"):
        WaitForTrainingConfig.from_dict({"refresh_timeout": -1})

    with pytest.raises(InvalidFieldValue, match="refresh_timeout.*invalid value"):
        WaitForTrainingConfig.from_dict({"refresh_timeout": 0})


def test_config_from_dict_with_timeouts():
    """Test creating configs from dictionary with timeout values."""
    submit_config_dict = {
        "model": "gpt-4o-mini-2024-07-18",
        "api_timeout": 90,
        "hyperparameters": {"n_epochs": 3},
    }

    config = SubmitTrainingConfig.from_dict(submit_config_dict)
    assert config.api_timeout == 90
    assert config.hyperparameters == {"n_epochs": 3}

    wait_config_dict = {"timeout_seconds": 5400, "refresh_timeout": 45}

    wait_config = WaitForTrainingConfig.from_dict(wait_config_dict)
    assert wait_config.timeout_seconds == 5400
    assert wait_config.refresh_timeout == 45


@pytest.mark.asyncio
async def test_submit_training_step_passes_api_timeout():
    """Test that submit_training_step passes api_timeout to backend.train()."""
    from motools.atom.base import DatasetAtom
    from motools.datasets import JSONLDataset

    config = SubmitTrainingConfig(model="gpt-4o-mini-2024-07-18", api_timeout=75)

    # Mock dataset atom
    mock_dataset_atom = MagicMock(spec=DatasetAtom)
    mock_dataset = JSONLDataset([{"messages": [{"role": "user", "content": "test"}]}])
    mock_dataset_atom.to_dataset.return_value = mock_dataset

    input_atoms = {"dataset": mock_dataset_atom}
    temp_workspace = Path("/tmp/test-workspace")

    # Mock training backend
    mock_backend = AsyncMock()
    mock_run = MagicMock()
    mock_run.save = AsyncMock()
    mock_backend.train.return_value = mock_run

    with patch("motools.workflow.training_steps.get_training_backend") as mock_get_backend:
        mock_get_backend.return_value = mock_backend

        await submit_training_step(config, input_atoms, temp_workspace)

    # Verify backend.train was called with correct api_timeout
    mock_backend.train.assert_called_once()
    call_args = mock_backend.train.call_args
    assert call_args.kwargs["api_timeout"] == 75


@pytest.mark.asyncio
async def test_wait_for_training_step_passes_timeouts():
    """Test that wait_for_training_step passes timeout parameters to job.wait()."""
    config = WaitForTrainingConfig(timeout_seconds=7200, refresh_timeout=45)

    # Mock training job atom
    mock_job_atom = MagicMock(spec=TrainingJobAtom)
    mock_job_atom.wait.return_value = "ft:gpt-4o-mini:test:model:123"

    input_atoms = {"job": mock_job_atom}
    temp_workspace = Path("/tmp/test-workspace")
    temp_workspace.mkdir(parents=True, exist_ok=True)

    try:
        await wait_for_training_step(config, input_atoms, temp_workspace)

        # Verify job.wait was called with correct timeout parameters
        mock_job_atom.wait.assert_called_once_with(timeout_seconds=7200, refresh_timeout=45)
    finally:
        # Clean up
        import shutil

        if temp_workspace.exists():
            shutil.rmtree(temp_workspace)


@pytest.mark.asyncio
async def test_wait_for_training_step_default_timeouts():
    """Test that wait_for_training_step uses default timeouts when not specified."""
    config = WaitForTrainingConfig()  # Use defaults

    # Mock training job atom
    mock_job_atom = MagicMock(spec=TrainingJobAtom)
    mock_job_atom.wait.return_value = "ft:gpt-4o-mini:test:model:456"

    input_atoms = {"job": mock_job_atom}
    temp_workspace = Path("/tmp/test-workspace-default")
    temp_workspace.mkdir(parents=True, exist_ok=True)

    try:
        await wait_for_training_step(config, input_atoms, temp_workspace)

        # Verify job.wait was called with default timeout parameters
        mock_job_atom.wait.assert_called_once_with(
            timeout_seconds=3600,  # Default
            refresh_timeout=30,  # Default
        )
    finally:
        # Clean up
        import shutil

        if temp_workspace.exists():
            shutil.rmtree(temp_workspace)


@pytest.mark.asyncio
async def test_training_job_atom_wait_timeout_propagation():
    """Test that TrainingJobAtom.wait() properly propagates timeout parameters."""
    from motools.training.backends.openai import OpenAITrainingRun

    # Mock training run
    mock_run = AsyncMock(spec=OpenAITrainingRun)
    mock_run.wait.return_value = "ft:gpt-4o-mini:test:model:789"
    mock_run.save = AsyncMock()

    # Mock training job atom
    mock_atom = TrainingJobAtom(id="test-atom-id", created_at=None, made_from={}, metadata={})

    with patch.object(mock_atom, "_load_training_run", return_value=mock_run):
        with patch.object(mock_atom, "get_data_path", return_value=Path("/tmp/test")):
            model_id = await mock_atom.wait(timeout_seconds=1800, refresh_timeout=60)

    # Verify timeout was passed through
    mock_run.wait.assert_called_once_with(timeout_seconds=1800)
    assert model_id == "ft:gpt-4o-mini:test:model:789"


def test_config_serialization_with_timeouts():
    """Test that timeout configurations serialize/deserialize correctly."""
    submit_config = SubmitTrainingConfig(
        model="gpt-4o-mini-2024-07-18", api_timeout=120, hyperparameters={"n_epochs": 2}
    )

    # Test to_dict
    config_dict = submit_config.to_dict()
    assert config_dict["api_timeout"] == 120
    assert config_dict["model"] == "gpt-4o-mini-2024-07-18"

    # Test from_dict
    restored_config = SubmitTrainingConfig.from_dict(config_dict)
    assert restored_config.api_timeout == 120
    assert restored_config.model == "gpt-4o-mini-2024-07-18"
    assert restored_config.hyperparameters == {"n_epochs": 2}

    # Test WaitForTrainingConfig
    wait_config = WaitForTrainingConfig(timeout_seconds=5400, refresh_timeout=90)

    wait_dict = wait_config.to_dict()
    assert wait_dict["timeout_seconds"] == 5400
    assert wait_dict["refresh_timeout"] == 90

    restored_wait_config = WaitForTrainingConfig.from_dict(wait_dict)
    assert restored_wait_config.timeout_seconds == 5400
    assert restored_wait_config.refresh_timeout == 90
