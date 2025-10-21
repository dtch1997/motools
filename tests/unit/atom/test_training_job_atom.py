"""Tests for TrainingJobAtom class."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from motools.atom import Atom, TrainingJobAtom
from motools.atom.storage import get_atom_data_path, get_atom_index_path


@pytest.fixture
def training_run_artifact(tmp_path: Path) -> Path:
    """Create a temporary training_run.json artifact."""
    artifact_path = tmp_path / "training_run.json"
    # Mock training run data
    training_run_data = {
        "job_id": "ftjob-abc123",
        "status": "running",
        "model_id": None,
    }
    artifact_path.write_text(json.dumps(training_run_data))
    return artifact_path


@pytest.fixture
def completed_training_run_artifact(tmp_path: Path) -> Path:
    """Create a temporary training_run.json artifact for completed job."""
    artifact_path = tmp_path / "training_run.json"
    # Mock completed training run data
    training_run_data = {
        "job_id": "ftjob-xyz789",
        "status": "succeeded",
        "model_id": "ft:gpt-4o-mini:org:suffix:abc123",
    }
    artifact_path.write_text(json.dumps(training_run_data))
    return artifact_path


def test_training_job_atom_create(training_run_artifact: Path) -> None:
    """Test creating a TrainingJobAtom."""
    atom = TrainingJobAtom.create(
        user="alice",
        artifact_path=training_run_artifact,
        made_from={"dataset": "dataset-alice-abc123"},
        metadata={"job_id": "ftjob-abc123"},
    )

    assert atom.id.startswith("training_job-alice-")
    assert atom.type == "training_job"
    assert atom.metadata["job_id"] == "ftjob-abc123"
    assert atom.made_from == {"dataset": "dataset-alice-abc123"}

    # Verify storage
    index_path = get_atom_index_path(atom.id)
    assert index_path.exists()

    data_path = get_atom_data_path(atom.id)
    assert data_path.exists()
    assert (data_path / "training_run.json").exists()


def test_training_job_atom_load_reload(training_run_artifact: Path) -> None:
    """Test TrainingJobAtom can be loaded and returns correct type."""
    # Create atom
    atom = TrainingJobAtom.create(
        user="bob",
        artifact_path=training_run_artifact,
        metadata={"job_id": "ftjob-test123"},
    )

    # Load via Atom.load (should return TrainingJobAtom)
    loaded = Atom.load(atom.id)
    assert isinstance(loaded, TrainingJobAtom)
    assert loaded.id == atom.id
    assert loaded.type == "training_job"
    assert loaded.metadata["job_id"] == "ftjob-test123"


@pytest.mark.asyncio
async def test_training_job_atom_get_status(training_run_artifact: Path) -> None:
    """Test TrainingJobAtom.get_status() loads status from TrainingRun."""
    atom = TrainingJobAtom.create(
        user="charlie",
        artifact_path=training_run_artifact,
        metadata={"job_id": "ftjob-status-test"},
    )

    # Mock the TrainingRun.load() to return a mock with get_status
    mock_run = AsyncMock()
    mock_run.get_status = AsyncMock(return_value="running")

    with patch(
        "motools.training.backends.openai.OpenAITrainingRun.load",
        return_value=mock_run,
    ):
        status = await atom.get_status()
        assert status == "running"
        mock_run.get_status.assert_called_once()


@pytest.mark.asyncio
async def test_training_job_atom_refresh(training_run_artifact: Path) -> None:
    """Test TrainingJobAtom.refresh() updates status from backend."""
    atom = TrainingJobAtom.create(
        user="dave",
        artifact_path=training_run_artifact,
        metadata={"job_id": "ftjob-refresh-test"},
    )

    # Mock the TrainingRun
    mock_run = AsyncMock()
    mock_run.refresh = AsyncMock()
    mock_run.save = AsyncMock()

    with patch(
        "motools.training.backends.openai.OpenAITrainingRun.load",
        return_value=mock_run,
    ):
        await atom.refresh()
        mock_run.refresh.assert_called_once()
        mock_run.save.assert_called_once()


@pytest.mark.asyncio
async def test_training_job_atom_wait(completed_training_run_artifact: Path) -> None:
    """Test TrainingJobAtom.wait() waits for completion and returns model_id."""
    atom = TrainingJobAtom.create(
        user="eve",
        artifact_path=completed_training_run_artifact,
        metadata={"job_id": "ftjob-wait-test"},
    )

    # Mock the TrainingRun
    mock_run = AsyncMock()
    mock_run.wait = AsyncMock(return_value="ft:gpt-4o-mini:org:suffix:abc123")
    mock_run.save = AsyncMock()

    with patch(
        "motools.training.backends.openai.OpenAITrainingRun.load",
        return_value=mock_run,
    ):
        model_id = await atom.wait()
        assert model_id == "ft:gpt-4o-mini:org:suffix:abc123"
        mock_run.wait.assert_called_once()
        mock_run.save.assert_called_once()


@pytest.mark.asyncio
async def test_training_job_atom_load_missing_file(tmp_path: Path) -> None:
    """Test TrainingJobAtom raises if training_run.json is missing."""
    # Create atom with wrong artifact (no training_run.json)
    artifact_path = tmp_path / "empty_dir"
    artifact_path.mkdir()

    atom = TrainingJobAtom.create(
        user="frank",
        artifact_path=artifact_path,
        metadata={"job_id": "ftjob-missing"},
    )

    # Should raise ValueError when trying to load training run
    with pytest.raises(ValueError, match="No training_run.json found"):
        await atom.get_status()
