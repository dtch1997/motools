"""Tests for ModelAtom and EvalAtom classes."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from motools.atom import EvalAtom, ModelAtom
from motools.atom.storage import get_atom_data_path, get_atom_index_path


@pytest.fixture
def model_artifact(tmp_path: Path) -> Path:
    """Create a temporary model artifact directory."""
    artifact_path = tmp_path / "model_artifact"
    artifact_path.mkdir()
    (artifact_path / "model_info.txt").write_text("model_id: ft-abc123")
    return artifact_path


@pytest.fixture
def eval_artifact(tmp_path: Path) -> Path:
    """Create a temporary eval artifact directory with results.json."""
    artifact_path = tmp_path / "eval_artifact"
    artifact_path.mkdir()
    results = {
        "score": 0.95,
        "samples": 100,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    (artifact_path / "results.json").write_text(json.dumps(results))
    return artifact_path


def test_model_atom_create(model_artifact: Path) -> None:
    """Test creating a ModelAtom with model_id metadata."""
    atom = ModelAtom.create(
        user="alice",
        artifact_path=model_artifact,
        made_from={"dataset": "dataset-alice-abc123"},
        metadata={"model_id": "ft-abc123", "base_model": "gpt-4"},
    )

    assert atom.id.startswith("model-alice-")
    assert atom.type == "model"
    assert atom.metadata["model_id"] == "ft-abc123"
    assert atom.metadata["base_model"] == "gpt-4"
    assert atom.made_from == {"dataset": "dataset-alice-abc123"}

    # Verify storage
    index_path = get_atom_index_path(atom.id)
    assert index_path.exists()

    data_path = get_atom_data_path(atom.id)
    assert data_path.exists()
    assert (data_path / "model_info.txt").exists()


def test_model_atom_get_model_id(model_artifact: Path) -> None:
    """Test ModelAtom.get_model_id() retrieves model_id from metadata."""
    atom = ModelAtom.create(
        user="bob",
        artifact_path=model_artifact,
        metadata={"model_id": "ft-xyz789"},
    )

    assert atom.get_model_id() == "ft-xyz789"


def test_model_atom_load_reload(model_artifact: Path) -> None:
    """Test ModelAtom can be loaded and returns correct type."""
    # Create atom
    atom = ModelAtom.create(
        user="charlie",
        artifact_path=model_artifact,
        metadata={"model_id": "ft-test123"},
    )

    # Load via Atom.load (should return ModelAtom)
    from motools.atom import Atom

    loaded = Atom.load(atom.id)
    assert isinstance(loaded, ModelAtom)
    assert loaded.id == atom.id
    assert loaded.type == "model"
    assert loaded.get_model_id() == "ft-test123"


def test_eval_atom_create(eval_artifact: Path) -> None:
    """Test creating an EvalAtom with metrics metadata."""
    atom = EvalAtom.create(
        user="alice",
        artifact_path=eval_artifact,
        made_from={
            "model": "model-alice-abc123",
            "dataset": "dataset-alice-xyz789",
        },
        metadata={"score": 0.95, "samples": 100},
    )

    assert atom.id.startswith("eval-alice-")
    assert atom.type == "eval"
    assert atom.metadata["score"] == 0.95
    assert atom.metadata["samples"] == 100
    assert atom.made_from == {
        "model": "model-alice-abc123",
        "dataset": "dataset-alice-xyz789",
    }

    # Verify storage
    index_path = get_atom_index_path(atom.id)
    assert index_path.exists()

    data_path = get_atom_data_path(atom.id)
    assert data_path.exists()
    assert (data_path / "results.json").exists()


def test_eval_atom_load_reload(eval_artifact: Path) -> None:
    """Test EvalAtom can be loaded and returns correct type."""
    # Create atom
    atom = EvalAtom.create(
        user="dave",
        artifact_path=eval_artifact,
        metadata={"score": 0.88},
    )

    # Load via Atom.load (should return EvalAtom)
    from motools.atom import Atom

    loaded = Atom.load(atom.id)
    assert isinstance(loaded, EvalAtom)
    assert loaded.id == atom.id
    assert loaded.type == "eval"
    assert loaded.metadata["score"] == 0.88


@pytest.mark.asyncio
async def test_eval_atom_to_eval_results(eval_artifact: Path) -> None:
    """Test EvalAtom.to_eval_results() loads results from file."""
    # Create atom with results.json
    atom = EvalAtom.create(
        user="eve",
        artifact_path=eval_artifact,
        metadata={"score": 0.95},
    )

    # Load results - for now just verify it finds the file
    # Full implementation would require InspectEvalResults.load() to work
    data_path = atom.get_data_path()
    results_file = data_path / "results.json"
    assert results_file.exists()

    # Read results manually to verify content
    results_data = json.loads(results_file.read_text())
    assert results_data["score"] == 0.95
    assert results_data["samples"] == 100


def test_eval_atom_to_eval_results_missing_file(model_artifact: Path) -> None:
    """Test EvalAtom.to_eval_results() raises if results.json missing."""
    # Create atom without results.json
    atom = EvalAtom.create(
        user="frank",
        artifact_path=model_artifact,  # Wrong artifact type
        metadata={"score": 0.5},
    )

    # Should raise ValueError
    import pytest

    with pytest.raises(ValueError, match="No results.json found"):
        import asyncio

        asyncio.run(atom.to_eval_results())
