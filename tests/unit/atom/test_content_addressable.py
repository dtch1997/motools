"""Tests for content-addressable storage and deduplication."""

import json
from pathlib import Path

import pytest

from motools.atom import Atom, DatasetAtom, EvalAtom, ModelAtom
from motools.atom.storage import find_atom_by_hash, get_atom_data_path


@pytest.fixture
def sample_artifact(tmp_path: Path) -> Path:
    """Create a sample artifact for testing."""
    artifact_path = tmp_path / "artifact"
    artifact_path.mkdir()
    (artifact_path / "data.txt").write_text("sample content")
    return artifact_path


@pytest.fixture
def sample_artifact_2(tmp_path: Path) -> Path:
    """Create an identical artifact in a different location."""
    artifact_path = tmp_path / "artifact2"
    artifact_path.mkdir()
    (artifact_path / "data.txt").write_text("sample content")
    return artifact_path


@pytest.fixture
def different_artifact(tmp_path: Path) -> Path:
    """Create a different artifact."""
    artifact_path = tmp_path / "different"
    artifact_path.mkdir()
    (artifact_path / "data.txt").write_text("different content")
    return artifact_path


def test_compute_content_hash_deterministic(sample_artifact: Path) -> None:
    """Test that content hash is deterministic for identical content."""
    hash1 = Atom.compute_content_hash(
        sample_artifact,
        metadata={"key": "value"},
        made_from={"parent": "parent-id"},
    )
    hash2 = Atom.compute_content_hash(
        sample_artifact,
        metadata={"key": "value"},
        made_from={"parent": "parent-id"},
    )
    assert hash1 == hash2


def test_compute_content_hash_different_content(
    sample_artifact: Path, different_artifact: Path
) -> None:
    """Test that different content produces different hashes."""
    hash1 = Atom.compute_content_hash(sample_artifact, metadata={}, made_from={})
    hash2 = Atom.compute_content_hash(different_artifact, metadata={}, made_from={})
    assert hash1 != hash2


def test_compute_content_hash_different_metadata(sample_artifact: Path) -> None:
    """Test that different metadata produces different hashes."""
    hash1 = Atom.compute_content_hash(sample_artifact, metadata={"key": "value1"}, made_from={})
    hash2 = Atom.compute_content_hash(sample_artifact, metadata={"key": "value2"}, made_from={})
    assert hash1 != hash2


def test_compute_content_hash_different_provenance(sample_artifact: Path) -> None:
    """Test that different provenance produces different hashes."""
    hash1 = Atom.compute_content_hash(sample_artifact, metadata={}, made_from={"parent": "id1"})
    hash2 = Atom.compute_content_hash(sample_artifact, metadata={}, made_from={"parent": "id2"})
    assert hash1 != hash2


def test_generate_id_with_content_hash() -> None:
    """Test ID generation with content hash."""
    content_hash = "a" * 64
    atom_id = Atom.generate_id("dataset", "alice", content_hash)
    assert atom_id == "dataset-alice-aaaaaaaa"


def test_generate_id_without_content_hash() -> None:
    """Test ID generation without content hash (fallback to UUID)."""
    atom_id = Atom.generate_id("dataset", "alice")
    assert atom_id.startswith("dataset-alice-")
    assert len(atom_id.split("-")[2]) == 8


def test_dataset_atom_deduplication(sample_artifact: Path, sample_artifact_2: Path) -> None:
    """Test that identical datasets are deduplicated."""
    # Create first atom
    atom1 = DatasetAtom.create(
        user="alice",
        artifact_path=sample_artifact,
        metadata={"samples": 100},
        made_from={},
    )

    # Try to create second atom with identical content
    atom2 = DatasetAtom.create(
        user="alice",
        artifact_path=sample_artifact_2,
        metadata={"samples": 100},
        made_from={},
    )

    # Should return same atom
    assert atom1.id == atom2.id
    assert atom1.content_hash == atom2.content_hash

    # Verify only one copy stored
    data_path = get_atom_data_path(atom1.id)
    assert data_path.exists()


def test_dataset_atom_different_metadata_no_deduplication(
    sample_artifact: Path, sample_artifact_2: Path
) -> None:
    """Test that different metadata prevents deduplication."""
    # Create first atom
    atom1 = DatasetAtom.create(
        user="alice",
        artifact_path=sample_artifact,
        metadata={"samples": 100},
        made_from={},
    )

    # Create second atom with different metadata
    atom2 = DatasetAtom.create(
        user="alice",
        artifact_path=sample_artifact_2,
        metadata={"samples": 200},
        made_from={},
    )

    # Should be different atoms
    assert atom1.id != atom2.id
    assert atom1.content_hash != atom2.content_hash


def test_model_atom_deduplication(sample_artifact: Path, sample_artifact_2: Path) -> None:
    """Test that identical models are deduplicated."""
    # Create first atom
    atom1 = ModelAtom.create(
        user="bob",
        artifact_path=sample_artifact,
        metadata={"model_id": "ft-123"},
        made_from={"dataset": "dataset-xyz"},
    )

    # Try to create second atom with identical content
    atom2 = ModelAtom.create(
        user="bob",
        artifact_path=sample_artifact_2,
        metadata={"model_id": "ft-123"},
        made_from={"dataset": "dataset-xyz"},
    )

    # Should return same atom
    assert atom1.id == atom2.id
    assert atom1.content_hash == atom2.content_hash


def test_eval_atom_deduplication(sample_artifact: Path, sample_artifact_2: Path) -> None:
    """Test that identical eval results are deduplicated."""
    # Add results.json to artifacts
    results = {"score": 0.95, "samples": 100}
    (sample_artifact / "results.json").write_text(json.dumps(results))
    (sample_artifact_2 / "results.json").write_text(json.dumps(results))

    # Create first atom
    atom1 = EvalAtom.create(
        user="charlie",
        artifact_path=sample_artifact,
        metadata={"score": 0.95},
        made_from={"model": "model-abc", "dataset": "dataset-xyz"},
    )

    # Try to create second atom with identical content
    atom2 = EvalAtom.create(
        user="charlie",
        artifact_path=sample_artifact_2,
        metadata={"score": 0.95},
        made_from={"model": "model-abc", "dataset": "dataset-xyz"},
    )

    # Should return same atom
    assert atom1.id == atom2.id
    assert atom1.content_hash == atom2.content_hash


def test_hash_index_lookup(sample_artifact: Path) -> None:
    """Test hash index lookup mechanism."""
    # Create an atom
    atom = DatasetAtom.create(
        user="dave",
        artifact_path=sample_artifact,
        metadata={"test": True},
        made_from={},
    )

    # Lookup by hash
    found_atom_id = find_atom_by_hash(atom.content_hash)
    assert found_atom_id == atom.id


def test_atom_to_dict_includes_content_hash(sample_artifact: Path) -> None:
    """Test that to_dict() includes content_hash."""
    atom = DatasetAtom.create(
        user="eve",
        artifact_path=sample_artifact,
        metadata={},
        made_from={},
    )

    atom_dict = atom.to_dict()
    assert "content_hash" in atom_dict
    assert atom_dict["content_hash"] == atom.content_hash


def test_atom_load_includes_content_hash(sample_artifact: Path) -> None:
    """Test that loaded atoms include content_hash."""
    # Create and save atom
    atom = DatasetAtom.create(
        user="frank",
        artifact_path=sample_artifact,
        metadata={},
        made_from={},
    )

    # Load atom
    loaded = Atom.load(atom.id)
    assert hasattr(loaded, "content_hash")
    assert loaded.content_hash == atom.content_hash


@pytest.mark.asyncio
async def test_async_deduplication(sample_artifact: Path, sample_artifact_2: Path) -> None:
    """Test that async create also deduplicates."""
    # Create first atom
    atom1 = await DatasetAtom.acreate(
        user="grace",
        artifact_path=sample_artifact,
        metadata={"samples": 50},
        made_from={},
    )

    # Try to create second atom with identical content
    atom2 = await DatasetAtom.acreate(
        user="grace",
        artifact_path=sample_artifact_2,
        metadata={"samples": 50},
        made_from={},
    )

    # Should return same atom
    assert atom1.id == atom2.id
    assert atom1.content_hash == atom2.content_hash
