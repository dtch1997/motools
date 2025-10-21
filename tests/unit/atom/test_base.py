"""Tests for atom base classes."""

from datetime import datetime

import pytest

from motools.atom import DatasetAtom, create_temp_workspace


def test_dataset_atom_create():
    """Test creating a DatasetAtom."""
    with create_temp_workspace() as temp_dir:
        # Create some test data
        (temp_dir / "train.jsonl").write_text('{"text": "hello"}\n')
        (temp_dir / "test.jsonl").write_text('{"text": "world"}\n')

        # Create atom
        atom = DatasetAtom.create(
            user="alice",
            artifact_path=temp_dir,
            metadata={"samples": 2, "format": "jsonl"},
        )

        assert atom.id.startswith("dataset-alice-")
        assert atom.type == "dataset"
        assert isinstance(atom.created_at, datetime)
        assert atom.metadata == {"samples": 2, "format": "jsonl"}
        assert atom.made_from == {}


def test_dataset_atom_create_with_provenance():
    """Test creating a DatasetAtom with provenance."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(
            user="bob",
            artifact_path=temp_dir,
            made_from={"raw_data": "dataset-alice-abc123"},
            metadata={"processed": True},
        )

        assert atom.made_from == {"raw_data": "dataset-alice-abc123"}
        assert atom.metadata["processed"] is True


def test_dataset_atom_load():
    """Test loading a DatasetAtom."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        # Create atom
        original = DatasetAtom.create(
            user="charlie",
            artifact_path=temp_dir,
            metadata={"test": True},
        )

        # Load atom
        loaded = DatasetAtom.load(original.id)

        assert loaded.id == original.id
        assert loaded.type == original.type
        assert loaded.metadata == original.metadata
        assert isinstance(loaded, DatasetAtom)


def test_dataset_atom_get_data_path():
    """Test accessing atom data."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "train.jsonl").write_text('{"text": "hello"}\n')

        atom = DatasetAtom.create(user="dave", artifact_path=temp_dir)

        # Get data path
        data_path = atom.get_data_path()
        assert data_path.exists()
        assert (data_path / "train.jsonl").exists()
        assert (data_path / "train.jsonl").read_text() == '{"text": "hello"}\n'


def test_atom_load_nonexistent():
    """Test loading a non-existent atom raises error."""
    with pytest.raises(FileNotFoundError):
        DatasetAtom.load("dataset-nobody-nonexistent")


def test_dataset_atom_single_file():
    """Test creating atom from single file."""
    with create_temp_workspace() as temp_dir:
        data_file = temp_dir / "data.jsonl"
        data_file.write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(user="eve", artifact_path=data_file)

        data_path = atom.get_data_path()
        assert (data_path / "data.jsonl").exists()
