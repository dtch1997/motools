"""Tests for atom base classes."""

from datetime import datetime

import pytest

from motools.atom import Atom, DatasetAtom, EvalAtom, ModelAtom, create_temp_workspace


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


# =====================================================================
# Async tests
# =====================================================================


@pytest.mark.asyncio
async def test_async_atom_create():
    """Test creating an atom asynchronously."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.txt").write_text("test content")

        atom = await Atom.acreate(
            atom_type="test",
            user="alice",
            artifact_path=temp_dir,
            metadata={"test": True},
        )

        assert atom.id.startswith("test-alice-")
        assert atom.type == "test"
        assert isinstance(atom.created_at, datetime)
        assert atom.metadata == {"test": True}


@pytest.mark.asyncio
async def test_async_atom_load():
    """Test loading an atom asynchronously."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.txt").write_text("test content")

        # Create atom synchronously first
        original = Atom.create(
            atom_type="test",
            user="bob",
            artifact_path=temp_dir,
            metadata={"async": True},
        )

        # Load asynchronously
        loaded = await Atom.aload(original.id)

        assert loaded.id == original.id
        assert loaded.type == original.type
        assert loaded.metadata == original.metadata


@pytest.mark.asyncio
async def test_async_dataset_atom_create():
    """Test creating a DatasetAtom asynchronously."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "train.jsonl").write_text('{"text": "hello"}\n')
        (temp_dir / "test.jsonl").write_text('{"text": "world"}\n')

        atom = await DatasetAtom.acreate(
            user="charlie",
            artifact_path=temp_dir,
            metadata={"samples": 2, "format": "jsonl"},
        )

        assert atom.id.startswith("dataset-charlie-")
        assert atom.type == "dataset"
        assert atom.metadata == {"samples": 2, "format": "jsonl"}


@pytest.mark.asyncio
async def test_async_model_atom_create():
    """Test creating a ModelAtom asynchronously."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "model_id.txt").write_text("ft:gpt-4o-mini:test")

        atom = await ModelAtom.acreate(
            user="dave",
            artifact_path=temp_dir,
            metadata={"model_id": "ft:gpt-4o-mini:test"},
        )

        assert atom.id.startswith("model-dave-")
        assert atom.type == "model"
        assert atom.get_model_id() == "ft:gpt-4o-mini:test"


@pytest.mark.asyncio
async def test_async_eval_atom_create():
    """Test creating an EvalAtom asynchronously."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "results.json").write_text('{"score": 0.95}')

        atom = await EvalAtom.acreate(
            user="eve",
            artifact_path=temp_dir,
            metadata={"score": 0.95},
        )

        assert atom.id.startswith("eval-eve-")
        assert atom.type == "eval"
        assert atom.metadata["score"] == 0.95


@pytest.mark.asyncio
async def test_async_compatibility():
    """Test that async and sync APIs produce compatible results."""
    with create_temp_workspace() as temp_dir1, create_temp_workspace() as temp_dir2:
        # Create identical content
        (temp_dir1 / "data.jsonl").write_text('{"text": "test"}\n')
        (temp_dir2 / "data.jsonl").write_text('{"text": "test"}\n')

        # Create one sync, one async
        sync_atom = DatasetAtom.create(
            user="frank",
            artifact_path=temp_dir1,
            metadata={"sync": True},
        )

        async_atom = await DatasetAtom.acreate(
            user="frank",
            artifact_path=temp_dir2,
            metadata={"async": True},
        )

        # Both should have similar structure
        assert sync_atom.type == async_atom.type
        assert sync_atom.id.startswith("dataset-frank-")
        assert async_atom.id.startswith("dataset-frank-")

        # Load sync atom asynchronously and vice versa
        loaded_async = await DatasetAtom.aload(sync_atom.id)
        loaded_sync = DatasetAtom.load(async_atom.id)

        assert loaded_async.id == sync_atom.id
        assert loaded_sync.id == async_atom.id


@pytest.mark.asyncio
async def test_async_atom_load_nonexistent():
    """Test async loading a non-existent atom raises error."""
    with pytest.raises(FileNotFoundError):
        await Atom.aload("test-nobody-nonexistent")
