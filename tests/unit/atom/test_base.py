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
        (temp_dir / "train.jsonl").write_text('{"text": "async test"}\n')
        (temp_dir / "test.jsonl").write_text('{"text": "async data"}\n')

        atom = await DatasetAtom.acreate(
            user="charlie",
            artifact_path=temp_dir,
            metadata={"samples": 2, "format": "jsonl", "async": True},
        )

        assert atom.id.startswith("dataset-charlie-")
        assert atom.type == "dataset"
        assert atom.metadata["samples"] == 2
        assert atom.metadata["format"] == "jsonl"
        assert atom.metadata["async"] is True


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


# =====================================================================
# Registry tests
# =====================================================================


def test_atom_registry_auto_registers_subclasses():
    """Test that atom subclasses are automatically registered."""
    # All known atom types should be in the registry
    assert "dataset" in Atom._registry
    assert "model" in Atom._registry
    assert "eval" in Atom._registry
    assert "task" in Atom._registry
    assert "training_job" in Atom._registry

    # Registry should map to correct classes
    assert Atom._registry["dataset"] == DatasetAtom
    assert Atom._registry["model"] == ModelAtom
    assert Atom._registry["eval"] == EvalAtom


def test_atom_load_uses_registry():
    """Test that load() uses registry to return correct subclass."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        # Create a DatasetAtom
        original = DatasetAtom.create(
            user="registry-test",
            artifact_path=temp_dir,
            metadata={"test": "registry"},
        )

        # Load using base Atom.load() - should return DatasetAtom
        loaded = Atom.load(original.id)

        assert isinstance(loaded, DatasetAtom)
        assert loaded.id == original.id
        assert loaded.type == "dataset"


@pytest.mark.asyncio
async def test_atom_aload_uses_registry():
    """Test that aload() uses registry to return correct subclass."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "model_id.txt").write_text("ft:test-model")

        # Create a ModelAtom
        original = await ModelAtom.acreate(
            user="async-registry-test",
            artifact_path=temp_dir,
            metadata={"model_id": "ft:test-model"},
        )

        # Load using base Atom.aload() - should return ModelAtom
        loaded = await Atom.aload(original.id)

        assert isinstance(loaded, ModelAtom)
        assert loaded.id == original.id
        assert loaded.type == "model"


def test_atom_load_multiple_types_via_registry():
    """Test loading different atom types all return correct subclass."""
    with create_temp_workspace() as dataset_dir, create_temp_workspace() as model_dir:
        # Create different atom types
        (dataset_dir / "data.jsonl").write_text('{"text": "test"}\n')
        (model_dir / "model_id.txt").write_text("ft:model-123")

        dataset_atom = DatasetAtom.create(
            user="multi-test",
            artifact_path=dataset_dir,
        )

        model_atom = ModelAtom.create(
            user="multi-test",
            artifact_path=model_dir,
            metadata={"model_id": "ft:model-123"},
        )

        # Load both using base Atom.load()
        loaded_dataset = Atom.load(dataset_atom.id)
        loaded_model = Atom.load(model_atom.id)

        # Each should be correct subclass
        assert isinstance(loaded_dataset, DatasetAtom)
        assert isinstance(loaded_model, ModelAtom)
        assert loaded_dataset.type == "dataset"
        assert loaded_model.type == "model"


def test_custom_atom_type_registration():
    """Test that custom atom types can be registered dynamically."""
    from dataclasses import dataclass, field
    from typing import Literal

    # Define a custom atom type
    @dataclass
    class CustomAtom(Atom):
        type: Literal["custom"] = field(default="custom", init=False)

    # Should be auto-registered
    assert "custom" in Atom._registry
    assert Atom._registry["custom"] == CustomAtom


def test_atom_load_invalid_type_warns():
    """Test that loading an atom with invalid type warns and falls back to base Atom."""
    from datetime import UTC, datetime

    from motools.atom.storage import save_atom_metadata

    with create_temp_workspace() as temp_dir:
        # Create atom with invalid type manually
        (temp_dir / "data.txt").write_text("test")

        # Create a fake atom with invalid type
        fake_atom = Atom(
            id="invalid-test-12345678",
            type="not_a_real_type",
            created_at=datetime.now(UTC),
            made_from={},
            metadata={},
            content_hash="fake",
        )

        # Manually save it to storage
        save_atom_metadata(fake_atom)

        # Loading should warn and return base Atom for unrecognized type
        with pytest.warns(match="Unrecognized atom_type 'not_a_real_type'"):
            loaded = Atom.load("invalid-test-12345678")
            assert isinstance(loaded, Atom)
            assert loaded.type == "not_a_real_type"


@pytest.mark.asyncio
async def test_atom_aload_invalid_type_warns():
    """Test that async loading an atom with invalid type warns and falls back to base Atom."""
    from datetime import UTC, datetime

    from motools.atom.storage import asave_atom_metadata

    with create_temp_workspace() as temp_dir:
        # Create atom with invalid type manually
        (temp_dir / "data.txt").write_text("test")

        # Create a fake atom with invalid type
        fake_atom = Atom(
            id="invalid-async-12345678",
            type="invalid_async_type",
            created_at=datetime.now(UTC),
            made_from={},
            metadata={},
            content_hash="fake",
        )

        # Manually save it to storage
        await asave_atom_metadata(fake_atom)

        # Loading should warn and return base Atom for unrecognized type
        with pytest.warns(match="Unrecognized atom_type 'invalid_async_type'"):
            loaded = await Atom.aload("invalid-async-12345678")
            assert isinstance(loaded, Atom)
            assert loaded.type == "invalid_async_type"
