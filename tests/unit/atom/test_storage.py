"""Tests for atom storage backend."""

import pytest

from motools.atom import DatasetAtom, create_temp_workspace
from motools.atom.storage import (
    alist_atoms,
    aload_atom_metadata,
    amove_artifact_to_storage,
    asave_atom_metadata,
    list_atoms,
)


def test_list_atoms_by_type():
    """Test listing atoms filtered by type."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(user="alice", artifact_path=temp_dir)

        # List all atoms
        all_atoms = list_atoms()
        assert atom.id in all_atoms

        # List only dataset atoms
        dataset_atoms = list_atoms(atom_type="dataset")
        assert atom.id in dataset_atoms

        # List non-existent type
        other_atoms = list_atoms(atom_type="nonexistent")
        assert atom.id not in other_atoms


@pytest.mark.asyncio
async def test_async_list_atoms_by_type():
    """Test async listing atoms filtered by type."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(user="alice", artifact_path=temp_dir)

        # List all atoms
        all_atoms = await alist_atoms()
        assert atom.id in all_atoms

        # List only dataset atoms
        dataset_atoms = await alist_atoms(atom_type="dataset")
        assert atom.id in dataset_atoms

        # List non-existent type
        other_atoms = await alist_atoms(atom_type="nonexistent")
        assert atom.id not in other_atoms


@pytest.mark.asyncio
async def test_async_save_and_load_metadata():
    """Test async saving and loading atom metadata."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        atom = DatasetAtom.create(user="alice", artifact_path=temp_dir)

        # Save metadata asynchronously
        await asave_atom_metadata(atom)

        # Load metadata asynchronously
        metadata = await aload_atom_metadata(atom.id)

        assert metadata["id"] == atom.id
        assert metadata["type"] == atom.type
        assert metadata["made_from"] == atom.made_from


@pytest.mark.asyncio
async def test_async_move_artifact():
    """Test async moving artifact to storage."""
    import shutil

    from motools.atom.storage import get_atom_cache_path, get_atom_data_path

    with create_temp_workspace() as temp_dir:
        # Create a test artifact directory
        artifact_dir = temp_dir / "artifact"
        artifact_dir.mkdir()
        (artifact_dir / "test.txt").write_text("test content")

        # Use a unique test ID
        import uuid

        test_atom_id = f"test-atom-{uuid.uuid4().hex[:8]}"

        # Clean up any existing test data
        cache_path = get_atom_cache_path(test_atom_id)
        if cache_path.exists():
            shutil.rmtree(cache_path)

        # Move it asynchronously
        await amove_artifact_to_storage(test_atom_id, artifact_dir)

        # Verify the artifact was moved
        data_path = get_atom_data_path(test_atom_id)
        assert data_path.exists()
        assert (data_path / "test.txt").exists()
        assert (data_path / "test.txt").read_text() == "test content"

        # Clean up after test
        if cache_path.exists():
            shutil.rmtree(cache_path)


@pytest.mark.asyncio
async def test_async_compatibility_with_sync():
    """Test that async and sync APIs produce same results."""
    with create_temp_workspace() as temp_dir:
        (temp_dir / "data.jsonl").write_text('{"text": "test"}\n')

        DatasetAtom.create(user="alice", artifact_path=temp_dir)

        # Compare sync and async list_atoms
        sync_atoms = list_atoms()
        async_atoms = await alist_atoms()
        assert sync_atoms == async_atoms

        sync_dataset_atoms = list_atoms(atom_type="dataset")
        async_dataset_atoms = await alist_atoms(atom_type="dataset")
        assert sync_dataset_atoms == async_dataset_atoms
