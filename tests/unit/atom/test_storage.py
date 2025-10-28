"""Tests for atom storage backend."""

import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import pytest

from motools.atom import DatasetAtom, create_temp_workspace
from motools.atom.storage import (
    alist_atoms,
    aload_atom_metadata,
    amove_artifact_to_storage,
    aregister_atom_hash,
    asave_atom_metadata,
    list_atoms,
    load_hash_index,
    register_atom_hash,
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


def test_concurrent_hash_registration_threads():
    """Test that concurrent hash registrations don't corrupt data (thread-level)."""
    with create_temp_workspace():
        # Register multiple hashes concurrently using threads
        num_threads = 10
        hashes = [(f"hash_{i}", f"atom_{i}") for i in range(num_threads)]

        def register_hash(content_hash: str, atom_id: str) -> None:
            register_atom_hash(content_hash, atom_id)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(register_hash, content_hash, atom_id)
                for content_hash, atom_id in hashes
            ]
            for future in futures:
                future.result()

        # Verify all hashes were registered correctly
        hash_index = load_hash_index()
        for content_hash, atom_id in hashes:
            assert hash_index[content_hash] == atom_id, (
                f"Hash {content_hash} should map to {atom_id}, got {hash_index.get(content_hash)}"
            )


def _register_hash_worker(content_hash: str, atom_id: str) -> None:
    """Worker function for multiprocessing test."""
    register_atom_hash(content_hash, atom_id)


def test_concurrent_hash_registration_processes():
    """Test that concurrent hash registrations don't corrupt data (process-level)."""
    with create_temp_workspace():
        # Register multiple hashes concurrently using processes
        num_processes = 5
        hashes = [(f"hash_proc_{i}", f"atom_proc_{i}") for i in range(num_processes)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(_register_hash_worker, hashes)

        # Verify all hashes were registered correctly
        hash_index = load_hash_index()
        for content_hash, atom_id in hashes:
            assert hash_index[content_hash] == atom_id, (
                f"Hash {content_hash} should map to {atom_id}, got {hash_index.get(content_hash)}"
            )


@pytest.mark.asyncio
async def test_concurrent_async_hash_registration():
    """Test that concurrent async hash registrations don't corrupt data."""
    with create_temp_workspace():
        # Register multiple hashes concurrently using async tasks
        num_tasks = 10
        hashes = [(f"hash_async_{i}", f"atom_async_{i}") for i in range(num_tasks)]

        tasks = [aregister_atom_hash(content_hash, atom_id) for content_hash, atom_id in hashes]
        await asyncio.gather(*tasks)

        # Verify all hashes were registered correctly
        hash_index = load_hash_index()
        for content_hash, atom_id in hashes:
            assert hash_index[content_hash] == atom_id, (
                f"Hash {content_hash} should map to {atom_id}, got {hash_index.get(content_hash)}"
            )


def test_concurrent_duplicate_hash_registration():
    """Test that concurrent duplicate hash registrations are handled correctly."""
    with create_temp_workspace():
        # Multiple threads try to register the same hash
        num_threads = 5
        same_hash = "duplicate_hash_123"
        same_atom_id = "atom_xyz"

        def register_hash(content_hash: str, atom_id: str) -> None:
            register_atom_hash(content_hash, atom_id)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(register_hash, same_hash, same_atom_id) for _ in range(num_threads)
            ]
            for future in futures:
                future.result()

        # Verify the hash was registered correctly (last write wins)
        hash_index = load_hash_index()
        assert hash_index[same_hash] == same_atom_id
