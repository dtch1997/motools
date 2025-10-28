"""Tests for atomic behavior in atom storage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from motools.atom.storage import (
    ATOM_METADATA_FILENAME,
    asave_atom_metadata,
    get_atom_cache_path,
    get_atom_index_path,
    save_atom_metadata,
)


class MockAtom:
    """Mock atom for testing."""

    def __init__(self, atom_id: str, metadata: dict) -> None:
        self.id = atom_id
        self._metadata = metadata

    def to_dict(self) -> dict:
        return {"id": self.id, **self._metadata}


def test_save_atom_metadata_atomic_on_error(tmp_path: Path, monkeypatch) -> None:
    """Test that save_atom_metadata doesn't leave partial state on error."""
    # Change to test directory to use relative paths
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("test-atom", {"type": "test", "data": "value"})

    # Mock atomic_write_yaml to fail on second call (cache write)
    call_count = 0
    original_atomic_write_yaml = None

    def failing_atomic_write_yaml(*args, **kwargs):
        nonlocal call_count, original_atomic_write_yaml
        call_count += 1
        if call_count == 1:
            # First call (index) should succeed
            original_atomic_write_yaml(*args, **kwargs)
        else:
            # Second call (cache) should fail
            raise OSError("Simulated write failure")

    with patch("motools.atom.storage.atomic_write_yaml") as mock_write:
        # Import the real function to use for first call
        from motools.fs_utils import atomic_write_yaml as real_atomic_write_yaml

        original_atomic_write_yaml = real_atomic_write_yaml

        mock_write.side_effect = failing_atomic_write_yaml

        with pytest.raises(OSError, match="Simulated write failure"):
            save_atom_metadata(atom)

    # Check state after failed save
    index_path = get_atom_index_path(atom.id)
    cache_path = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

    # Index should exist (first write succeeded)
    assert index_path.exists()
    index_data = yaml.safe_load(index_path.read_text())
    assert index_data["id"] == "test-atom"

    # Cache should not exist (second write failed and atomic write cleaned up)
    assert not cache_path.exists()


@pytest.mark.asyncio
async def test_asave_atom_metadata_atomic_on_error(tmp_path: Path, monkeypatch) -> None:
    """Test that async save_atom_metadata doesn't leave partial state on error."""
    # Change to test directory
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("async-test-atom", {"type": "test", "async": True})

    # Mock atomic_write_yaml_async to fail on second call
    call_count = 0
    original_atomic_write_yaml_async = None

    async def failing_atomic_write_yaml_async(*args, **kwargs):
        nonlocal call_count, original_atomic_write_yaml_async
        call_count += 1
        if call_count == 1:
            # First call should succeed
            await original_atomic_write_yaml_async(*args, **kwargs)
        else:
            # Second call should fail
            raise OSError("Async simulated write failure")

    with patch("motools.atom.storage.atomic_write_yaml_async") as mock_write:
        # Import the real function
        from motools.fs_utils import atomic_write_yaml_async as real_function

        original_atomic_write_yaml_async = real_function

        mock_write.side_effect = failing_atomic_write_yaml_async

        with pytest.raises(OSError, match="Async simulated write failure"):
            await asave_atom_metadata(atom)

    # Check state after failed save
    index_path = get_atom_index_path(atom.id)
    cache_path = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

    # Index should exist (first write succeeded)
    assert index_path.exists()
    index_data = yaml.safe_load(index_path.read_text())
    assert index_data["id"] == "async-test-atom"

    # Cache should not exist (second write failed)
    assert not cache_path.exists()


def test_save_atom_metadata_both_succeed(tmp_path: Path, monkeypatch) -> None:
    """Test that both index and cache are written when save succeeds."""
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("success-atom", {"type": "test", "status": "success"})

    save_atom_metadata(atom)

    # Both files should exist and have same content
    index_path = get_atom_index_path(atom.id)
    cache_path = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

    assert index_path.exists()
    assert cache_path.exists()

    index_data = yaml.safe_load(index_path.read_text())
    cache_data = yaml.safe_load(cache_path.read_text())

    expected_data = {"id": "success-atom", "type": "test", "status": "success"}
    assert index_data == expected_data
    assert cache_data == expected_data


@pytest.mark.asyncio
async def test_asave_atom_metadata_both_succeed(tmp_path: Path, monkeypatch) -> None:
    """Test that both index and cache are written when async save succeeds."""
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("async-success-atom", {"type": "test", "async": True, "status": "success"})

    await asave_atom_metadata(atom)

    # Both files should exist and have same content
    index_path = get_atom_index_path(atom.id)
    cache_path = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

    assert index_path.exists()
    assert cache_path.exists()

    index_data = yaml.safe_load(index_path.read_text())
    cache_data = yaml.safe_load(cache_path.read_text())

    expected_data = {"id": "async-success-atom", "type": "test", "async": True, "status": "success"}
    assert index_data == expected_data
    assert cache_data == expected_data


def test_save_atom_metadata_uses_atomic_writes(tmp_path: Path, monkeypatch) -> None:
    """Test that save_atom_metadata uses atomic writes."""
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("atomic-test", {"type": "test"})

    with patch("motools.atom.storage.atomic_write_yaml") as mock_atomic_write:
        save_atom_metadata(atom)

        # Should be called twice (index + cache)
        assert mock_atomic_write.call_count == 2

        # Check the paths and data
        calls = mock_atomic_write.call_args_list
        paths = [call[0][0] for call in calls]
        data = [call[0][1] for call in calls]

        # Should write to both index and cache paths
        expected_index = get_atom_index_path(atom.id)
        expected_cache = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

        assert expected_index in paths
        assert expected_cache in paths

        # Both should have same data
        expected_data = {"id": "atomic-test", "type": "test"}
        assert all(d == expected_data for d in data)


@pytest.mark.asyncio
async def test_asave_atom_metadata_uses_atomic_writes(tmp_path: Path, monkeypatch) -> None:
    """Test that async save_atom_metadata uses atomic writes."""
    monkeypatch.chdir(tmp_path)

    atom = MockAtom("async-atomic-test", {"type": "test", "async": True})

    with patch("motools.atom.storage.atomic_write_yaml_async") as mock_atomic_write:
        await asave_atom_metadata(atom)

        # Should be called twice (index + cache)
        assert mock_atomic_write.call_count == 2

        # Check the paths and data
        calls = mock_atomic_write.call_args_list
        paths = [call[0][0] for call in calls]
        data = [call[0][1] for call in calls]

        # Should write to both index and cache paths
        expected_index = get_atom_index_path(atom.id)
        expected_cache = get_atom_cache_path(atom.id) / ATOM_METADATA_FILENAME

        assert expected_index in paths
        assert expected_cache in paths

        # Both should have same data
        expected_data = {"id": "async-atomic-test", "type": "test", "async": True}
        assert all(d == expected_data for d in data)
