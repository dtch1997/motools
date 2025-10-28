"""Tests for filesystem utilities."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from motools.fs_utils import (
    atomic_write,
    atomic_write_async,
    atomic_write_yaml,
    atomic_write_yaml_async,
)


def test_atomic_write_text(tmp_path: Path) -> None:
    """Test atomic write with text content."""
    file_path = tmp_path / "test.txt"
    content = "Hello, World!"

    atomic_write(file_path, content)

    assert file_path.exists()
    assert file_path.read_text() == content


def test_atomic_write_binary(tmp_path: Path) -> None:
    """Test atomic write with binary content."""
    file_path = tmp_path / "test.bin"
    content = b"Binary content \x00\xff"

    atomic_write(file_path, content, mode="wb")

    assert file_path.exists()
    assert file_path.read_bytes() == content


def test_atomic_write_creates_parent_dirs(tmp_path: Path) -> None:
    """Test that atomic write creates parent directories if needed."""
    file_path = tmp_path / "nested" / "deep" / "test.txt"
    content = "Nested file"

    atomic_write(file_path, content)

    assert file_path.exists()
    assert file_path.read_text() == content


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    """Test that atomic write properly overwrites existing files."""
    file_path = tmp_path / "test.txt"

    # Write initial content
    atomic_write(file_path, "Initial")
    assert file_path.read_text() == "Initial"

    # Overwrite with new content
    atomic_write(file_path, "Updated")
    assert file_path.read_text() == "Updated"


def test_atomic_write_type_validation(tmp_path: Path) -> None:
    """Test that atomic write validates content type matches mode."""
    file_path = tmp_path / "test.txt"

    # Text mode with bytes should fail
    with pytest.raises(TypeError, match="Text mode requires string content"):
        atomic_write(file_path, b"bytes", mode="w")

    # Binary mode with string should fail
    with pytest.raises(TypeError, match="Binary mode requires bytes content"):
        atomic_write(file_path, "string", mode="wb")


def test_atomic_write_cleanup_on_error(tmp_path: Path) -> None:
    """Test that temp files are cleaned up on write errors."""
    file_path = tmp_path / "test.txt"

    with patch("os.replace") as mock_replace:
        mock_replace.side_effect = OSError("Rename failed")

        with pytest.raises(OSError):
            atomic_write(file_path, "content")

        # Check no temp files left behind
        temp_files = list(tmp_path.glob(".test.txt.*.tmp"))
        assert len(temp_files) == 0


def test_atomic_write_no_partial_writes(tmp_path: Path) -> None:
    """Test that atomic write doesn't leave partial writes on interruption."""
    file_path = tmp_path / "test.txt"

    # Write initial content
    atomic_write(file_path, "Initial content")

    # Simulate write interruption
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.side_effect = KeyboardInterrupt("Simulated interruption")

        with pytest.raises(KeyboardInterrupt):
            atomic_write(file_path, "New content that should not be written")

        # Original file should be unchanged
        assert file_path.read_text() == "Initial content"


def test_atomic_write_yaml(tmp_path: Path) -> None:
    """Test atomic YAML write."""
    file_path = tmp_path / "test.yaml"
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}

    atomic_write_yaml(file_path, data)

    assert file_path.exists()
    loaded = yaml.safe_load(file_path.read_text())
    assert loaded == data


def test_atomic_write_yaml_preserves_order(tmp_path: Path) -> None:
    """Test that atomic YAML write preserves key order by default."""
    file_path = tmp_path / "test.yaml"
    data = {"zebra": 1, "apple": 2, "middle": 3}

    atomic_write_yaml(file_path, data)

    # Check that keys appear in original order (not sorted)
    content = file_path.read_text()
    zebra_pos = content.index("zebra")
    apple_pos = content.index("apple")
    middle_pos = content.index("middle")
    assert zebra_pos < apple_pos < middle_pos


@pytest.mark.asyncio
async def test_atomic_write_async_text(tmp_path: Path) -> None:
    """Test async atomic write with text content."""
    file_path = tmp_path / "test.txt"
    content = "Async Hello!"

    await atomic_write_async(file_path, content)

    assert file_path.exists()
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_atomic_write_async_binary(tmp_path: Path) -> None:
    """Test async atomic write with binary content."""
    file_path = tmp_path / "test.bin"
    content = b"Async binary \x00\xff"

    await atomic_write_async(file_path, content, mode="wb")

    assert file_path.exists()
    assert file_path.read_bytes() == content


@pytest.mark.asyncio
async def test_atomic_write_async_cleanup_on_error(tmp_path: Path) -> None:
    """Test that async atomic write cleans up temp files on error."""
    file_path = tmp_path / "test.txt"

    with patch("os.replace") as mock_replace:
        mock_replace.side_effect = OSError("Async rename failed")

        with pytest.raises(OSError):
            await atomic_write_async(file_path, "content")

        # Check no temp files left behind
        temp_files = list(tmp_path.glob(".test.txt.*.tmp"))
        assert len(temp_files) == 0


@pytest.mark.asyncio
async def test_atomic_write_yaml_async(tmp_path: Path) -> None:
    """Test async atomic YAML write."""
    file_path = tmp_path / "test.yaml"
    data = {"async": True, "value": "test", "nested": {"key": "value"}}

    await atomic_write_yaml_async(file_path, data)

    assert file_path.exists()
    loaded = yaml.safe_load(file_path.read_text())
    assert loaded == data


def test_concurrent_atomic_writes(tmp_path: Path) -> None:
    """Test that concurrent atomic writes don't corrupt files."""
    file_path = tmp_path / "concurrent.txt"

    def write_worker(worker_id: int, iterations: int) -> None:
        for i in range(iterations):
            content = f"Worker {worker_id} iteration {i}"
            atomic_write(file_path, content)
            time.sleep(0.001)  # Small delay to increase contention

    # Start multiple threads writing concurrently
    threads = []
    for worker_id in range(5):
        thread = threading.Thread(target=write_worker, args=(worker_id, 10))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # File should exist and contain valid content from one of the workers
    assert file_path.exists()
    content = file_path.read_text()
    assert content.startswith("Worker")
    assert "iteration" in content

    # Content should be complete (not partial)
    parts = content.split()
    assert len(parts) == 4  # "Worker X iteration Y"


@pytest.mark.asyncio
async def test_concurrent_atomic_writes_async(tmp_path: Path) -> None:
    """Test that concurrent async atomic writes don't corrupt files."""
    file_path = tmp_path / "concurrent_async.txt"

    async def write_worker(worker_id: int, iterations: int) -> None:
        for i in range(iterations):
            content = f"Async worker {worker_id} iteration {i}"
            await atomic_write_async(file_path, content)
            await asyncio.sleep(0.001)

    # Start multiple async tasks writing concurrently
    tasks = [write_worker(worker_id, 10) for worker_id in range(5)]
    await asyncio.gather(*tasks)

    # File should exist and contain valid content
    assert file_path.exists()
    content = file_path.read_text()
    assert content.startswith("Async worker")
    assert "iteration" in content

    # Content should be complete
    parts = content.split()
    assert len(parts) == 5  # "Async worker X iteration Y"
