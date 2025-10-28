"""Filesystem utilities for safe file operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import aiofiles
import yaml


def atomic_write(path: Path | str, content: str | bytes, mode: str = "w") -> None:
    """Write content to a file atomically using temp file + rename.

    This ensures that the file is either completely written or not written at all,
    preventing partial writes or corruption on crashes/interruptions.

    Args:
        path: Path to the target file
        content: Content to write (string for text mode, bytes for binary mode)
        mode: Write mode ('w' for text, 'wb' for binary)

    Raises:
        TypeError: If content type doesn't match mode
        OSError: If write or rename fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate content type matches mode
    is_binary = "b" in mode
    if is_binary and isinstance(content, str):
        raise TypeError("Binary mode requires bytes content")
    if not is_binary and isinstance(content, bytes):
        raise TypeError("Text mode requires string content")

    # Write to temp file in same directory (ensures same filesystem)
    with tempfile.NamedTemporaryFile(
        mode=mode, delete=False, dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Atomic rename (POSIX guarantees atomicity on same filesystem)
    try:
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise


def atomic_write_yaml(path: Path | str, data: Any, **yaml_kwargs: Any) -> None:
    """Write YAML data to a file atomically.

    Args:
        path: Path to the target YAML file
        data: Data to serialize to YAML
        **yaml_kwargs: Additional arguments for yaml.dump()
    """
    yaml_kwargs.setdefault("sort_keys", False)
    content = yaml.dump(data, **yaml_kwargs)
    atomic_write(path, content)


async def atomic_write_async(
    path: Path | str, content: str | bytes, mode: str = "w"
) -> None:
    """Write content to a file atomically (async version).

    This ensures that the file is either completely written or not written at all,
    preventing partial writes or corruption on crashes/interruptions.

    Args:
        path: Path to the target file
        content: Content to write (string for text mode, bytes for binary mode)
        mode: Write mode ('w' for text, 'wb' for binary)

    Raises:
        TypeError: If content type doesn't match mode
        OSError: If write or rename fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate content type matches mode
    is_binary = "b" in mode
    if is_binary and isinstance(content, str):
        raise TypeError("Binary mode requires bytes content")
    if not is_binary and isinstance(content, bytes):
        raise TypeError("Text mode requires string content")

    # Generate temp file path
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(tmp_fd)  # Close the file descriptor, we'll use aiofiles

    try:
        # Write content asynchronously
        async with aiofiles.open(tmp_path, mode=mode) as tmp:
            await tmp.write(content)

        # Atomic rename (still synchronous as os.replace is fast)
        os.replace(tmp_path, path)
    except Exception:
        # Clean up temp file on error
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise


async def atomic_write_yaml_async(path: Path | str, data: Any, **yaml_kwargs: Any) -> None:
    """Write YAML data to a file atomically (async version).

    Args:
        path: Path to the target YAML file
        data: Data to serialize to YAML
        **yaml_kwargs: Additional arguments for yaml.dump()
    """
    yaml_kwargs.setdefault("sort_keys", False)
    content = yaml.dump(data, **yaml_kwargs)
    await atomic_write_async(path, content)