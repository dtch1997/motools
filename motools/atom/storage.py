"""Storage backend for atoms - local filesystem only."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import yaml
from filelock import FileLock

from motools.protocols import AtomProtocol

# Storage directories
ATOMS_BASE_DIR = Path(".motools/atoms")
ATOMS_INDEX_DIR = ATOMS_BASE_DIR / "index"
ATOMS_DATA_DIR = ATOMS_BASE_DIR / "data"
ATOMS_HASH_INDEX = ATOMS_BASE_DIR / "hash_index.yaml"
ATOMS_HASH_INDEX_LOCK = ATOMS_BASE_DIR / "hash_index.lock"
ATOM_METADATA_FILENAME = "_atom.yaml"
ATOM_DATA_SUBDIR = "data"


def get_atom_index_path(atom_id: str) -> Path:
    """Get path to atom's index file.

    Args:
        atom_id: Atom identifier

    Returns:
        Path to index YAML file
    """
    return ATOMS_INDEX_DIR / f"{atom_id}.yaml"


def get_atom_cache_path(atom_id: str) -> Path:
    """Get path to atom's data cache directory.

    Args:
        atom_id: Atom identifier

    Returns:
        Path to cache directory (contains _atom.yaml + data/)
    """
    return ATOMS_DATA_DIR / atom_id


def get_atom_data_path(atom_id: str) -> Path:
    """Get path to atom's actual data directory.

    Args:
        atom_id: Atom identifier

    Returns:
        Path to data subdirectory
    """
    return get_atom_cache_path(atom_id) / ATOM_DATA_SUBDIR


def save_atom_metadata(atom: AtomProtocol) -> None:
    """Save atom metadata to index and cache.

    Args:
        atom: Atom instance to save
    """
    # Ensure directories exist
    ATOMS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_atom_cache_path(atom.id)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Serialize atom
    atom_dict = atom.to_dict()

    # Save to index
    index_path = get_atom_index_path(atom.id)
    with open(index_path, "w") as f:
        yaml.dump(atom_dict, f, sort_keys=False)

    # Save to cache
    cache_metadata_path = cache_path / ATOM_METADATA_FILENAME
    with open(cache_metadata_path, "w") as f:
        yaml.dump(atom_dict, f, sort_keys=False)


def move_artifact_to_storage(atom_id: str, artifact_path: Path) -> None:
    """Move artifact data to atom storage.

    Args:
        atom_id: Atom identifier
        artifact_path: Path to artifact (file or directory)

    Raises:
        ValueError: If destination already exists
    """
    data_path = get_atom_data_path(atom_id)

    # Safety check
    if data_path.exists():
        raise ValueError(f"Cannot move artifact - destination already exists: {data_path}")

    # Create parent directory
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if artifact_path.is_dir():
        # Move directory contents into data/
        data_path.mkdir(exist_ok=True)
        for item in artifact_path.iterdir():
            target = data_path / item.name
            if target.exists():
                raise ValueError(f"Cannot move {item} - target exists: {target}")
            shutil.move(str(item), str(target))
    else:
        # Move single file into data/
        data_path.mkdir(exist_ok=True)
        target_file = data_path / artifact_path.name
        if target_file.exists():
            raise ValueError(f"Cannot move file - target exists: {target_file}")
        shutil.move(str(artifact_path), str(target_file))


def list_atoms(atom_type: str | None = None) -> list[str]:
    """List all atom IDs, optionally filtered by type.

    Args:
        atom_type: Optional type filter (e.g., "dataset")

    Returns:
        List of atom IDs
    """
    if not ATOMS_INDEX_DIR.exists():
        return []

    atom_ids = []
    for yaml_file in ATOMS_INDEX_DIR.glob("*.yaml"):
        atom_id = yaml_file.stem
        if atom_type is None or atom_id.startswith(f"{atom_type}-"):
            atom_ids.append(atom_id)

    return sorted(atom_ids)


def load_hash_index() -> dict[str, str]:
    """Load the hash index (maps content_hash -> atom_id).

    Returns:
        Dictionary mapping content hashes to atom IDs
    """
    if not ATOMS_HASH_INDEX.exists():
        return {}

    with open(ATOMS_HASH_INDEX) as f:
        result = yaml.safe_load(f)
        return result or {}  # type: ignore[return-value]


def save_hash_index(hash_index: dict[str, str]) -> None:
    """Save the hash index to disk atomically.

    Uses atomic write (temp file + rename) to prevent corruption if interrupted.

    Args:
        hash_index: Dictionary mapping content hashes to atom IDs
    """
    import os
    import tempfile

    ATOMS_BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then atomically rename
    fd, temp_path = tempfile.mkstemp(dir=ATOMS_BASE_DIR, prefix=".hash_index_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(hash_index, f, sort_keys=False)
        # Atomic rename
        os.replace(temp_path, ATOMS_HASH_INDEX)
    except Exception:
        # Clean up temp file on error
        if Path(temp_path).exists():
            os.unlink(temp_path)
        raise


def find_atom_by_hash(content_hash: str) -> str | None:
    """Look up an atom ID by content hash.

    Args:
        content_hash: Content hash to look up

    Returns:
        Atom ID if found, None otherwise
    """
    hash_index = load_hash_index()
    return hash_index.get(content_hash)


def register_atom_hash(content_hash: str, atom_id: str) -> None:
    """Register a content hash -> atom ID mapping.

    This operation is thread-safe and process-safe using file locking to prevent
    race conditions during concurrent hash registrations.

    Args:
        content_hash: Content hash
        atom_id: Atom ID
    """
    with FileLock(ATOMS_HASH_INDEX_LOCK, timeout=10):
        hash_index = load_hash_index()
        hash_index[content_hash] = atom_id
        save_hash_index(hash_index)


# =====================================================================
# Async versions of storage functions (primary API)
# =====================================================================


async def asave_atom_metadata(atom: AtomProtocol) -> None:
    """Save atom metadata to index and cache asynchronously.

    Args:
        atom: Atom instance to save
    """
    # Ensure directories exist
    await asyncio.to_thread(ATOMS_INDEX_DIR.mkdir, parents=True, exist_ok=True)
    cache_path = get_atom_cache_path(atom.id)
    await asyncio.to_thread(cache_path.mkdir, parents=True, exist_ok=True)

    # Serialize atom
    atom_dict = atom.to_dict()
    yaml_content = yaml.dump(atom_dict, sort_keys=False)

    # Save to index
    index_path = get_atom_index_path(atom.id)
    async with aiofiles.open(index_path, "w") as f:
        await f.write(yaml_content)

    # Save to cache
    cache_metadata_path = cache_path / ATOM_METADATA_FILENAME
    async with aiofiles.open(cache_metadata_path, "w") as f:
        await f.write(yaml_content)


async def amove_artifact_to_storage(atom_id: str, artifact_path: Path) -> None:
    """Move artifact data to atom storage asynchronously.

    Args:
        atom_id: Atom identifier
        artifact_path: Path to artifact (file or directory)

    Raises:
        ValueError: If destination already exists
    """
    data_path = get_atom_data_path(atom_id)

    # Safety check
    if await asyncio.to_thread(data_path.exists):
        raise ValueError(f"Cannot move artifact - destination already exists: {data_path}")

    # Create parent directory
    await asyncio.to_thread(data_path.parent.mkdir, parents=True, exist_ok=True)

    # Use asyncio.to_thread for shutil operations
    if await asyncio.to_thread(artifact_path.is_dir):
        # Move directory contents into data/
        await asyncio.to_thread(data_path.mkdir, exist_ok=True)
        items = await asyncio.to_thread(list, artifact_path.iterdir())
        for item in items:
            target = data_path / item.name
            if await asyncio.to_thread(target.exists):
                raise ValueError(f"Cannot move {item} - target exists: {target}")
            await asyncio.to_thread(shutil.move, str(item), str(target))
    else:
        # Move single file into data/
        await asyncio.to_thread(data_path.mkdir, exist_ok=True)
        target_file = data_path / artifact_path.name
        if await asyncio.to_thread(target_file.exists):
            raise ValueError(f"Cannot move file - target exists: {target_file}")
        await asyncio.to_thread(shutil.move, str(artifact_path), str(target_file))


async def aload_atom_metadata(atom_id: str) -> dict[str, Any]:
    """Load atom metadata asynchronously.

    Args:
        atom_id: Atom identifier

    Returns:
        Atom metadata dictionary

    Raises:
        FileNotFoundError: If atom not found
    """
    index_path = get_atom_index_path(atom_id)

    if not await asyncio.to_thread(index_path.exists):
        raise FileNotFoundError(f"Atom not found: {atom_id}")

    async with aiofiles.open(index_path) as f:
        content = await f.read()
        result = yaml.safe_load(content)
        return result  # type: ignore[no-any-return]


async def alist_atoms(atom_type: str | None = None) -> list[str]:
    """List all atom IDs asynchronously, optionally filtered by type.

    Args:
        atom_type: Optional type filter (e.g., "dataset")

    Returns:
        List of atom IDs
    """
    if not await asyncio.to_thread(ATOMS_INDEX_DIR.exists):
        return []

    atom_ids = []
    yaml_files = await asyncio.to_thread(list, ATOMS_INDEX_DIR.glob("*.yaml"))
    for yaml_file in yaml_files:
        atom_id = yaml_file.stem
        if atom_type is None or atom_id.startswith(f"{atom_type}-"):
            atom_ids.append(atom_id)

    return sorted(atom_ids)


async def aload_hash_index() -> dict[str, str]:
    """Load the hash index asynchronously.

    Returns:
        Dictionary mapping content hashes to atom IDs
    """
    if not await asyncio.to_thread(ATOMS_HASH_INDEX.exists):
        return {}

    async with aiofiles.open(ATOMS_HASH_INDEX) as f:
        content = await f.read()
        result = yaml.safe_load(content)
        return result or {}  # type: ignore[return-value]


async def asave_hash_index(hash_index: dict[str, str]) -> None:
    """Save the hash index to disk asynchronously.

    Args:
        hash_index: Dictionary mapping content hashes to atom IDs
    """
    await asyncio.to_thread(ATOMS_BASE_DIR.mkdir, parents=True, exist_ok=True)
    yaml_content = yaml.dump(hash_index, sort_keys=False)
    async with aiofiles.open(ATOMS_HASH_INDEX, "w") as f:
        await f.write(yaml_content)


async def afind_atom_by_hash(content_hash: str) -> str | None:
    """Look up an atom ID by content hash asynchronously.

    Args:
        content_hash: Content hash to look up

    Returns:
        Atom ID if found, None otherwise
    """
    hash_index = await aload_hash_index()
    return hash_index.get(content_hash)


async def aregister_atom_hash(content_hash: str, atom_id: str) -> None:
    """Register a content hash -> atom ID mapping asynchronously.

    This operation is thread-safe and process-safe using file locking to prevent
    race conditions during concurrent hash registrations.

    Args:
        content_hash: Content hash
        atom_id: Atom ID
    """
    # Use synchronous locking to ensure atomicity across processes
    await asyncio.to_thread(register_atom_hash, content_hash, atom_id)
