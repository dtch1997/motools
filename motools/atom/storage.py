"""Storage backend for atoms - local filesystem only."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import yaml
from filelock import FileLock

from motools.fs_utils import atomic_write_yaml, atomic_write_yaml_async
from motools.protocols import AtomProtocol


@dataclass
class StorageConfig:
    """Configuration for atom storage paths.

    Provides configurable storage locations with sensible defaults.
    Supports environment-based configuration.
    """

    base_dir: Path = Path(".motools/atoms")

    @property
    def index_dir(self) -> Path:
        """Directory for atom index files."""
        return self.base_dir / "index"

    @property
    def data_dir(self) -> Path:
        """Directory for atom data cache."""
        return self.base_dir / "data"

    @property
    def hash_index(self) -> Path:
        """Path to hash index file."""
        return self.base_dir / "hash_index.yaml"

    @property
    def hash_index_lock(self) -> Path:
        """Path to hash index lock file."""
        return self.base_dir / "hash_index.lock"

    @classmethod
    def from_env(cls) -> StorageConfig:
        """Create StorageConfig from environment variables.

        Uses MOTOOLS_STORAGE_DIR environment variable if set.
        """
        base_dir = os.getenv("MOTOOLS_STORAGE_DIR", ".motools/atoms")
        return cls(base_dir=Path(base_dir))


class AtomStorage:
    """Storage backend for atoms with dependency injection.

    Encapsulates all storage operations and allows customization
    of storage locations for testing and different environments.
    """

    def __init__(self, config: StorageConfig | None = None):
        self.config = config or StorageConfig()
        self._metadata_filename = "_atom.yaml"
        self._data_subdir = "data"

    def get_atom_index_path(self, atom_id: str) -> Path:
        """Get path to atom's index file."""
        return self.config.index_dir / f"{atom_id}.yaml"

    def get_atom_cache_path(self, atom_id: str) -> Path:
        """Get path to atom's data cache directory."""
        return self.config.data_dir / atom_id

    def get_atom_data_path(self, atom_id: str) -> Path:
        """Get path to atom's actual data directory."""
        return self.get_atom_cache_path(atom_id) / self._data_subdir

    def save_atom_metadata(self, atom: AtomProtocol) -> None:
        """Save atom metadata to index and cache atomically."""
        # Ensure directories exist
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.get_atom_cache_path(atom.id)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Serialize atom
        atom_dict = atom.to_dict()

        # Save to index atomically
        index_path = self.get_atom_index_path(atom.id)
        atomic_write_yaml(index_path, atom_dict)

        # Save to cache atomically
        cache_metadata_path = cache_path / self._metadata_filename
        atomic_write_yaml(cache_metadata_path, atom_dict)

    def move_artifact_to_storage(self, atom_id: str, artifact_path: Path) -> None:
        """Move artifact data to atom storage."""
        data_path = self.get_atom_data_path(atom_id)

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

    def find_atom_by_hash(self, content_hash: str) -> str | None:
        """Look up an atom ID by content hash."""
        hash_index = self.load_hash_index()
        return hash_index.get(content_hash)

    def load_hash_index(self) -> dict[str, str]:
        """Load the hash index (maps content_hash -> atom_id)."""
        if not self.config.hash_index.exists():
            return {}

        with open(self.config.hash_index) as f:
            result = yaml.safe_load(f)
            return result or {}

    def save_hash_index(self, hash_index: dict[str, str]) -> None:
        """Save the hash index to disk atomically."""
        self.config.base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(self.config.hash_index, hash_index)

    def register_atom_hash(self, content_hash: str, atom_id: str) -> None:
        """Register a content hash -> atom ID mapping."""
        with FileLock(self.config.hash_index_lock, timeout=10):
            hash_index = self.load_hash_index()
            hash_index[content_hash] = atom_id
            self.save_hash_index(hash_index)


# =====================================================================
# Default storage singleton for backward compatibility
# =====================================================================

_default_storage: AtomStorage | None = None


def get_default_storage() -> AtomStorage:
    """Get the default storage instance (singleton pattern)."""
    global _default_storage
    if _default_storage is None:
        _default_storage = AtomStorage(StorageConfig.from_env())
    return _default_storage


def set_default_storage(storage: AtomStorage) -> None:
    """Set the default storage instance."""
    global _default_storage
    _default_storage = storage


# Storage directories
ATOMS_BASE_DIR = Path(".motools/atoms")
ATOMS_INDEX_DIR = ATOMS_BASE_DIR / "index"
ATOMS_DATA_DIR = ATOMS_BASE_DIR / "data"
ATOMS_HASH_INDEX = ATOMS_BASE_DIR / "hash_index.yaml"
ATOMS_HASH_INDEX_LOCK = ATOMS_BASE_DIR / "hash_index.lock"
ATOM_METADATA_FILENAME = "_atom.yaml"
ATOM_DATA_SUBDIR = "data"


# =====================================================================
# Backward compatibility functions (delegate to default storage)
# =====================================================================


def get_atom_index_path(atom_id: str) -> Path:
    """Get path to atom's index file."""
    return get_default_storage().get_atom_index_path(atom_id)


def get_atom_cache_path(atom_id: str) -> Path:
    """Get path to atom's data cache directory."""
    return get_default_storage().get_atom_cache_path(atom_id)


def get_atom_data_path(atom_id: str) -> Path:
    """Get path to atom's actual data directory."""
    return get_default_storage().get_atom_data_path(atom_id)


def save_atom_metadata(atom: AtomProtocol) -> None:
    """Save atom metadata to index and cache atomically."""
    get_default_storage().save_atom_metadata(atom)


def move_artifact_to_storage(atom_id: str, artifact_path: Path) -> None:
    """Move artifact data to atom storage."""
    get_default_storage().move_artifact_to_storage(atom_id, artifact_path)


def list_atoms(atom_type: str | None = None) -> list[str]:
    """List all atom IDs, optionally filtered by type."""
    storage = get_default_storage()
    if not storage.config.index_dir.exists():
        return []

    atom_ids = []
    for yaml_file in storage.config.index_dir.glob("*.yaml"):
        atom_id = yaml_file.stem
        if atom_type is None or atom_id.startswith(f"{atom_type}-"):
            atom_ids.append(atom_id)

    return sorted(atom_ids)


def load_hash_index() -> dict[str, str]:
    """Load the hash index (maps content_hash -> atom_id)."""
    return get_default_storage().load_hash_index()


def save_hash_index(hash_index: dict[str, str]) -> None:
    """Save the hash index to disk atomically."""
    get_default_storage().save_hash_index(hash_index)


def find_atom_by_hash(content_hash: str) -> str | None:
    """Look up an atom ID by content hash."""
    return get_default_storage().find_atom_by_hash(content_hash)


def register_atom_hash(content_hash: str, atom_id: str) -> None:
    """Register a content hash -> atom ID mapping."""
    get_default_storage().register_atom_hash(content_hash, atom_id)


# =====================================================================
# Async versions of storage functions (primary API)
# =====================================================================


async def asave_atom_metadata(atom: AtomProtocol) -> None:
    """Save atom metadata to index and cache asynchronously with atomic writes.

    Uses atomic writes to prevent partial writes or corruption if interrupted.

    Args:
        atom: Atom instance to save
    """
    # Ensure directories exist
    await asyncio.to_thread(ATOMS_INDEX_DIR.mkdir, parents=True, exist_ok=True)
    cache_path = get_atom_cache_path(atom.id)
    await asyncio.to_thread(cache_path.mkdir, parents=True, exist_ok=True)

    # Serialize atom
    atom_dict = atom.to_dict()

    # Save to index atomically
    index_path = get_atom_index_path(atom.id)
    await atomic_write_yaml_async(index_path, atom_dict)

    # Save to cache atomically
    cache_metadata_path = cache_path / ATOM_METADATA_FILENAME
    await atomic_write_yaml_async(cache_metadata_path, atom_dict)


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
        return result or {}


async def asave_hash_index(hash_index: dict[str, str]) -> None:
    """Save the hash index to disk asynchronously with atomic writes.

    Uses atomic writes to prevent partial writes or corruption if interrupted.

    Args:
        hash_index: Dictionary mapping content hashes to atom IDs
    """
    await asyncio.to_thread(ATOMS_BASE_DIR.mkdir, parents=True, exist_ok=True)
    await atomic_write_yaml_async(ATOMS_HASH_INDEX, hash_index)


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
