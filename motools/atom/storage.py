"""Storage backend for atoms - local filesystem only."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from motools.atom.base import Atom

# Storage directories
ATOMS_BASE_DIR = Path(".motools/atoms")
ATOMS_INDEX_DIR = ATOMS_BASE_DIR / "index"
ATOMS_DATA_DIR = ATOMS_BASE_DIR / "data"
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


def save_atom_metadata(atom: Atom) -> None:
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
