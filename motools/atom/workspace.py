"""Temporary workspace utilities for building atoms."""

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

TEMP_WORKSPACE_DIR = Path(".motools/temp")


@contextmanager
def create_temp_workspace():
    """Create a temporary workspace for building atom artifacts.

    This creates a unique temporary directory where you can build your data,
    then pass the directory path to Atom.create(). The directory will be
    automatically cleaned up after the atom is created (data is moved, not copied).

    Usage:
        with create_temp_workspace() as temp_dir:
            # Build your data in temp_dir
            (temp_dir / "train.jsonl").write_text("...")
            (temp_dir / "test.jsonl").write_text("...")

            # Create atom (moves data from temp_dir)
            atom = DatasetAtom.create(
                user="alice",
                artifact_path=temp_dir,
                metadata={"samples": 100}
            )

    Yields:
        Path to temporary workspace directory
    """
    workspace_id = uuid.uuid4().hex[:8]
    temp_dir = TEMP_WORKSPACE_DIR / workspace_id

    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        yield temp_dir
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
