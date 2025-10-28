"""Tests for TaskAtom class."""

import pickle
from pathlib import Path

import pytest

from motools.atom import Atom, TaskAtom
from motools.atom.storage import get_atom_data_path, get_atom_index_path


class MockTask:
    """Mock Inspect AI Task for testing."""

    def __init__(self, dataset, solver=None, scorer=None):
        self.dataset = dataset
        self.solver = solver
        self.scorer = scorer


@pytest.fixture
def task_artifact(tmp_path: Path) -> Path:
    """Create a temporary task artifact directory."""
    artifact_path = tmp_path / "task_artifact"
    artifact_path.mkdir()

    # Create a mock task and serialize it
    mock_task = MockTask(
        dataset=[{"input": "test", "target": "response"}],
        solver="mock_solver",
        scorer="mock_scorer",
    )
    with open(artifact_path / "task.pkl", "wb") as f:
        pickle.dump(mock_task, f)

    return artifact_path


def test_task_atom_create(task_artifact: Path) -> None:
    """Test creating a TaskAtom."""
    atom = TaskAtom.create(
        user="alice",
        artifact_path=task_artifact,
        made_from={"dataset": "dataset-alice-abc123"},
        metadata={"task_name": "test_task"},
    )

    assert atom.id.startswith("task-alice-")
    assert atom.type == "task"
    assert atom.metadata["task_name"] == "test_task"
    assert atom.made_from == {"dataset": "dataset-alice-abc123"}

    # Verify storage
    index_path = get_atom_index_path(atom.id)
    assert index_path.exists()

    data_path = get_atom_data_path(atom.id)
    assert data_path.exists()
    assert (data_path / "task.pkl").exists()


def test_task_atom_load_reload(task_artifact: Path) -> None:
    """Test TaskAtom can be loaded and returns correct type."""
    # Create atom
    atom = TaskAtom.create(
        user="bob",
        artifact_path=task_artifact,
        metadata={"task_name": "hello_world"},
    )

    # Load via Atom.load (should return TaskAtom)
    loaded = Atom.load(atom.id)
    assert isinstance(loaded, TaskAtom)
    assert loaded.id == atom.id
    assert loaded.type == "task"
    assert loaded.metadata["task_name"] == "hello_world"


@pytest.mark.asyncio
async def test_task_atom_from_task() -> None:
    """Test creating TaskAtom from a Task instance."""
    # Create a mock task
    mock_task = MockTask(
        dataset=[{"input": "hello", "target": "world"}],
        solver="generate",
        scorer="match",
    )

    # Create TaskAtom from task
    atom = await TaskAtom.from_task(
        task=mock_task,
        user="charlie",
        metadata={"task_name": "mock_task"},
    )

    assert atom.id.startswith("task-charlie-")
    assert atom.type == "task"
    assert atom.metadata["task_name"] == "mock_task"

    # Verify the task was serialized
    data_path = atom.get_data_path()
    assert (data_path / "task.pkl").exists()


@pytest.mark.asyncio
async def test_task_atom_to_task(task_artifact: Path) -> None:
    """Test loading a Task from TaskAtom."""
    # Create atom
    atom = TaskAtom.create(
        user="dave",
        artifact_path=task_artifact,
        metadata={"task_name": "test_task"},
    )

    # Load task from atom
    loaded_task = await atom.to_task()
    assert isinstance(loaded_task, MockTask)
    assert loaded_task.dataset == [{"input": "test", "target": "response"}]
    assert loaded_task.solver == "mock_solver"
    assert loaded_task.scorer == "mock_scorer"


@pytest.mark.asyncio
async def test_task_atom_to_task_missing_file(tmp_path: Path) -> None:
    """Test TaskAtom.to_task() raises if task.pkl missing."""
    # Create atom without task.pkl
    artifact_path = tmp_path / "empty_artifact"
    artifact_path.mkdir()
    (artifact_path / "dummy.txt").write_text("dummy")

    atom = TaskAtom.create(
        user="eve",
        artifact_path=artifact_path,
        metadata={"task_name": "empty"},
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="No task.pkl found"):
        await atom.to_task()


def test_task_atom_content_deduplication(task_artifact: Path, tmp_path: Path) -> None:
    """Test that identical tasks create the same atom (content-addressable)."""
    # Create a copy of the artifact for the second creation
    import shutil

    task_artifact_copy = tmp_path / "task_artifact_copy"
    shutil.copytree(task_artifact, task_artifact_copy)

    # Create first atom
    atom1 = TaskAtom.create(
        user="alice",
        artifact_path=task_artifact,
        metadata={"version": "1.0"},
    )

    # Create second atom with identical content
    atom2 = TaskAtom.create(
        user="alice",
        artifact_path=task_artifact_copy,
        metadata={"version": "1.0"},
    )

    # Should have the same ID (content-addressable deduplication)
    assert atom1.id == atom2.id
    assert atom1.content_hash == atom2.content_hash


def test_task_atom_different_metadata_different_hash(task_artifact: Path, tmp_path: Path) -> None:
    """Test that different metadata creates different atoms."""
    # Create a copy of the artifact for the second creation
    import shutil

    task_artifact_copy = tmp_path / "task_artifact_copy"
    shutil.copytree(task_artifact, task_artifact_copy)

    # Create first atom
    atom1 = TaskAtom.create(
        user="alice",
        artifact_path=task_artifact,
        metadata={"version": "1.0"},
    )

    # Create second atom with different metadata
    atom2 = TaskAtom.create(
        user="alice",
        artifact_path=task_artifact_copy,
        metadata={"version": "2.0"},
    )

    # Should have different IDs
    assert atom1.id != atom2.id
    assert atom1.content_hash != atom2.content_hash


@pytest.mark.asyncio
async def test_task_atom_roundtrip_with_complex_task() -> None:
    """Test roundtrip serialization with a more complex task."""
    # Create a complex mock task
    mock_task = MockTask(
        dataset=[
            {"input": f"prompt_{i}", "target": f"response_{i}", "id": f"sample_{i}"}
            for i in range(10)
        ],
        solver={"type": "generate", "max_tokens": 100},
        scorer={"type": "match", "case_sensitive": False},
    )

    # Create TaskAtom from task
    atom = await TaskAtom.from_task(
        task=mock_task,
        user="frank",
        metadata={"samples": 10},
    )

    # Load task back
    loaded_task = await atom.to_task()

    # Verify roundtrip
    assert isinstance(loaded_task, MockTask)
    assert len(loaded_task.dataset) == 10
    assert loaded_task.dataset[0] == {"input": "prompt_0", "target": "response_0", "id": "sample_0"}
    assert loaded_task.solver == {"type": "generate", "max_tokens": 100}
    assert loaded_task.scorer == {"type": "match", "case_sensitive": False}
