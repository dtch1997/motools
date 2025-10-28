"""Tests for PrepareTaskStep."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from motools.steps.prepare_task import PrepareTaskStep


@dataclass
class MockPrepareTaskConfig:
    """Mock config for PrepareTaskStep."""

    task_loader: str
    loader_kwargs: dict[str, Any] | None = None


class MockTask:
    """Mock Inspect AI Task for testing."""

    def __init__(self, dataset, solver=None, scorer=None):
        self.dataset = dataset
        self.solver = solver
        self.scorer = scorer


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for test outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def mock_task() -> MockTask:
    """Create a mock task for testing."""
    return MockTask(
        dataset=[{"input": "test", "target": "response"}],
        solver="mock_solver",
        scorer="mock_scorer",
    )


@pytest.mark.asyncio
async def test_prepare_task_step_execute(temp_workspace: Path, mock_task: MockTask) -> None:
    """Test PrepareTaskStep.execute() with synchronous loader."""
    step = PrepareTaskStep()

    # Create config
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_task",
        loader_kwargs={"param": "value"},
    )

    # Mock the import_function to return a function that returns our mock task
    with patch("motools.steps.prepare_task.import_function") as mock_import:
        mock_loader = MagicMock(return_value=mock_task)
        mock_import.return_value = mock_loader

        # Execute the step
        constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "prepared_task"
    assert constructor.type == "task"
    assert constructor.path == temp_workspace / "task.pkl"

    # Verify metadata
    assert hasattr(constructor, "metadata")
    assert constructor.metadata == {"task_loader": "tests.test_module:get_task"}  # type: ignore[attr-defined]

    # Verify the loader was called with kwargs
    mock_loader.assert_called_once_with(param="value")

    # Verify task was serialized
    assert constructor.path.exists()
    with open(constructor.path, "rb") as f:
        loaded_task = pickle.load(f)
    assert isinstance(loaded_task, MockTask)
    assert loaded_task.dataset == mock_task.dataset


@pytest.mark.asyncio
async def test_prepare_task_step_execute_async_loader(
    temp_workspace: Path, mock_task: MockTask
) -> None:
    """Test PrepareTaskStep.execute() with asynchronous loader."""
    step = PrepareTaskStep()

    # Create config
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:async_get_task",
    )

    # Mock the import_function to return an async function
    with patch("motools.steps.prepare_task.import_function") as mock_import:

        async def async_loader():
            return mock_task

        # Create a coroutine that returns the mock task
        mock_import.return_value = lambda: async_loader()

        # Execute the step
        constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "prepared_task"
    assert constructor.type == "task"

    # Verify task was serialized
    assert constructor.path.exists()
    with open(constructor.path, "rb") as f:
        loaded_task = pickle.load(f)
    assert isinstance(loaded_task, MockTask)


@pytest.mark.asyncio
async def test_prepare_task_step_no_kwargs(temp_workspace: Path, mock_task: MockTask) -> None:
    """Test PrepareTaskStep.execute() without loader_kwargs."""
    step = PrepareTaskStep()

    # Create config without loader_kwargs
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_task",
    )

    # Mock the import_function
    with patch("motools.steps.prepare_task.import_function") as mock_import:
        mock_loader = MagicMock(return_value=mock_task)
        mock_import.return_value = mock_loader

        # Execute the step
        constructors = await step.execute(config, {}, temp_workspace)

    # Verify the loader was called without kwargs
    mock_loader.assert_called_once_with()

    # Verify results
    assert len(constructors) == 1
    assert constructors[0].type == "task"


def test_prepare_task_step_class_metadata() -> None:
    """Test PrepareTaskStep class-level metadata."""
    assert PrepareTaskStep.name == "prepare_task"
    assert PrepareTaskStep.input_atom_types == {}
    assert PrepareTaskStep.output_atom_types == {"prepared_task": "task"}
    assert PrepareTaskStep.config_class is Any  # Will be replaced with actual config class


def test_prepare_task_step_as_step() -> None:
    """Test PrepareTaskStep.as_step() creates a FunctionStep."""
    function_step = PrepareTaskStep.as_step()

    assert function_step.name == "prepare_task"
    assert function_step.input_atom_types == {}
    assert function_step.output_atom_types == {"prepared_task": "task"}
    assert function_step.config_class is Any
    assert callable(function_step.fn)


@pytest.mark.asyncio
async def test_prepare_task_step_with_complex_task(temp_workspace: Path) -> None:
    """Test PrepareTaskStep with a more complex task."""
    step = PrepareTaskStep()

    # Create a complex task
    complex_task = MockTask(
        dataset=[
            {"input": f"prompt_{i}", "target": f"response_{i}", "id": f"sample_{i}"}
            for i in range(10)
        ],
        solver={"type": "generate", "max_tokens": 100},
        scorer={"type": "match", "case_sensitive": False},
    )

    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_complex_task",
        loader_kwargs={"num_samples": 10},
    )

    # Mock the import_function
    with patch("motools.steps.prepare_task.import_function") as mock_import:
        mock_loader = MagicMock(return_value=complex_task)
        mock_import.return_value = mock_loader

        # Execute the step
        constructors = await step.execute(config, {}, temp_workspace)

    # Verify the complex task was serialized correctly
    assert len(constructors) == 1
    constructor = constructors[0]

    with open(constructor.path, "rb") as f:
        loaded_task = pickle.load(f)

    assert len(loaded_task.dataset) == 10
    assert loaded_task.dataset[0]["id"] == "sample_0"
    assert loaded_task.solver["max_tokens"] == 100
