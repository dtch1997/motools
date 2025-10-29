"""Tests for PrepareTaskStep."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from motools.steps.prepare_task import PrepareTaskStep


@dataclass
class MockPrepareTaskConfig:
    """Mock config for PrepareTaskStep."""

    task_loader: str
    loader_kwargs: dict[str, Any] | None = None


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for test outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.mark.asyncio
async def test_prepare_task_step_execute(temp_workspace: Path) -> None:
    """Test PrepareTaskStep.execute() with synchronous loader."""
    step = PrepareTaskStep()

    # Create config
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_task",
        loader_kwargs={"param": "value"},
    )

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "task"
    assert constructor.type == "task"
    assert constructor.path == temp_workspace / "task_spec.json"

    # Verify metadata
    assert hasattr(constructor, "metadata")
    assert constructor.metadata == {"task_loader": "tests.test_module:get_task"}  # type: ignore[attr-defined]

    # Verify task spec was saved
    assert constructor.path.exists()
    with open(constructor.path) as f:
        task_spec = json.load(f)
    assert task_spec["task_loader"] == "tests.test_module:get_task"
    assert task_spec["loader_kwargs"] == {"param": "value"}


@pytest.mark.asyncio
async def test_prepare_task_step_execute_async_loader(temp_workspace: Path) -> None:
    """Test PrepareTaskStep.execute() with asynchronous loader."""
    step = PrepareTaskStep()

    # Create config
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:async_get_task",
    )

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "task"
    assert constructor.type == "task"

    # Verify task spec was saved
    assert constructor.path.exists()
    with open(constructor.path) as f:
        task_spec = json.load(f)
    assert task_spec["task_loader"] == "tests.test_module:async_get_task"
    assert task_spec["loader_kwargs"] == {}


@pytest.mark.asyncio
async def test_prepare_task_step_no_kwargs(temp_workspace: Path) -> None:
    """Test PrepareTaskStep.execute() without loader_kwargs."""
    step = PrepareTaskStep()

    # Create config without loader_kwargs
    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_task",
    )

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    assert constructors[0].type == "task"

    # Verify task spec has empty loader_kwargs
    with open(constructors[0].path) as f:
        task_spec = json.load(f)
    assert task_spec["loader_kwargs"] == {}


def test_prepare_task_step_class_metadata() -> None:
    """Test PrepareTaskStep class-level metadata."""
    from motools.steps.configs import PrepareTaskConfig

    assert PrepareTaskStep.name == "prepare_task"
    assert PrepareTaskStep.input_atom_types == {}
    assert PrepareTaskStep.output_atom_types == {"task": "task"}
    assert PrepareTaskStep.config_class is PrepareTaskConfig


def test_prepare_task_step_as_step() -> None:
    """Test PrepareTaskStep.as_step() creates a FunctionStep."""
    from motools.steps.configs import PrepareTaskConfig

    function_step = PrepareTaskStep.as_step()

    assert function_step.name == "prepare_task"
    assert function_step.input_atom_types == {}
    assert function_step.output_atom_types == {"task": "task"}
    assert function_step.config_class is PrepareTaskConfig
    assert callable(function_step.fn)


@pytest.mark.asyncio
async def test_prepare_task_step_with_complex_task(temp_workspace: Path) -> None:
    """Test PrepareTaskStep with a more complex task."""
    step = PrepareTaskStep()

    config = MockPrepareTaskConfig(
        task_loader="tests.test_module:get_complex_task",
        loader_kwargs={"num_samples": 10},
    )

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify the task spec was saved correctly
    assert len(constructors) == 1
    constructor = constructors[0]

    with open(constructor.path) as f:
        task_spec = json.load(f)

    assert task_spec["task_loader"] == "tests.test_module:get_complex_task"
    assert task_spec["loader_kwargs"] == {"num_samples": 10}
