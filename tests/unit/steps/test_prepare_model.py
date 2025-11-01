"""Tests for PrepareModelStep."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.steps.prepare_model import PrepareModelStep


@dataclass
class MockPrepareModelConfig:
    """Mock config for PrepareModelStep."""

    model_id: str


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for test outputs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.mark.asyncio
async def test_prepare_model_step_execute(temp_workspace: Path) -> None:
    """Test PrepareModelStep.execute() with a model ID."""
    step = PrepareModelStep()

    # Create config
    config = MockPrepareModelConfig(model_id="gpt-4")

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.name == "model"
    assert constructor.type == "model"
    assert constructor.path == temp_workspace

    # Verify metadata
    assert hasattr(constructor, "metadata")
    assert constructor.metadata == {"model_id": "gpt-4"}  # type: ignore[attr-defined]

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.exists()
    assert model_id_path.read_text() == "gpt-4"


@pytest.mark.asyncio
async def test_prepare_model_step_with_provider_prefix(temp_workspace: Path) -> None:
    """Test PrepareModelStep.execute() with provider-prefixed model ID."""
    step = PrepareModelStep()

    # Create config with provider prefix
    config = MockPrepareModelConfig(model_id="openai/gpt-4o")

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.metadata == {"model_id": "openai/gpt-4o"}  # type: ignore[attr-defined]

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.read_text() == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_prepare_model_step_with_anthropic_model(temp_workspace: Path) -> None:
    """Test PrepareModelStep.execute() with Anthropic model."""
    step = PrepareModelStep()

    # Create config with Anthropic model
    config = MockPrepareModelConfig(model_id="anthropic/claude-3.5-sonnet")

    # Execute the step
    constructors = await step.execute(config, {}, temp_workspace)

    # Verify results
    assert len(constructors) == 1
    constructor = constructors[0]

    assert constructor.metadata == {"model_id": "anthropic/claude-3.5-sonnet"}  # type: ignore[attr-defined]

    # Verify model_id was saved
    model_id_path = temp_workspace / "model_id.txt"
    assert model_id_path.read_text() == "anthropic/claude-3.5-sonnet"


def test_prepare_model_step_class_metadata() -> None:
    """Test PrepareModelStep class-level metadata."""
    from motools.steps.configs import PrepareModelConfig

    assert PrepareModelStep.name == "prepare_model"
    assert PrepareModelStep.input_atom_types == {}
    assert PrepareModelStep.output_atom_types == {"model": "model"}
    assert PrepareModelStep.config_class is PrepareModelConfig


def test_prepare_model_step_as_step() -> None:
    """Test PrepareModelStep.as_step() creates a FunctionStep."""
    from motools.steps.configs import PrepareModelConfig

    function_step = PrepareModelStep.as_step()

    assert function_step.name == "prepare_model"
    assert function_step.input_atom_types == {}
    assert function_step.output_atom_types == {"model": "model"}
    assert function_step.config_class is PrepareModelConfig
    assert callable(function_step.fn)
