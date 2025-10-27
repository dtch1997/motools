"""Unit tests for SequentialRunner error handling improvements."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from motools.atom import Atom, DatasetAtom, create_temp_workspace
from motools.workflow import AtomConstructor, FunctionStep, StepConfig, Workflow, WorkflowConfig
from motools.workflow.errors import (
    NetworkError,
    ValidationError,
    WorkflowTimeoutError,
)
from motools.workflow.runners import SequentialRunner
from motools.workflow.state import StepState, WorkflowState

# ============ Test Step Configs ============


@dataclass
class ErrorTestConfig(StepConfig):
    """Config for test steps."""

    fail_type: str = "none"  # none, network, timeout, validation, generic
    fail_count: int = 0  # Number of times to fail before succeeding


# ============ Test Step Functions ============


async def flaky_network_step(
    config: ErrorTestConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """A step that can fail with network errors."""
    # Track call count (stored as a class attribute for simplicity in tests)
    if not hasattr(flaky_network_step, "call_count"):
        flaky_network_step.call_count = 0
    flaky_network_step.call_count += 1

    if flaky_network_step.call_count <= config.fail_count:
        if config.fail_type == "network":
            raise NetworkError("Network connection failed")
        elif config.fail_type == "timeout":
            raise WorkflowTimeoutError("Operation timed out")
        elif config.fail_type == "validation":
            raise ValidationError("Invalid configuration")
        elif config.fail_type == "generic":
            raise RuntimeError("Generic error")

    # Success case
    (temp_workspace / "output.txt").write_text("success")
    return [AtomConstructor(name="output_data", path=temp_workspace / "output.txt", type="dataset")]


async def step_with_cleanup(
    config: ErrorTestConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """A step that needs cleanup."""
    if config.fail_type != "none":
        raise RuntimeError("Step failed")

    (temp_workspace / "output.txt").write_text("success")
    return [AtomConstructor(name="output_data", path=temp_workspace / "output.txt", type="dataset")]


# Add cleanup method to the step
step_with_cleanup.cleanup = AsyncMock()


# ============ Test Workflow Definitions ============


@dataclass
class ErrorTestWorkflowConfig(WorkflowConfig):
    """Config for error test workflow."""

    test_step: ErrorTestConfig


# ============ Tests ============


@pytest.mark.asyncio
async def test_full_stack_trace_captured():
    """Test that full stack traces are captured in step_state.error."""
    runner = SequentialRunner()

    # Create workflow that always fails
    failing_workflow = Workflow(
        name="failing_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=lambda c, i, t: (_ for _ in ()).throw(ValueError("Test error with traceback")),
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    config = ErrorTestWorkflowConfig(test_step=ErrorTestConfig())

    # Run and expect failure
    with pytest.raises(RuntimeError, match="Test error with traceback"):
        await runner.run(
            workflow=failing_workflow,
            input_atoms={"input_data": input_atom.id},
            config=config,
            user="test",
        )

    # Get the state to check error field
    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=failing_workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Run the step directly to check error field
    with pytest.raises(RuntimeError):
        state = await runner.run_step(failing_workflow, state, "test_step", "test")

    # Verify stack trace is captured
    error_text = state.step_states[0].error
    assert error_text is not None
    assert "Traceback" in error_text
    assert "ValueError: Test error with traceback" in error_text
    assert "sequential.py" in error_text  # File name should be in the trace


@pytest.mark.asyncio
async def test_retry_on_network_error():
    """Test that network errors trigger retries."""
    # Reset call count
    if hasattr(flaky_network_step, "call_count"):
        flaky_network_step.call_count = 0

    runner = SequentialRunner()

    workflow = Workflow(
        name="network_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=flaky_network_step,
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    # Configure to fail twice with network errors, then succeed
    config = ErrorTestWorkflowConfig(test_step=ErrorTestConfig(fail_type="network", fail_count=2))

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Run with retries - should eventually succeed
    state = await runner.run_step(
        workflow, state, "test_step", "test", max_retries=3, retry_delay=0.1
    )

    # Verify it succeeded after retries
    assert state.step_states[0].status == "FINISHED"
    assert flaky_network_step.call_count == 3  # Failed twice, succeeded on third


@pytest.mark.asyncio
async def test_retry_on_timeout_error():
    """Test that timeout errors trigger retries."""
    # Reset call count
    if hasattr(flaky_network_step, "call_count"):
        flaky_network_step.call_count = 0

    runner = SequentialRunner()

    workflow = Workflow(
        name="timeout_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=flaky_network_step,
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    # Configure to fail once with timeout, then succeed
    config = ErrorTestWorkflowConfig(test_step=ErrorTestConfig(fail_type="timeout", fail_count=1))

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Run with retries - should succeed after one retry
    state = await runner.run_step(
        workflow, state, "test_step", "test", max_retries=2, retry_delay=0.1
    )

    assert state.step_states[0].status == "FINISHED"
    assert flaky_network_step.call_count == 2  # Failed once, succeeded on second


@pytest.mark.asyncio
async def test_no_retry_on_validation_error():
    """Test that validation errors do not trigger retries."""
    # Reset call count
    if hasattr(flaky_network_step, "call_count"):
        flaky_network_step.call_count = 0

    runner = SequentialRunner()

    workflow = Workflow(
        name="validation_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=flaky_network_step,
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    # Configure to fail with validation error
    config = ErrorTestWorkflowConfig(
        test_step=ErrorTestConfig(fail_type="validation", fail_count=1)
    )

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Run with retries - should fail immediately without retrying
    with pytest.raises(RuntimeError, match="Invalid configuration"):
        await runner.run_step(workflow, state, "test_step", "test", max_retries=3, retry_delay=0.1)

    # Verify it only tried once (no retries for validation errors)
    assert flaky_network_step.call_count == 1


@pytest.mark.asyncio
async def test_cleanup_runs_on_failure():
    """Test that cleanup is called even when step fails."""
    runner = SequentialRunner()

    # Reset mock
    step_with_cleanup.cleanup.reset_mock()

    # Create a mock step with cleanup
    mock_step = MagicMock()
    mock_step.name = "test_step"
    mock_step.input_atom_types = {"input_data": "dataset"}
    mock_step.output_atom_types = {"output_data": "dataset"}
    mock_step.execute = AsyncMock(side_effect=RuntimeError("Step failed"))
    mock_step.cleanup = AsyncMock()
    mock_step.validate_outputs = MagicMock(return_value=[])

    workflow = MagicMock()
    workflow.name = "cleanup_workflow"
    workflow.steps_by_name = {"test_step": mock_step}

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=None,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=None))

    # Run and expect failure
    with pytest.raises(RuntimeError, match="Step failed"):
        await runner.run_step(workflow, state, "test_step", "test", max_retries=0)

    # Verify cleanup was called
    mock_step.cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test that retry delays use exponential backoff."""
    runner = SequentialRunner()

    # Track sleep calls
    sleep_calls = []

    async def mock_sleep(delay):
        sleep_calls.append(delay)

    # Create a step that fails multiple times
    call_count = {"count": 0}

    async def failing_step(config, input_atoms, temp_workspace):
        call_count["count"] += 1
        if call_count["count"] <= 3:
            raise NetworkError("Network error")
        (temp_workspace / "output.txt").write_text("success")
        return [
            AtomConstructor(name="output_data", path=temp_workspace / "output.txt", type="dataset")
        ]

    workflow = Workflow(
        name="backoff_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=failing_step,
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    config = ErrorTestWorkflowConfig(test_step=ErrorTestConfig())

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Patch asyncio.sleep to track delays
    with patch("asyncio.sleep", side_effect=mock_sleep):
        state = await runner.run_step(
            workflow, state, "test_step", "test", max_retries=3, retry_delay=1.0
        )

    # Verify exponential backoff: 1.0, 2.0, 4.0
    assert len(sleep_calls) == 3
    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0
    assert sleep_calls[2] == 4.0


@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """Test that error is raised after max retries are exceeded."""
    # Reset call count
    if hasattr(flaky_network_step, "call_count"):
        flaky_network_step.call_count = 0

    runner = SequentialRunner()

    workflow = Workflow(
        name="max_retry_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            FunctionStep(
                name="test_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ErrorTestConfig,
                fn=flaky_network_step,
            ),
        ],
        config_class=ErrorTestWorkflowConfig,
    )

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(user="test", artifact_path=temp / "test.txt", metadata={})

    # Configure to always fail
    config = ErrorTestWorkflowConfig(test_step=ErrorTestConfig(fail_type="network", fail_count=10))

    from datetime import UTC, datetime

    state = WorkflowState(
        workflow_name=workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )
    state.step_states.append(StepState(step_name="test_step", config=config.test_step))

    # Run with max 2 retries - should fail
    with pytest.raises(RuntimeError, match="failed after 2 retries"):
        await runner.run_step(workflow, state, "test_step", "test", max_retries=2, retry_delay=0.1)

    # Verify it tried max_retries + 1 times
    assert flaky_network_step.call_count == 3  # Initial + 2 retries
