"""Unit tests for SequentialRunner."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.atom import Atom, DatasetAtom, create_temp_workspace
from motools.workflow import AtomConstructor, Step, StepConfig, Workflow, WorkflowConfig
from motools.workflow.runners import SequentialRunner

# ============ Test Step Configs ============


@dataclass
class ProcessConfig(StepConfig):
    """Config for process step."""

    multiplier: int = 2


@dataclass
class AggregateConfig(StepConfig):
    """Config for aggregate step."""

    prefix: str = "total"


# ============ Test Step Functions ============


async def process_fn(
    config: ProcessConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Process input dataset."""
    input_data = input_atoms["input_data"]

    # Read input
    data_path = input_data.get_data_path()
    values = [int(line) for line in (data_path / "values.txt").read_text().splitlines()]

    # Process
    processed = [v * config.multiplier for v in values]

    # Write output
    (temp_workspace / "processed.txt").write_text("\n".join(str(v) for v in processed))

    return [
        AtomConstructor(
            name="processed_data", path=temp_workspace / "processed.txt", type="dataset"
        )
    ]


async def aggregate_fn(
    config: AggregateConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Aggregate processed data."""
    processed = input_atoms["processed_data"]

    # Read input
    data_path = processed.get_data_path()
    values = [int(line) for line in (data_path / "processed.txt").read_text().splitlines()]

    # Aggregate
    result = {config.prefix: sum(values)}

    # Write output
    (temp_workspace / "result.json").write_text(json.dumps(result))

    return [
        AtomConstructor(name="aggregated_data", path=temp_workspace / "result.json", type="dataset")
    ]


# ============ Test Workflow Definition ============


@dataclass
class TestWorkflowConfig(WorkflowConfig):
    """Config for test workflow."""

    process: ProcessConfig
    aggregate: AggregateConfig


test_workflow = Workflow(
    name="test_workflow",
    input_atom_types={"input_data": "dataset"},
    steps=[
        Step(
            name="process",
            input_atom_types={"input_data": "dataset"},
            output_atom_types={"processed_data": "dataset"},
            config_class=ProcessConfig,
            fn=process_fn,
        ),
        Step(
            name="aggregate",
            input_atom_types={"processed_data": "dataset"},
            output_atom_types={"aggregated_data": "dataset"},
            config_class=AggregateConfig,
            fn=aggregate_fn,
        ),
    ],
    config_class=TestWorkflowConfig,
)


# ============ Tests ============


def test_sequential_runner_run():
    """Test SequentialRunner.run() executes all steps sequentially."""
    runner = SequentialRunner()

    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2, 3, 4, 5]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create config
    config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=3),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run workflow
    state = runner.run(
        workflow=test_workflow,
        input_atoms={"input_data": input_atom.id},
        config=config,
        user="test",
        config_name="test_config",
    )

    # Verify all steps completed
    assert len(state.step_states) == 2
    assert state.step_states[0].status == "FINISHED"
    assert state.step_states[1].status == "FINISHED"

    # Verify intermediate step
    processed_id = state.step_states[0].output_atoms["processed_data"]
    processed_atom = DatasetAtom.load(processed_id)
    data_path = processed_atom.get_data_path()
    processed_values = [
        int(line) for line in (data_path / "processed.txt").read_text().splitlines()
    ]
    assert processed_values == [3, 6, 9, 12, 15]

    # Verify final output
    aggregated_id = state.step_states[1].output_atoms["aggregated_data"]
    aggregated_atom = DatasetAtom.load(aggregated_id)
    data_path = aggregated_atom.get_data_path()
    result = json.loads((data_path / "result.json").read_text())
    assert result == {"sum": 45}

    # Verify workflow state
    assert state.workflow_name == "test_workflow"
    assert state.config_name == "test_config"
    assert state.time_started is not None
    assert state.time_finished is not None


def test_sequential_runner_run_step():
    """Test SequentialRunner.run_step() executes a single step."""
    runner = SequentialRunner()

    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2, 3]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create config
    config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=2),
        aggregate=AggregateConfig(prefix="total"),
    )

    # Initialize workflow state manually
    from datetime import UTC, datetime

    from motools.workflow.state import StepState, WorkflowState

    state = WorkflowState(
        workflow_name=test_workflow.name,
        input_atoms={"input_data": input_atom.id},
        config=config,
        config_name="test",
        time_started=datetime.now(UTC),
    )

    # Add step states
    for step in test_workflow.steps:
        step_config = getattr(config, step.name, None)
        state.step_states.append(StepState(step_name=step.name, config=step_config))

    # Run first step only
    state = runner.run_step(test_workflow, state, "process", "test")

    # Verify first step completed
    assert state.step_states[0].status == "FINISHED"
    assert state.step_states[1].status == "PENDING"

    # Verify output
    processed_id = state.step_states[0].output_atoms["processed_data"]
    processed_atom = DatasetAtom.load(processed_id)
    data_path = processed_atom.get_data_path()
    processed_values = [
        int(line) for line in (data_path / "processed.txt").read_text().splitlines()
    ]
    assert processed_values == [2, 4, 6]


def test_sequential_runner_validates_inputs():
    """Test SequentialRunner validates workflow inputs."""
    runner = SequentialRunner()

    # Create config
    config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=2),
        aggregate=AggregateConfig(prefix="total"),
    )

    # Test missing input
    with pytest.raises(ValueError, match="missing required inputs"):
        runner.run(
            workflow=test_workflow,
            input_atoms={},  # Missing required input
            config=config,
            user="test",
        )


def test_sequential_runner_handles_step_failure():
    """Test SequentialRunner handles step failures gracefully."""

    def failing_fn(
        config: ProcessConfig,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """A step that always fails."""
        raise RuntimeError("Intentional failure")

    # Create workflow with failing step
    failing_workflow = Workflow(
        name="failing_workflow",
        input_atom_types={"input_data": "dataset"},
        steps=[
            Step(
                name="failing_step",
                input_atom_types={"input_data": "dataset"},
                output_atom_types={"output_data": "dataset"},
                config_class=ProcessConfig,
                fn=failing_fn,
            ),
        ],
        config_class=TestWorkflowConfig,
    )

    runner = SequentialRunner()

    # Create input atom
    with create_temp_workspace() as temp:
        (temp / "test.txt").write_text("test")
        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "test.txt", metadata={"source": "test"}
        )

    config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=2),
        aggregate=AggregateConfig(prefix="total"),
    )

    # Run and expect failure
    with pytest.raises(RuntimeError, match="Intentional failure"):
        runner.run(
            workflow=failing_workflow,
            input_atoms={"input_data": input_atom.id},
            config=config,
            user="test",
        )
