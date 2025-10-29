"""Unit tests for run_sweep utility."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from motools.atom import Atom, DatasetAtom, create_temp_workspace
from motools.workflow import (
    AtomConstructor,
    FunctionStep,
    StepConfig,
    Workflow,
    WorkflowConfig,
    run_sweep,
)

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
        FunctionStep(
            name="process",
            input_atom_types={"input_data": "dataset"},
            output_atom_types={"processed_data": "dataset"},
            config_class=ProcessConfig,
            fn=process_fn,
        ),
        FunctionStep(
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


@pytest.mark.asyncio
async def test_run_sweep_basic():
    """Test run_sweep executes workflow with different parameter combinations."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2, 3]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),  # Will be overridden
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={
            "process": [
                ProcessConfig(multiplier=2),
                ProcessConfig(multiplier=3),
            ],
        },
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    # Should have run 2 times (2 multiplier values)
    assert len(states) == 2

    # Verify all runs completed
    for state in states:
        assert len(state.step_states) == 2
        assert state.step_states[0].status == "FINISHED"
        assert state.step_states[1].status == "FINISHED"

    # Verify first run (multiplier=2)
    aggregated_id_0 = states[0].step_states[1].output_atoms["aggregated_data"]
    aggregated_atom_0 = DatasetAtom.load(aggregated_id_0)
    data_path_0 = aggregated_atom_0.get_data_path()
    result_0 = json.loads((data_path_0 / "result.json").read_text())
    assert result_0 == {"sum": 12}  # (1+2+3) * 2 = 12

    # Verify second run (multiplier=3)
    aggregated_id_1 = states[1].step_states[1].output_atoms["aggregated_data"]
    aggregated_atom_1 = DatasetAtom.load(aggregated_id_1)
    data_path_1 = aggregated_atom_1.get_data_path()
    result_1 = json.loads((data_path_1 / "result.json").read_text())
    assert result_1 == {"sum": 18}  # (1+2+3) * 3 = 18


@pytest.mark.asyncio
async def test_run_sweep_cartesian_product():
    """Test run_sweep generates cartesian product of parameters."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep with 2x2 grid
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={
            "process": [ProcessConfig(multiplier=2), ProcessConfig(multiplier=3)],
            "aggregate": [AggregateConfig(prefix="sum"), AggregateConfig(prefix="total")],
        },
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    # Should have run 4 times (2 * 2 combinations)
    assert len(states) == 4

    # Verify all runs completed
    for state in states:
        assert all(step.status == "FINISHED" for step in state.step_states)

    # Extract all unique prefix values from results
    prefixes = set()
    for state in states:
        aggregated_id = state.step_states[1].output_atoms["aggregated_data"]
        aggregated_atom = DatasetAtom.load(aggregated_id)
        data_path = aggregated_atom.get_data_path()
        result = json.loads((data_path / "result.json").read_text())
        prefixes.update(result.keys())

    # Should have both prefixes
    assert prefixes == {"sum", "total"}


@pytest.mark.asyncio
async def test_run_sweep_max_parallel():
    """Test run_sweep respects max_parallel limit."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep with max_parallel=2
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={
            "process": [ProcessConfig(multiplier=i) for i in range(1, 5)],  # 4 combinations
        },
        input_atoms={"input_data": input_atom.id},
        user="test",
        max_parallel=2,
    )

    # Should have run 4 times
    assert len(states) == 4

    # All should complete successfully
    for state in states:
        assert all(step.status == "FINISHED" for step in state.step_states)


@pytest.mark.asyncio
async def test_run_sweep_empty_grid():
    """Test run_sweep handles empty parameter grid."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=2),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep with empty grid (should run once with base config)
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={},
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    # Should have run once with base config
    assert len(states) == 1
    assert states[0].config.process.multiplier == 2


@pytest.mark.asyncio
async def test_run_sweep_nested_parameters():
    """Test run_sweep supports nested parameter paths."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2, 3]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),  # Will be overridden
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep with nested parameters
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={
            "process.multiplier": [2, 3, 4],  # Nested parameter
            "aggregate.prefix": ["total", "sum"],  # Another nested parameter
        },
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    # Should have run 6 times (3 * 2 combinations)
    assert len(states) == 6

    # Verify all runs completed
    for state in states:
        assert len(state.step_states) == 2
        assert state.step_states[0].status == "FINISHED"
        assert state.step_states[1].status == "FINISHED"

    # Verify we got all expected combinations
    multipliers = set()
    prefixes = set()

    for state in states:
        # Check multiplier from config
        multipliers.add(state.config.process.multiplier)

        # Check prefix from result
        aggregated_id = state.step_states[1].output_atoms["aggregated_data"]
        aggregated_atom = DatasetAtom.load(aggregated_id)
        data_path = aggregated_atom.get_data_path()
        result = json.loads((data_path / "result.json").read_text())
        prefixes.update(result.keys())

    assert multipliers == {2, 3, 4}
    assert prefixes == {"total", "sum"}


@pytest.mark.asyncio
async def test_run_sweep_mixed_nested_and_flat():
    """Test run_sweep handles both nested and flat parameters together."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [10]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep with mixed parameters
    states = await run_sweep(
        workflow=test_workflow,
        base_config=base_config,
        param_grid={
            "process.multiplier": [5, 10],  # Nested parameter
            "aggregate": [  # Flat parameter (whole object)
                AggregateConfig(prefix="result"),
                AggregateConfig(prefix="output"),
            ],
        },
        input_atoms={"input_data": input_atom.id},
        user="test",
    )

    # Should have run 4 times (2 * 2 combinations)
    assert len(states) == 4

    # Verify all runs completed
    for state in states:
        assert all(step.status == "FINISHED" for step in state.step_states)

    # Verify combinations
    results = []
    for state in states:
        multiplier = state.config.process.multiplier
        aggregated_id = state.step_states[1].output_atoms["aggregated_data"]
        aggregated_atom = DatasetAtom.load(aggregated_id)
        data_path = aggregated_atom.get_data_path()
        result = json.loads((data_path / "result.json").read_text())
        prefix = list(result.keys())[0]
        value = result[prefix]
        results.append((multiplier, prefix, value))

    # Check we got all expected combinations
    expected = [
        (5, "result", 50),  # 10 * 5
        (5, "output", 50),  # 10 * 5
        (10, "result", 100),  # 10 * 10
        (10, "output", 100),  # 10 * 10
    ]

    assert sorted(results) == sorted(expected)


@pytest.mark.asyncio
async def test_run_sweep_handles_workflow_failures(caplog):
    """Test run_sweep handles individual workflow failures gracefully."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2, 3]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Mock run_workflow to simulate failures for certain parameter combinations
    from motools.workflow.execution import run_workflow

    original_run_workflow = run_workflow

    async def mock_run_workflow(workflow, input_atoms, config, user, config_name):
        # Simulate failure when multiplier is 3
        if config.process.multiplier == 3:
            raise RuntimeError("Simulated training failure")
        # Success for other multipliers
        return await original_run_workflow(workflow, input_atoms, config, user, config_name)

    # Test with multiple parameter combinations where some fail
    with patch("motools.workflow.execution.run_workflow", side_effect=mock_run_workflow):
        with caplog.at_level(logging.WARNING):
            states = await run_sweep(
                workflow=test_workflow,
                base_config=base_config,
                param_grid={
                    "process": [
                        ProcessConfig(multiplier=2),  # Should succeed
                        ProcessConfig(multiplier=3),  # Should fail
                        ProcessConfig(multiplier=4),  # Should succeed
                        ProcessConfig(multiplier=5),  # Should succeed
                    ],
                },
                input_atoms={"input_data": input_atom.id},
                user="test",
            )

    # Should return only successful workflows (3 out of 4)
    assert len(states) == 3

    # All returned states should be successful
    for state in states:
        assert all(step.status == "FINISHED" for step in state.step_states)
        # Should not include the failed multiplier=3
        assert state.config.process.multiplier in [2, 4, 5]

    # Check that failure was logged with parameter information
    log_messages = caplog.text
    assert "⚠️  Workflow 1 failed with parameters" in log_messages
    assert "{'process': ProcessConfig(multiplier=3)}" in log_messages
    assert "Simulated training failure" in log_messages

    # Check summary logging
    assert "3/4 workflows succeeded, 1 failed" in log_messages


@pytest.mark.asyncio
async def test_run_sweep_all_workflows_fail(caplog):
    """Test run_sweep handles case where all workflows fail."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Mock run_workflow to always fail
    async def failing_run_workflow(*args, **kwargs):
        raise RuntimeError("All workflows failing")

    with patch("motools.workflow.execution.run_workflow", side_effect=failing_run_workflow):
        with caplog.at_level(logging.WARNING):
            states = await run_sweep(
                workflow=test_workflow,
                base_config=base_config,
                param_grid={
                    "process": [
                        ProcessConfig(multiplier=2),
                        ProcessConfig(multiplier=3),
                    ],
                },
                input_atoms={"input_data": input_atom.id},
                user="test",
            )

    # Should return empty list when all workflows fail
    assert len(states) == 0

    # Check that all failures were logged
    log_messages = caplog.text
    assert "⚠️  Workflow 0 failed" in log_messages
    assert "⚠️  Workflow 1 failed" in log_messages
    assert "All workflows failing" in log_messages

    # Check summary logging shows all failed
    assert "0/2 workflows succeeded, 2 failed" in log_messages


@pytest.mark.asyncio
async def test_run_sweep_all_workflows_succeed(caplog):
    """Test run_sweep logging when all workflows succeed."""
    # Create input atom
    with create_temp_workspace() as temp:
        values = [1, 2]
        (temp / "values.txt").write_text("\n".join(str(v) for v in values))

        input_atom = DatasetAtom.create(
            user="test", artifact_path=temp / "values.txt", metadata={"source": "test"}
        )

    # Create base config
    base_config = TestWorkflowConfig(
        process=ProcessConfig(multiplier=1),
        aggregate=AggregateConfig(prefix="sum"),
    )

    # Run sweep without any mocking (all should succeed)
    with caplog.at_level(logging.INFO):
        states = await run_sweep(
            workflow=test_workflow,
            base_config=base_config,
            param_grid={
                "process": [
                    ProcessConfig(multiplier=2),
                    ProcessConfig(multiplier=3),
                ],
            },
            input_atoms={"input_data": input_atom.id},
            user="test",
        )

    # Should return all workflows
    assert len(states) == 2

    # All should be successful
    for state in states:
        assert all(step.status == "FINISHED" for step in state.step_states)

    # Check success logging
    log_messages = caplog.text
    assert "Sweep completed successfully: 2/2 workflows" in log_messages
    # Should not have any failure warnings
    assert "⚠️" not in log_messages
