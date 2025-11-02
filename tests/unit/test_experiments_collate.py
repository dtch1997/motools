"""Tests for sweep collation utilities."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from motools.atom.base import EvalAtom
from motools.evals.backends.inspect import InspectEvalResults
from motools.experiments.analysis.collate import collate_sweep_evals
from motools.workflow.state import StepState, WorkflowState


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    lr: float
    dropout: float
    batch_size: int = 32


async def create_mock_eval_atom(
    user: str,
    metrics: dict[str, dict[str, Any]],
    model_id: str = "test-model",
) -> str:
    """Create a mock EvalAtom for testing.

    Args:
        user: User identifier
        metrics: Metrics dict (task_name -> metric_dict)
        model_id: Model ID

    Returns:
        Atom ID
    """
    # Create mock eval results
    samples = []
    for task_name in metrics.keys():
        samples.append(
            {
                "task": task_name,
                "id": 1,
                "input": "test input",
                "target": "test target",
                "messages": [],
                "output": {},
                "scores": {},
            }
        )

    eval_results = InspectEvalResults(
        model_id=model_id,
        samples=samples,
        metrics=metrics,
        metadata={},
    )

    # Create temp directory for artifact
    temp_dir = Path(tempfile.mkdtemp())
    results_file = temp_dir / "results.json"

    await eval_results.save(str(results_file))

    # Create EvalAtom (this moves the directory to atom storage)
    atom = await EvalAtom.acreate(
        user=user,
        artifact_path=temp_dir,
        metadata={"model_id": model_id},
    )

    return atom.id


@pytest.mark.asyncio
class TestCollateSweepEvals:
    """Tests for collate_sweep_evals function."""

    async def test_basic_collation(self):
        """Test basic sweep collation with all metrics."""
        # Create mock sweep states
        configs = [
            MockConfig(lr=1e-3, dropout=0.1),
            MockConfig(lr=1e-4, dropout=0.1),
            MockConfig(lr=1e-3, dropout=0.2),
        ]

        sweep_states = []
        for i, config in enumerate(configs):
            # Create eval atom with metrics
            eval_atom_id = await create_mock_eval_atom(
                user="test",
                metrics={
                    "task1": {"accuracy": 0.9 + i * 0.01, "f1": 0.85 + i * 0.01},
                },
                model_id=f"model-{i}",
            )

            # Create workflow state
            step_state = StepState(
                step_name="evaluate",
                config={},
                status="FINISHED",
                output_atoms={"eval": eval_atom_id},
            )

            workflow_state = WorkflowState(
                workflow_name="test_workflow",
                input_atoms={},
                config=config,
                step_states=[step_state],
            )

            sweep_states.append(workflow_state)

        # Collate results
        df = await collate_sweep_evals(sweep_states, eval_step_name="evaluate")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "lr" in df.columns
        assert "dropout" in df.columns
        assert "task" in df.columns
        assert "accuracy" in df.columns
        assert "f1" in df.columns

        # Verify values
        assert df["lr"].tolist() == [1e-3, 1e-4, 1e-3]
        assert df["dropout"].tolist() == [0.1, 0.1, 0.2]
        assert df["accuracy"].tolist() == pytest.approx([0.9, 0.91, 0.92])

    async def test_single_metric_selection(self):
        """Test selecting a single metric."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={"task1": {"accuracy": 0.9, "f1": 0.85, "precision": 0.88}},
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        # Collate with single metric
        df = await collate_sweep_evals([workflow_state], "evaluate", metrics="accuracy")

        assert "accuracy" in df.columns
        assert "f1" not in df.columns
        assert "precision" not in df.columns

    async def test_multiple_metric_selection(self):
        """Test selecting multiple metrics."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={"task1": {"accuracy": 0.9, "f1": 0.85, "precision": 0.88}},
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        # Collate with multiple metrics
        df = await collate_sweep_evals([workflow_state], "evaluate", metrics=["accuracy", "f1"])

        assert "accuracy" in df.columns
        assert "f1" in df.columns
        assert "precision" not in df.columns

    async def test_multiple_tasks(self):
        """Test collation with multiple evaluation tasks."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={
                "task1": {"accuracy": 0.9},
                "task2": {"accuracy": 0.85},
                "task3": {"accuracy": 0.95},
            },
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        # Collate results
        df = await collate_sweep_evals([workflow_state], "evaluate")

        # Should have one row per task
        assert len(df) == 3
        assert set(df["task"].tolist()) == {"task1", "task2", "task3"}
        assert df["accuracy"].tolist() == pytest.approx([0.9, 0.85, 0.95])

    async def test_empty_sweep_states_raises(self):
        """Test that empty sweep_states raises ValueError."""
        with pytest.raises(ValueError, match="sweep_states cannot be empty"):
            await collate_sweep_evals([], "evaluate")

    async def test_missing_step_raises(self):
        """Test that missing step raises ValueError."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[],  # No steps
        )

        with pytest.raises(ValueError, match="Step 'evaluate' not found"):
            await collate_sweep_evals([workflow_state], "evaluate")

    async def test_incomplete_step_raises(self):
        """Test that incomplete step raises ValueError."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FAILED",  # Not FINISHED
            output_atoms={},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        with pytest.raises(ValueError, match="did not complete successfully"):
            await collate_sweep_evals([workflow_state], "evaluate")

    async def test_missing_eval_output_raises(self):
        """Test that missing eval output raises ValueError."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"other": "atom-123"},  # No 'eval' key
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        with pytest.raises(ValueError, match="did not produce an 'eval' output atom"):
            await collate_sweep_evals([workflow_state], "evaluate")

    async def test_missing_metric_raises(self):
        """Test that requesting non-existent metric raises ValueError."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={"task1": {"accuracy": 0.9}},
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        with pytest.raises(ValueError, match="Metric 'f1' not found"):
            await collate_sweep_evals([workflow_state], "evaluate", metrics="f1")

    async def test_config_as_dict(self):
        """Test collation when config is a dict instead of dataclass."""
        config = {"lr": 1e-3, "dropout": 0.1, "batch_size": 32}

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={"task1": {"accuracy": 0.9}},
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        df = await collate_sweep_evals([workflow_state], "evaluate")

        assert "lr" in df.columns
        assert "dropout" in df.columns
        assert df["lr"].iloc[0] == 1e-3

    async def test_stats_excluded_from_metrics(self):
        """Test that 'stats' key is excluded from extracted metrics."""
        config = MockConfig(lr=1e-3, dropout=0.1)

        eval_atom_id = await create_mock_eval_atom(
            user="test",
            metrics={
                "task1": {
                    "accuracy": 0.9,
                    "stats": {"runtime": 123.45},  # Should be excluded
                }
            },
        )

        step_state = StepState(
            step_name="evaluate",
            config={},
            status="FINISHED",
            output_atoms={"eval": eval_atom_id},
        )

        workflow_state = WorkflowState(
            workflow_name="test_workflow",
            input_atoms={},
            config=config,
            step_states=[step_state],
        )

        df = await collate_sweep_evals([workflow_state], "evaluate")

        assert "accuracy" in df.columns
        assert "stats" not in df.columns
