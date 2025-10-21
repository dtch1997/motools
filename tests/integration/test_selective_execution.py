"""Integration tests for selective stage execution and caching."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.workflow import (
    AtomConstructor,
    Step,
    StepConfig,
    Workflow,
    WorkflowConfig,
    run_workflow,
)

# ============ Step Configs ============


@dataclass
class PrepareConfig(StepConfig):
    """Config for prepare step."""

    dataset_size: int = 100


@dataclass
class TrainConfig(StepConfig):
    """Config for train step."""

    epochs: int = 10


@dataclass
class EvaluateConfig(StepConfig):
    """Config for evaluate step."""

    batch_size: int = 32


# ============ Step Functions ============


def prepare_fn(
    config: PrepareConfig,
    input_atoms: dict[str, any],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Prepare dataset step."""
    # Create dummy dataset
    data = [{"id": i, "value": i * 2} for i in range(config.dataset_size)]

    # Write output
    (temp_workspace / "dataset.jsonl").write_text("\n".join(json.dumps(item) for item in data))

    return [
        AtomConstructor(
            name="prepared_dataset",
            path=temp_workspace / "dataset.jsonl",
            type="dataset",
        )
    ]


def train_fn(
    config: TrainConfig,
    input_atoms: dict[str, any],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Train model step."""
    dataset = input_atoms["prepared_dataset"]

    # Read dataset
    data_path = dataset.get_data_path()
    data = [json.loads(line) for line in (data_path / "dataset.jsonl").read_text().splitlines()]

    # Create dummy model
    model_info = {
        "type": "dummy",
        "epochs": config.epochs,
        "dataset_size": len(data),
    }

    (temp_workspace / "model.json").write_text(json.dumps(model_info))

    return [
        AtomConstructor(
            name="trained_model",
            path=temp_workspace / "model.json",
            type="model",
        )
    ]


def evaluate_fn(
    config: EvaluateConfig,
    input_atoms: dict[str, any],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Evaluate model step."""
    input_atoms["trained_model"]
    input_atoms["prepared_dataset"]

    # Create dummy evaluation results
    eval_results = {
        "accuracy": 0.95,
        "batch_size": config.batch_size,
    }

    (temp_workspace / "eval.json").write_text(json.dumps(eval_results))

    return [
        AtomConstructor(
            name="evaluation",
            path=temp_workspace / "eval.json",
            type="eval",
        )
    ]


# ============ Workflow Definition ============


@dataclass
class TestWorkflowConfig(WorkflowConfig):
    """Config for test workflow."""

    prepare: PrepareConfig
    train: TrainConfig
    evaluate: EvaluateConfig


test_workflow = Workflow(
    name="test_selective",
    input_atom_types={},
    steps=[
        Step(
            name="prepare",
            input_atom_types={},
            output_atom_types={"prepared_dataset": "dataset"},
            config_class=PrepareConfig,
            fn=prepare_fn,
        ),
        Step(
            name="train",
            input_atom_types={"prepared_dataset": "dataset"},
            output_atom_types={"trained_model": "model"},
            config_class=TrainConfig,
            fn=train_fn,
        ),
        Step(
            name="evaluate",
            input_atom_types={
                "trained_model": "model",
                "prepared_dataset": "dataset",
            },
            output_atom_types={"evaluation": "eval"},
            config_class=EvaluateConfig,
            fn=evaluate_fn,
        ),
    ],
    config_class=TestWorkflowConfig,
)


# ============ Tests ============


class TestSelectiveExecution:
    """Test selective stage execution."""

    @pytest.fixture
    def config(self):
        """Create workflow config."""
        return TestWorkflowConfig(
            prepare=PrepareConfig(dataset_size=50),
            train=TrainConfig(epochs=5),
            evaluate=EvaluateConfig(batch_size=16),
        )

    def test_run_all_stages(self, config):
        """Test running all stages."""
        result = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
        )

        assert len(result.step_states) == 3
        assert all(s.status == "FINISHED" for s in result.step_states)

        # Check outputs exist
        assert "prepared_dataset" in result.step_states[0].output_atoms
        assert "trained_model" in result.step_states[1].output_atoms
        assert "evaluation" in result.step_states[2].output_atoms

    def test_run_selected_stages(self, config):
        """Test running only selected stages."""
        # First run prepare and train
        result1 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare", "train"],
        )

        # Check only selected stages ran
        assert result1.step_states[0].status == "FINISHED"  # prepare
        assert result1.step_states[1].status == "FINISHED"  # train
        assert result1.step_states[2].status == "PENDING"  # evaluate (not run)

    def test_run_range_selection(self, config):
        """Test running a range of stages."""
        # This should fail because train requires prepared_dataset
        # which is not available without running prepare
        with pytest.raises(ValueError, match="requires input 'prepared_dataset'"):
            run_workflow(
                workflow=test_workflow,
                input_atoms={},
                config=config,
                user="test",
                selected_stages=["train", "evaluate"],  # Will fail without prepare
                no_cache=True,  # Disable cache to ensure failure
            )

    def test_cache_hit(self, config):
        """Test that cache hits work correctly."""
        with tempfile.TemporaryDirectory():
            # First run
            result1 = run_workflow(
                workflow=test_workflow,
                input_atoms={},
                config=config,
                user="test",
                selected_stages=["prepare"],
            )

            prepare_output = result1.step_states[0].output_atoms["prepared_dataset"]

            # Second run with same config should hit cache
            result2 = run_workflow(
                workflow=test_workflow,
                input_atoms={},
                config=config,
                user="test",
                selected_stages=["prepare"],
            )

            # Should get same output atom ID from cache
            assert result2.step_states[0].output_atoms["prepared_dataset"] == prepare_output

    def test_force_rerun(self, config):
        """Test force rerun bypasses cache."""
        # First run
        result1 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare"],
        )

        prepare_output1 = result1.step_states[0].output_atoms["prepared_dataset"]

        # Second run with force_rerun
        result2 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare"],
            force_rerun=True,
        )

        prepare_output2 = result2.step_states[0].output_atoms["prepared_dataset"]

        # Should get different output atom IDs (new execution)
        assert prepare_output2 != prepare_output1

    def test_no_cache(self, config):
        """Test no_cache prevents caching."""
        # Run with no_cache
        result1 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare"],
            no_cache=True,
        )

        prepare_output1 = result1.step_states[0].output_atoms["prepared_dataset"]

        # Second run should not hit cache
        result2 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare"],
        )

        prepare_output2 = result2.step_states[0].output_atoms["prepared_dataset"]

        # Should get different outputs (no cache hit)
        assert prepare_output2 != prepare_output1

    def test_partial_workflow_with_cache(self, config):
        """Test running partial workflow with cached dependencies."""
        # First run prepare to populate cache
        run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["prepare"],
        )

        # Now run train and evaluate - prepare should be loaded from cache
        result2 = run_workflow(
            workflow=test_workflow,
            input_atoms={},
            config=config,
            user="test",
            selected_stages=["train", "evaluate"],
        )

        # All stages should have completed
        assert result2.step_states[0].status == "FINISHED"  # prepare from cache
        assert result2.step_states[1].status == "FINISHED"  # train
        assert result2.step_states[2].status == "FINISHED"  # evaluate
