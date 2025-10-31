"""Tests for workflow validation functionality."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.atom import Atom
from motools.workflow.base import AtomConstructor, FunctionStep, Workflow


@dataclass
class DummyConfig:
    """Dummy config for testing."""

    value: str = "test"


@dataclass
class StepConfig:
    """Config for a single step."""

    param: str = "default"


@dataclass
class WorkflowConfig:
    """Config that includes all steps."""

    step1: StepConfig
    step2: StepConfig
    step3: StepConfig


async def dummy_step(
    config: StepConfig, input_atoms: dict[str, Atom], temp_workspace: Path
) -> list[AtomConstructor]:
    """Dummy step function that just passes through."""
    return [
        AtomConstructor(
            name="output",
            path=temp_workspace / "output.txt",
            type="dataset",
        )
    ]


async def transform_step(
    config: StepConfig, input_atoms: dict[str, Atom], temp_workspace: Path
) -> list[AtomConstructor]:
    """Step that transforms input to different type."""
    return [
        AtomConstructor(
            name="transformed",
            path=temp_workspace / "transformed.txt",
            type="model",
        )
    ]


class TestWorkflowValidation:
    """Test workflow validation logic."""

    def test_valid_workflow_passes_validation(self):
        """Test that a valid workflow passes validation."""
        # Create a valid workflow where each step's inputs are satisfied
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={"input_data": "dataset"},
                    output_atom_types={"processed": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
                FunctionStep(
                    name="step2",
                    input_atom_types={"processed": "dataset"},
                    output_atom_types={"output": "model"},
                    config_class=StepConfig,
                    fn=transform_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        # Should not raise any exceptions
        workflow.validate()

    def test_missing_input_detection(self):
        """Test that validation detects missing inputs."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    # This step requires 'missing_input' which is not provided
                    input_atom_types={"missing_input": "dataset"},
                    output_atom_types={"processed": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        with pytest.raises(ValueError) as excinfo:
            workflow.validate()

        assert "Step 'step1' requires input 'missing_input' which is not available" in str(
            excinfo.value
        )

    def test_type_mismatch_detection(self):
        """Test that validation detects type mismatches."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    # This step expects 'model' but workflow provides 'dataset'
                    input_atom_types={"input_data": "model"},
                    output_atom_types={"processed": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        with pytest.raises(ValueError) as excinfo:
            workflow.validate()

        assert "Step 'step1' expects 'input_data' to be type 'model' but got 'dataset'" in str(
            excinfo.value
        )

    def test_multiple_steps_with_chaining(self):
        """Test validation with multiple chained steps."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={"input_data": "dataset"},
                    output_atom_types={"intermediate": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
                FunctionStep(
                    name="step2",
                    input_atom_types={"intermediate": "dataset"},
                    output_atom_types={"transformed": "model"},
                    config_class=StepConfig,
                    fn=transform_step,
                ),
                FunctionStep(
                    name="step3",
                    input_atom_types={
                        "intermediate": "dataset",  # From step1
                        "transformed": "model",  # From step2
                    },
                    output_atom_types={"final": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        # Should not raise any exceptions
        workflow.validate()

    def test_broken_chain_detection(self):
        """Test that validation detects broken chains in multi-step workflows."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={"input_data": "dataset"},
                    output_atom_types={"intermediate": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
                FunctionStep(
                    name="step2",
                    # This step expects 'other_data' which is not produced by step1
                    input_atom_types={"other_data": "dataset"},
                    output_atom_types={"output": "model"},
                    config_class=StepConfig,
                    fn=transform_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        with pytest.raises(ValueError) as excinfo:
            workflow.validate()

        assert "Step 'step2' requires input 'other_data' which is not available" in str(
            excinfo.value
        )

    def test_type_mismatch_in_chain(self):
        """Test that validation detects type mismatches in chains."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={"input_data": "dataset"},
                    output_atom_types={"intermediate": "dataset"},  # Produces dataset
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
                FunctionStep(
                    name="step2",
                    input_atom_types={"intermediate": "model"},  # Expects model!
                    output_atom_types={"output": "model"},
                    config_class=StepConfig,
                    fn=transform_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        with pytest.raises(ValueError) as excinfo:
            workflow.validate()

        assert "Step 'step2' expects 'intermediate' to be type 'model' but got 'dataset'" in str(
            excinfo.value
        )

    def test_multiple_outputs_from_single_step(self):
        """Test validation with steps that produce multiple outputs."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={"input_data": "dataset"},
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={"input_data": "dataset"},
                    output_atom_types={
                        "output1": "dataset",
                        "output2": "model",
                        "output3": "eval",
                    },
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
                FunctionStep(
                    name="step2",
                    input_atom_types={
                        "output1": "dataset",
                        "output2": "model",
                    },
                    output_atom_types={"final": "dataset"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        # Should not raise any exceptions
        workflow.validate()

    def test_workflow_with_multiple_initial_inputs(self):
        """Test validation with multiple workflow inputs."""
        workflow = Workflow(
            name="test_workflow",
            input_atom_types={
                "train_data": "dataset",
                "val_data": "dataset",
                "base_model": "model",
            },
            steps=[
                FunctionStep(
                    name="step1",
                    input_atom_types={
                        "train_data": "dataset",
                        "val_data": "dataset",
                        "base_model": "model",
                    },
                    output_atom_types={"trained_model": "model"},
                    config_class=StepConfig,
                    fn=dummy_step,
                ),
            ],
            config_class=WorkflowConfig,
        )

        # Should not raise any exceptions
        workflow.validate()
