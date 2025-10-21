"""Integration test for two-step training workflow using TrainingJobAtom."""

import json

from motools.atom import Atom, DatasetAtom, ModelAtom, TrainingJobAtom, create_temp_workspace
from motools.workflow import Step, Workflow, WorkflowConfig, run_workflow
from motools.workflow.training_steps import (
    SubmitTrainingConfig,
    WaitForTrainingConfig,
    submit_training_step,
    wait_for_training_step,
)


def test_two_step_training_workflow() -> None:
    """Test submit and wait training steps with TrainingJobAtom.

    This test verifies:
    1. Submit step creates a TrainingJobAtom
    2. Wait step consumes TrainingJobAtom and produces ModelAtom
    3. Full workflow completes successfully with dummy backend
    """

    # Step 1: Create a test dataset
    dataset_data = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 3+3?"},
                {"role": "assistant", "content": "6"},
            ]
        },
    ]

    with create_temp_workspace() as workspace:
        dataset_path = workspace / "test_dataset.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset_data))

        dataset_atom = DatasetAtom.create(
            user="test-user",
            artifact_path=dataset_path,
            made_from={},
            metadata={"samples": len(dataset_data)},
        )

    # Step 2: Define two-step workflow
    workflow = Workflow(
        name="two_step_training",
        input_atom_types={"dataset": "dataset"},
        steps=[
            Step(
                name="submit_training",
                input_atom_types={"dataset": "dataset"},
                output_atom_types={"job": "training_job"},
                config_class=SubmitTrainingConfig,
                fn=submit_training_step,
            ),
            Step(
                name="wait_for_training",
                input_atom_types={"job": "training_job"},
                output_atom_types={"model": "model"},
                config_class=WaitForTrainingConfig,
                fn=wait_for_training_step,
            ),
        ],
        config_class=WorkflowConfig,
    )

    # Step 3: Run workflow with dummy backend
    from dataclasses import dataclass

    @dataclass
    class TwoStepTrainingConfig(WorkflowConfig):
        submit_training: SubmitTrainingConfig
        wait_for_training: WaitForTrainingConfig

    config = TwoStepTrainingConfig(
        submit_training=SubmitTrainingConfig(
            model="gpt-4o-mini-2024-07-18",
            backend_name="dummy",  # Use dummy backend for testing
        ),
        wait_for_training=WaitForTrainingConfig(),
    )

    result = run_workflow(
        workflow=workflow,
        input_atoms={"dataset": dataset_atom.id},
        config=config,
        user="test-user",
    )

    # Step 4: Verify results
    assert len(result.step_states) == 2

    # Verify submit step produced TrainingJobAtom
    submit_state = result.step_states[0]
    assert submit_state.step_name == "submit_training"
    assert "job" in submit_state.output_atoms
    job_atom_id = submit_state.output_atoms["job"]

    job_atom = Atom.load(job_atom_id)
    assert isinstance(job_atom, TrainingJobAtom)
    assert job_atom.type == "training_job"

    # Verify wait step produced ModelAtom
    wait_state = result.step_states[1]
    assert wait_state.step_name == "wait_for_training"
    assert "model" in wait_state.output_atoms
    model_atom_id = wait_state.output_atoms["model"]

    model_atom = Atom.load(model_atom_id)
    assert isinstance(model_atom, ModelAtom)
    assert model_atom.type == "model"
    assert model_atom.get_model_id() == "gpt-4o-mini-2024-07-18"  # Dummy backend returns base model

    # Verify provenance chain
    assert model_atom.made_from["job"] == job_atom_id
    assert job_atom.made_from["dataset"] == dataset_atom.id
