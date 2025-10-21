"""Step functions for training workflows.

This module provides reusable workflow steps for training:
- SubmitTrainingStep: Submit training job and return TrainingJobAtom
- WaitForTrainingStep: Wait for training completion and return ModelAtom
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motools.atom import Atom, DatasetAtom, TrainingJobAtom
from motools.training import get_backend as get_training_backend
from motools.workflow import AtomConstructor, StepConfig


@dataclass
class SubmitTrainingConfig(StepConfig):
    """Config for submit_training_step.

    Attributes:
        model: Base model to finetune
        hyperparameters: Training hyperparameters (None = backend defaults)
        suffix: Model name suffix
        backend_name: Training backend to use (default: "openai")
    """

    model: str
    hyperparameters: dict[str, Any] | None = None
    suffix: str | None = None
    backend_name: str = "openai"


@dataclass
class WaitForTrainingConfig(StepConfig):
    """Config for wait_for_training_step.

    Currently has no config options, but included for consistency.
    """

    pass


def submit_training_step(
    config: SubmitTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Submit training job and return TrainingJobAtom immediately.

    Args:
        config: Training configuration
        input_atoms: Input atoms (must contain "dataset")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing TrainingJobAtom constructor for the submitted job
    """
    # Load dataset atom
    dataset_atom = input_atoms["dataset"]
    assert isinstance(dataset_atom, DatasetAtom)

    # Convert to Dataset
    dataset = asyncio.run(dataset_atom.to_dataset())

    # Get training backend
    backend = get_training_backend(config.backend_name)

    # Submit training job (non-blocking)
    async def submit_training():
        training_run = await backend.train(
            dataset=dataset,
            model=config.model,
            hyperparameters=config.hyperparameters,
            suffix=config.suffix,
        )
        # Save TrainingRun state
        await training_run.save(str(temp_workspace / "training_run.json"))
        return training_run

    asyncio.run(submit_training())

    # Create TrainingJobAtom constructor
    constructor = AtomConstructor(
        name="job",
        path=temp_workspace / "training_run.json",
        type="training_job",
    )

    return [constructor]


def wait_for_training_step(
    config: WaitForTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Wait for training completion and return ModelAtom.

    Args:
        config: Wait configuration (currently unused)
        input_atoms: Input atoms (must contain "job")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing ModelAtom constructor for the trained model
    """
    del config  # Unused

    # Load training job atom
    job_atom = input_atoms["job"]
    assert isinstance(job_atom, TrainingJobAtom)

    # Wait for completion
    async def wait_for_completion():
        model_id = await job_atom.wait()
        return model_id

    model_id = asyncio.run(wait_for_completion())

    # Save model_id to temp workspace
    model_id_path = temp_workspace / "model_id.txt"
    model_id_path.write_text(model_id)

    # Create ModelAtom constructor
    constructor = AtomConstructor(
        name="model",
        path=temp_workspace,
        type="model",
    )
    # Add metadata with model_id
    constructor.metadata = {"model_id": model_id}  # type: ignore[attr-defined]

    return [constructor]
