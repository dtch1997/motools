"""SubmitTrainingStep - submits training jobs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mashumaro import field_options

from motools.atom import Atom, DatasetAtom
from motools.training import get_backend as get_training_backend
from motools.workflow import StepConfig
from motools.workflow.base import AtomConstructor
from motools.workflow.validators import (
    validate_enum,
    validate_model_name,
    validate_non_empty_string,
)


@dataclass
class SubmitTrainingConfig(StepConfig):
    """Config for SubmitTrainingStep.

    Attributes:
        model: Base model to finetune
        hyperparameters: Training hyperparameters (None = backend defaults)
        suffix: Model name suffix
        backend_name: Training backend to use (default: "openai")
    """

    model: str = field(
        metadata=field_options(deserialize=lambda x: validate_model_name(x, "model"))
    )
    hyperparameters: dict[str, Any] | None = None
    suffix: str | None = field(
        default=None,
        metadata=field_options(
            deserialize=lambda x: validate_non_empty_string(x, "suffix") if x is not None else x
        ),
    )
    backend_name: str = field(
        default="openai",
        metadata=field_options(
            deserialize=lambda x: validate_enum(
                x, {"openai", "tinker", "dummy", "openweights"}, "backend_name"
            )
        ),
    )


async def submit_training_step(
    config: SubmitTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
    *,
    input_dataset_name: str = "prepared_dataset",
    output_name: str = "training_job",
) -> list[AtomConstructor]:
    """Submit training job and return TrainingJobAtom immediately.

    This step:
    - Loads dataset from DatasetAtom
    - Gets training backend
    - Submits training job (non-blocking)
    - Saves TrainingRun state
    - Returns TrainingJobAtom constructor

    Args:
        config: SubmitTrainingConfig instance
        input_atoms: Input atoms
        temp_workspace: Temporary workspace for output files
        input_dataset_name: Name of dataset atom in input_atoms (default: "prepared_dataset")
        output_name: Name for output atom constructor (default: "training_job")

    Returns:
        List containing TrainingJobAtom constructor for the submitted job
    """
    # Load dataset atom
    dataset_atom = input_atoms[input_dataset_name]
    assert isinstance(dataset_atom, DatasetAtom)
    dataset = await dataset_atom.to_dataset()
    backend = get_training_backend(config.backend_name)

    # Submit training job (non-blocking)
    training_run = await backend.train(
        dataset=dataset,
        model=config.model,
        hyperparameters=config.hyperparameters,
        suffix=config.suffix,
    )

    # Save TrainingRun state
    training_run_path = temp_workspace / "training_run.json"
    await training_run.save(str(training_run_path))

    # Create TrainingJobAtom constructor
    constructor = AtomConstructor(
        name=output_name,
        path=temp_workspace / "training_run.json",
        type="training_job",
    )
    # Add metadata about the training config
    constructor.metadata = {
        "model": config.model,
        "backend": config.backend_name,
        "hyperparameters": config.hyperparameters,
        "suffix": config.suffix,
    }

    return [constructor]
