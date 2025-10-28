"""Step functions for training workflows.

This module provides reusable workflow steps for training:
- SubmitTrainingStep: Submit training job and return TrainingJobAtom
- WaitForTrainingStep: Wait for training completion and return ModelAtom
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mashumaro import field_options

from motools.atom import Atom, DatasetAtom, TrainingJobAtom
from motools.training import get_backend as get_training_backend
from motools.workflow import AtomConstructor, StepConfig
from motools.workflow.validators import (
    validate_enum,
    validate_model_name,
    validate_non_empty_string,
)


@dataclass
class SubmitTrainingConfig(StepConfig):
    """Config for submit_training_step.

    Attributes:
        model: Base model to finetune
        hyperparameters: Training hyperparameters (None = backend defaults)
        suffix: Model name suffix
        backend_name: Training backend to use (default: "openai")
        api_timeout: Timeout for individual API calls in seconds (default: 60)
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
            deserialize=lambda x: validate_enum(x, {"openai", "tinker"}, "backend_name")
        ),
    )
    api_timeout: int = field(
        default=60,
        metadata=field_options(
            deserialize=lambda x: x
            if isinstance(x, int) and x > 0
            else (_ for _ in ()).throw(
                ValueError(f"api_timeout must be a positive integer, got {x}")
            )
        ),
    )


@dataclass
class WaitForTrainingConfig(StepConfig):
    """Config for wait_for_training_step.

    Attributes:
        timeout_seconds: Maximum time to wait for training completion in seconds (default: 3600)
        refresh_timeout: Timeout for individual API refresh calls in seconds (default: 30)
    """

    timeout_seconds: int = field(
        default=3600,
        metadata=field_options(
            deserialize=lambda x: x
            if isinstance(x, int) and x > 0
            else (_ for _ in ()).throw(
                ValueError(f"timeout_seconds must be a positive integer, got {x}")
            )
        ),
    )
    refresh_timeout: int = field(
        default=30,
        metadata=field_options(
            deserialize=lambda x: x
            if isinstance(x, int) and x > 0
            else (_ for _ in ()).throw(
                ValueError(f"refresh_timeout must be a positive integer, got {x}")
            )
        ),
    )


async def submit_training_step(
    config: SubmitTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Submit training job and return TrainingJobAtom immediately.

    Args:
        config: Training configuration
        input_atoms: Input atoms (must contain "prepared_dataset" or "dataset")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing TrainingJobAtom constructor for the submitted job
    """
    # Load dataset atom (support both "prepared_dataset" and "dataset" keys for compatibility)
    dataset_atom = input_atoms.get("prepared_dataset") or input_atoms["dataset"]
    assert isinstance(dataset_atom, DatasetAtom)

    # Convert to Dataset
    dataset = await dataset_atom.to_dataset()

    # Get training backend
    backend = get_training_backend(config.backend_name)

    # Submit training job (non-blocking)
    training_run = await backend.train(
        dataset=dataset,
        model=config.model,
        hyperparameters=config.hyperparameters,
        suffix=config.suffix,
        api_timeout=config.api_timeout,
    )
    # Save TrainingRun state
    await training_run.save(str(temp_workspace / "training_run.json"))

    # Create TrainingJobAtom constructor
    constructor = AtomConstructor(
        name="job",
        path=temp_workspace / "training_run.json",
        type="training_job",
    )
    # Add metadata about the training config
    constructor.metadata = {  # type: ignore[attr-defined]
        "model": config.model,
        "backend": config.backend_name,
        "hyperparameters": config.hyperparameters,
        "suffix": config.suffix,
    }

    return [constructor]


async def wait_for_training_step(
    config: WaitForTrainingConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Wait for training completion and return ModelAtom.

    Args:
        config: Wait configuration with timeout settings
        input_atoms: Input atoms (must contain "job")
        temp_workspace: Temporary workspace for output files

    Returns:
        List containing ModelAtom constructor for the trained model
    """
    # Load training job atom
    job_atom = input_atoms["job"]
    assert isinstance(job_atom, TrainingJobAtom)

    # Wait for completion with configured timeouts
    model_id = await job_atom.wait(
        timeout_seconds=config.timeout_seconds,
        refresh_timeout=config.refresh_timeout,
    )

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
