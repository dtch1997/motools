"""SubmitTrainingStep - submits training jobs."""

from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom, DatasetAtom
from motools.training import get_backend as get_training_backend
from motools.workflow.base import AtomConstructor
from motools.workflow.training_steps import SubmitTrainingConfig

from .base import BaseStep


class SubmitTrainingStep(BaseStep):
    """Submit training job and return TrainingJobAtom immediately.

    This step:
    - Loads dataset from DatasetAtom
    - Gets training backend
    - Submits training job (non-blocking)
    - Saves TrainingRun state
    - Returns TrainingJobAtom constructor
    """

    name = "submit_training"
    input_atom_types = {"prepared_dataset": "dataset"}
    output_atom_types = {"job": "training_job"}
    config_class: ClassVar[type[Any]] = SubmitTrainingConfig

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute training job submission asynchronously.

        Args:
            config: SubmitTrainingConfig instance
            input_atoms: Input atoms (must contain "prepared_dataset")
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
