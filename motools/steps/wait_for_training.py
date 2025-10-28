"""WaitForTrainingStep - waits for training job completion."""

from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom, TrainingJobAtom
from motools.workflow.base import AtomConstructor
from motools.workflow.training_steps import WaitForTrainingConfig

from .base import BaseStep


class WaitForTrainingStep(BaseStep):
    """Wait for training completion and return ModelAtom.

    This step:
    - Loads training job from TrainingJobAtom
    - Waits for training completion
    - Saves model ID to temp workspace
    - Returns ModelAtom constructor
    """

    name = "wait_for_training"
    input_atom_types = {"job": "training_job"}
    output_atom_types = {"model": "model"}
    config_class: ClassVar[type[Any]] = WaitForTrainingConfig

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute training wait asynchronously.

        Args:
            config: WaitForTrainingConfig instance (currently unused)
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
        model_id = await job_atom.wait()

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
