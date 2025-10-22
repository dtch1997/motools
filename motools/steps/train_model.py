"""TrainModelStep - trains models on prepared datasets."""

from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom, DatasetAtom
from motools.training import get_backend as get_training_backend
from motools.workflow.base import AtomConstructor
from mozoo.workflows.train_and_evaluate.config import TrainModelConfig

from .base import BaseStep


class TrainModelStep(BaseStep):
    """Train model on prepared dataset.

    This step:
    - Loads dataset from DatasetAtom
    - Gets training backend
    - Starts training and waits for completion
    - Saves training run metadata
    - Returns ModelAtom constructor
    """

    name = "train_model"
    input_atom_types = {"prepared_dataset": "dataset"}
    output_atom_types = {"trained_model": "model"}
    config_class: ClassVar[type[Any]] = TrainModelConfig

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute model training asynchronously.

        Args:
            config: TrainModelConfig instance
            input_atoms: Input atoms (must contain "prepared_dataset")
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing ModelAtom constructor for the trained model
        """
        # Load dataset atom
        dataset_atom = input_atoms["prepared_dataset"]
        assert isinstance(dataset_atom, DatasetAtom)

        # Convert to Dataset
        dataset = await dataset_atom.to_dataset()

        # Get training backend
        backend = get_training_backend(config.backend_name)

        # Start training and wait for completion
        training_run = await backend.train(
            dataset=dataset,
            model=config.model,
            hyperparameters=config.hyperparameters,
            suffix=config.suffix,
        )
        model_id = await training_run.wait()
        await training_run.save(str(temp_workspace / "training_run.json"))

        # Save model_id to temp workspace
        model_id_path = temp_workspace / "model_id.txt"
        model_id_path.write_text(model_id)

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="trained_model",
            path=temp_workspace,
            type="model",
        )
        # Add metadata as an attribute (will be picked up by workflow execution)
        constructor.metadata = {"model_id": model_id}  # type: ignore[attr-defined]

        return [constructor]
