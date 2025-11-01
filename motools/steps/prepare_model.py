"""PrepareModelStep - prepares off-the-shelf models for evaluation."""

from pathlib import Path
from typing import ClassVar

from motools.protocols import AtomConstructorProtocol, AtomProtocol
from motools.workflow.base import AtomConstructor

from .base import BaseStep
from .configs import PrepareModelConfig


class PrepareModelStep(BaseStep):
    """Prepare an off-the-shelf model for evaluation.

    This step:
    - Takes a model ID from config
    - Saves model ID to temp workspace
    - Returns ModelAtom constructor with metadata

    This is useful for evaluating pre-trained models without fine-tuning.
    """

    name = "prepare_model"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"model": "model"}
    config_class: ClassVar[type[PrepareModelConfig]] = PrepareModelConfig

    async def execute(
        self,
        config: PrepareModelConfig,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute model preparation asynchronously.

        Args:
            config: PrepareModelConfig instance
            input_atoms: Input atoms (unused for this step)
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing ModelAtom constructor for the prepared model
        """
        del input_atoms  # Unused

        # Save model_id to temp workspace
        model_id_path = temp_workspace / "model_id.txt"
        model_id_path.write_text(config.model_id)

        # Create ModelAtom constructor
        constructor = AtomConstructor(
            name="model",
            path=temp_workspace,
            type="model",
        )
        # Add metadata with model_id
        constructor.metadata = {"model_id": config.model_id}

        return [constructor]
