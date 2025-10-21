"""PrepareDatasetStep - downloads and prepares datasets."""

import asyncio
import inspect
from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom
from motools.imports import import_function
from motools.workflow.base import AtomConstructor
from mozoo.workflows.train_and_evaluate.config import PrepareDatasetConfig

from .base import BaseStep


class PrepareDatasetStep(BaseStep):
    """Download and prepare dataset using configured loader.

    This step:
    - Imports and calls a dataset loader function
    - Handles async loaders
    - Saves dataset to temp workspace
    - Returns DatasetAtom constructor
    """

    name = "prepare_dataset"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"prepared_dataset": "dataset"}
    config_class: ClassVar[type[Any]] = PrepareDatasetConfig

    def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute dataset preparation.

        Args:
            config: PrepareDatasetConfig instance
            input_atoms: Input atoms (unused for this step)
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing DatasetAtom constructor for the prepared dataset
        """
        del input_atoms  # Unused

        # Import dataset loader function
        loader_fn = import_function(config.dataset_loader)

        # Call loader with kwargs
        kwargs = config.loader_kwargs or {}
        result = loader_fn(**kwargs)

        # Handle async functions
        if inspect.iscoroutine(result):
            dataset = asyncio.run(result)
        else:
            dataset = result

        # Save to temp workspace
        output_path = temp_workspace / "dataset.jsonl"
        asyncio.run(dataset.save(str(output_path)))

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="prepared_dataset",
            path=output_path,
            type="dataset",
        )
        # Add metadata about dataset size
        constructor.metadata = {"samples": len(dataset)}  # type: ignore[attr-defined]

        return [constructor]
