"""PrepareTaskStep - loads and prepares Inspect AI tasks."""

import inspect
import pickle
from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom
from motools.imports import import_function
from motools.workflow.base import AtomConstructor

from .base import BaseStep


class PrepareTaskStep(BaseStep):
    """Load and prepare Inspect AI task using configured loader.

    This step:
    - Imports and calls a task loader function
    - Handles async loaders
    - Serializes task to temp workspace
    - Returns TaskAtom constructor
    """

    name = "prepare_task"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"prepared_task": "task"}
    config_class: ClassVar[type[Any]] = (
        Any  # Will be replaced with PrepareTaskConfig when available
    )

    async def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute task preparation asynchronously.

        Args:
            config: PrepareTaskConfig instance (expects task_loader and optional loader_kwargs)
            input_atoms: Input atoms (unused for this step)
            temp_workspace: Temporary workspace for output files

        Returns:
            List containing TaskAtom constructor for the prepared task
        """
        del input_atoms  # Unused

        # Import task loader function
        loader_fn = import_function(config.task_loader)

        # Call loader with kwargs
        kwargs = config.loader_kwargs or {}
        result = loader_fn(**kwargs)

        # Handle async functions
        if inspect.iscoroutine(result):
            task = await result
        else:
            task = result

        # Serialize task to temp workspace
        output_path = temp_workspace / "task.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(task, f)

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="prepared_task",
            path=output_path,
            type="task",
        )
        # Add metadata about task
        if hasattr(config, "task_loader"):
            constructor.metadata = {"task_loader": config.task_loader}  # type: ignore[attr-defined]

        return [constructor]
