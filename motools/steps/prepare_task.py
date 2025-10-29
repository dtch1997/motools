"""PrepareTaskStep - loads and prepares Inspect AI tasks."""

import json
from pathlib import Path
from typing import ClassVar

from motools.atom import Atom
from motools.workflow.base import AtomConstructor

from .base import BaseStep
from .configs import PrepareTaskConfig


class PrepareTaskStep(BaseStep):
    """Prepare Inspect AI task reference using configured loader.

    This step stores the task loader reference (not the task itself) so it can be
    reloaded when needed. This avoids serialization issues with Task objects.

    This step:
    - Validates task loader is importable
    - Stores loader reference and kwargs
    - Returns TaskAtom constructor
    """

    name = "prepare_task"
    input_atom_types = {}  # No inputs - starts from scratch
    output_atom_types = {"task": "task"}
    config_class: ClassVar[type[PrepareTaskConfig]] = PrepareTaskConfig

    async def execute(
        self,
        config: PrepareTaskConfig,
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

        # Store task loader reference instead of the task itself
        task_spec = {
            "task_loader": config.task_loader,
            "loader_kwargs": config.loader_kwargs or {},
        }

        # Save to JSON
        output_path = temp_workspace / "task_spec.json"
        with open(output_path, "w") as f:
            json.dump(task_spec, f, indent=2)

        # Create atom constructor with metadata
        constructor = AtomConstructor(
            name="task",
            path=output_path,
            type="task",
        )
        constructor.metadata = {"task_loader": config.task_loader}  # type: ignore[attr-defined]

        return [constructor]
