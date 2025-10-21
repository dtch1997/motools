"""Base class for workflow steps."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from motools.atom import Atom
from motools.workflow.base import AtomConstructor, Step


class BaseStep(ABC):
    """Abstract base class for workflow steps.

    Subclasses should:
    - Define class-level metadata (name, input_atom_types, output_atom_types, config_class)
    - Implement the execute() method with step logic

    Example:
        class PrepareDatasetStep(BaseStep):
            name = "prepare_dataset"
            input_atom_types = {}
            output_atom_types = {"prepared_dataset": "dataset"}
            config_class = PrepareDatasetConfig

            def execute(self, config, input_atoms, temp_workspace):
                # Step implementation
                ...
    """

    # Class-level metadata (to be overridden by subclasses)
    name: ClassVar[str]
    input_atom_types: ClassVar[dict[str, str]]
    output_atom_types: ClassVar[dict[str, str]]
    config_class: ClassVar[type[Any]]

    @abstractmethod
    def execute(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute the step logic.

        Args:
            config: Step configuration instance
            input_atoms: Loaded input atoms
            temp_workspace: Temporary workspace for outputs

        Returns:
            List of atom constructors for outputs
        """
        pass

    def __call__(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Make instance callable for use as Step.fn.

        This allows step instances to be used as the fn parameter
        in Step dataclass, maintaining compatibility with existing
        workflow execution engine.

        Args:
            config: Step configuration
            input_atoms: Loaded input atoms
            temp_workspace: Temporary workspace for outputs

        Returns:
            List of atom constructors for outputs
        """
        return self.execute(config, input_atoms, temp_workspace)

    @classmethod
    def as_step(cls) -> Step:
        """Create a Step wrapper instance from this class.

        Returns:
            Step instance configured with class metadata and callable instance

        Example:
            step = PrepareDatasetStep.as_step()
            # Can be used in workflow.steps list
        """
        return Step(
            name=cls.name,
            input_atom_types=cls.input_atom_types,
            output_atom_types=cls.output_atom_types,
            config_class=cls.config_class,
            fn=cls(),  # Instantiate and use as callable
        )
