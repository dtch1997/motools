"""Base classes for workflow system."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from motools.atom import Atom


@dataclass
class AtomConstructor:
    """Request to create an atom from a step's output.

    Similar to LineageNodeConstructor in pollux.
    Steps return these to specify what atoms should be created.
    """

    name: str = field(metadata={"description": "Output argument name for chaining"})
    path: Path = field(metadata={"description": "Path to artifact data"})
    type: str = field(metadata={"description": "Atom type (e.g., 'dataset')"})
    tags: list[str] = field(
        default_factory=list,
        metadata={"description": "Tags to attach to created atom"},
    )


@dataclass
class Step:
    """A single step in a workflow with typed inputs and outputs.

    Steps are pure functions that:
    - Read input atoms from disk
    - Read their config
    - Write outputs to temp workspace
    - Return atom constructors for outputs
    """

    name: str = field(metadata={"description": "Step name"})
    input_atom_types: dict[str, str] = field(
        metadata={"description": "Required input atoms: arg_name -> atom_type"}
    )
    output_atom_types: dict[str, str] = field(
        metadata={"description": "Expected output atoms: arg_name -> atom_type"}
    )
    config_class: type[Any] = field(metadata={"description": "Configuration class for this step"})
    fn: Callable[[Any, dict[str, Atom], Path], list[AtomConstructor]] = field(
        metadata={"description": "Step function"}
    )

    def __call__(
        self,
        config: Any,
        input_atoms: dict[str, Atom],
        temp_workspace: Path,
    ) -> list[AtomConstructor]:
        """Execute the step function.

        Args:
            config: Step configuration
            input_atoms: Loaded input atoms
            temp_workspace: Temporary workspace for outputs

        Returns:
            List of atom constructors for outputs

        Raises:
            Exception: If step function fails
        """
        # Validate inputs
        self._validate_inputs(input_atoms)

        # Execute
        try:
            return self.fn(config, input_atoms, temp_workspace)
        except Exception as e:
            raise RuntimeError(f"Step '{self.name}' failed: {e}") from e

    def _validate_inputs(self, input_atoms: dict[str, Atom]) -> None:
        """Validate that all required inputs are present with correct types.

        Args:
            input_atoms: Loaded input atoms

        Raises:
            ValueError: If inputs are invalid
        """
        # Check all required inputs present
        missing = set(self.input_atom_types.keys()) - set(input_atoms.keys())
        if missing:
            raise ValueError(f"Step '{self.name}' missing required inputs: {missing}")

        # Check input types match
        for arg_name, expected_type in self.input_atom_types.items():
            actual_type = input_atoms[arg_name].type
            if actual_type != expected_type:
                raise ValueError(
                    f"Step '{self.name}' input '{arg_name}' has type '{actual_type}' "
                    f"but expected '{expected_type}'"
                )

    def validate_outputs(self, atom_constructors: list[AtomConstructor]) -> list[str]:
        """Validate that all expected outputs are present.

        Args:
            atom_constructors: Constructors returned by step

        Returns:
            List of missing output names (empty if all present)
        """
        output_names = {c.name for c in atom_constructors}
        missing = set(self.output_atom_types.keys()) - output_names
        return list(missing)


@dataclass
class Workflow:
    """A collection of steps forming a DAG.

    Workflows define:
    - Required input atoms
    - Sequence of steps to execute
    - Configuration schema
    """

    name: str = field(metadata={"description": "Workflow name"})
    input_atom_types: dict[str, str] = field(
        metadata={"description": "Required input atoms: arg_name -> atom_type"}
    )
    steps: list[Step] = field(metadata={"description": "Steps to execute in order"})
    config_class: type[Any] = field(
        metadata={"description": "Configuration class for this workflow"}
    )

    def __post_init__(self) -> None:
        """Build steps lookup dict."""
        self.steps_by_name: dict[str, Step] = {step.name: step for step in self.steps}

        # Validate uniqueness
        if len(self.steps_by_name) != len(self.steps):
            raise ValueError(f"Workflow '{self.name}' has duplicate step names")
