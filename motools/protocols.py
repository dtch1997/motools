"""Protocol definitions to avoid circular imports.

This module defines abstract interfaces (protocols) that can be used
throughout the codebase without creating circular dependencies.
Concrete implementations should depend on these protocols rather than
on each other.
"""

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AtomProtocol(Protocol):
    """Protocol for Atom-like objects."""

    id: str
    user: str
    made_from: dict[str, str]
    metadata: dict[str, Any]

    def save(self) -> None:
        """Save the atom to storage."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert the atom to a dictionary representation."""
        ...

    @classmethod
    def load(cls, atom_id: str) -> "AtomProtocol":
        """Load an atom from storage."""
        ...


@runtime_checkable
class StorageProtocol(Protocol):
    """Protocol for storage backends."""

    async def save_atom(self, atom: AtomProtocol) -> None:
        """Save an atom to storage."""
        ...

    async def load_atom(self, atom_id: str) -> AtomProtocol:
        """Load an atom from storage."""
        ...

    async def find_atom_by_hash(self, content_hash: str) -> str | None:
        """Find an atom by its content hash."""
        ...

    async def move_artifact_to_storage(self, artifact_path: Path, storage_path: Path) -> None:
        """Move an artifact to storage."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache backends."""

    def get(self, workflow_name: str, step_name: str,
            step_config: Any, input_atoms: dict[str, str]) -> Any | None:
        """Get cached result for a step."""
        ...

    def put(self, workflow_name: str, step_name: str,
            step_config: Any, input_atoms: dict[str, str],
            step_state: Any) -> None:
        """Cache the result of a step."""
        ...

    def clear(self) -> int:
        """Clear all cache entries."""
        ...


@runtime_checkable
class TrainingBackendProtocol(Protocol):
    """Protocol for training backends."""

    async def initialize(self) -> None:
        """Initialize the backend."""
        ...

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        ...

    async def run_training(
        self,
        dataset: Any,
        model: Any | None = None,
        config: Any | None = None,
    ) -> Any:
        """Run a training job."""
        ...

    async def get_model(self, job_id: str) -> Any:
        """Get a trained model from a job."""
        ...


@runtime_checkable
class EvalBackendProtocol(Protocol):
    """Protocol for evaluation backends."""

    async def initialize(self) -> None:
        """Initialize the backend."""
        ...

    async def cleanup(self) -> None:
        """Clean up backend resources."""
        ...

    async def run_eval(
        self,
        model: Any,
        dataset: Any,
        config: Any | None = None,
    ) -> Any:
        """Run an evaluation."""
        ...

    async def get_results(self, eval_id: str) -> Any:
        """Get evaluation results."""
        ...


@runtime_checkable
class ClientProtocol(Protocol):
    """Protocol for MOTools client."""

    @property
    def training_backend(self) -> TrainingBackendProtocol:
        """Get the training backend."""
        ...

    @property
    def eval_backend(self) -> EvalBackendProtocol:
        """Get the evaluation backend."""
        ...

    async def close(self) -> None:
        """Close the client and clean up resources."""
        ...


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Protocol for workflow objects."""

    name: str
    steps: list[Any]
    steps_by_name: dict[str, Any]
    input_atom_types: dict[str, str]

    def validate(self) -> None:
        """Validate the workflow configuration."""
        ...


@runtime_checkable
class StepStateProtocol(Protocol):
    """Protocol for step state objects."""

    step_name: str
    config: Any
    input_atoms: dict[str, str]
    output_atoms: dict[str, str]
    runtime_seconds: float | None
    status: str
    error: str | None

