"""Protocol definitions to avoid circular imports.

This module defines abstract interfaces (protocols) that can be used
throughout the codebase without creating circular dependencies.
Concrete implementations should depend on these protocols rather than
on each other.
"""

from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Type variables for generic protocols
T = TypeVar("T")
ConfigT = TypeVar("ConfigT")  # For step configs


# Base protocols for common async patterns
@runtime_checkable
class AsyncWaitable(Protocol, Generic[T]):
    """Protocol for objects that can be waited on asynchronously."""

    async def wait(self) -> T:
        """Wait for completion and return result."""
        ...


@runtime_checkable
class AsyncSavable(Protocol):
    """Protocol for objects that can be saved asynchronously."""

    async def save(self, path: str) -> None:
        """Save to the specified path."""
        ...


@runtime_checkable
class AtomProtocol(Protocol):
    """Protocol for Atom-like objects."""

    id: str
    made_from: dict[str, str]
    metadata: dict[str, Any]

    @property
    def user(self) -> str:
        """Get the user identifier."""
        ...

    def save(self) -> None:
        """Save the atom to storage."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert the atom to a dictionary representation."""
        ...

    @classmethod
    def load(cls, atom_id: str) -> Any:
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

    def get(
        self, workflow_name: str, step_name: str, step_config: Any, input_atoms: dict[str, str]
    ) -> Any | None:
        """Get cached result for a step."""
        ...

    def put(
        self,
        workflow_name: str,
        step_name: str,
        step_config: Any,
        input_atoms: dict[str, str],
        step_state: Any,
    ) -> None:
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
        dataset: "DatasetProtocol",
        model: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> "TrainingRunProtocol":
        """Run a training job."""
        ...

    async def get_model(self, job_id: str) -> str:
        """Get a trained model ID from a job."""
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
        model: str,
        dataset: "DatasetProtocol",
        config: dict[str, Any] | None = None,
    ) -> "EvalJobProtocol":
        """Run an evaluation."""
        ...

    async def get_results(self, eval_id: str) -> "EvalResultsProtocol":
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


@runtime_checkable
class AtomConstructorProtocol(Protocol):
    """Protocol for AtomConstructor objects."""

    name: str
    path: Path
    type: str
    metadata: dict[str, Any] | None


@runtime_checkable
class StepProtocol(Protocol, Generic[ConfigT]):
    """Protocol for workflow steps."""

    name: str
    input_atom_types: dict[str, str]
    output_atom_types: dict[str, str]
    config_class: type[ConfigT]

    async def execute(
        self,
        config: ConfigT,
        input_atoms: dict[str, AtomProtocol],
        temp_workspace: Path,
    ) -> list[AtomConstructorProtocol]:
        """Execute the step."""
        ...


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for dataset objects."""

    def to_jsonl(self, path: Path) -> None:
        """Save dataset to JSONL file."""
        ...

    def __len__(self) -> int:
        """Get dataset length."""
        ...


@runtime_checkable
class EvalResultsProtocol(Protocol):
    """Protocol for evaluation results objects."""

    def summary(self) -> Any:  # Should return DataFrame-like object
        """Get summary of evaluation results."""
        ...

    async def save(self, path: str) -> None:
        """Save evaluation results."""
        ...


@runtime_checkable
class TrainingRunProtocol(AsyncWaitable[str], AsyncSavable, Protocol):
    """Protocol for training run objects.

    Waits for training completion, returns model ID, and can save metadata.
    """

    # Inherits wait() -> str and save(path: str) from base protocols
    pass


@runtime_checkable
class EvalJobProtocol(AsyncWaitable[EvalResultsProtocol], Protocol):
    """Protocol for evaluation job objects.

    Waits for evaluation completion and returns evaluation results.
    """

    # Inherits wait() -> EvalResultsProtocol from AsyncWaitable
    pass
