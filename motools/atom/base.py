"""Base classes for Atoms - immutable artifact tracking."""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class Atom:
    """Base class for immutable artifacts with provenance tracking.

    Atoms are created via Atom.create() and loaded via Atom.load().
    They track full lineage (what inputs were used) and metadata.

    Attributes:
        id: Unique identifier (format: {type}-{user}-{suffix})
        type: Type discriminator for polymorphic deserialization
        created_at: Timestamp when atom was created
        made_from: Provenance - maps argument names to atom IDs
        metadata: Arbitrary metadata about this atom
    """

    id: str = field(metadata={"description": "Unique identifier"})
    type: str = field(metadata={"description": "Atom type discriminator"})
    created_at: datetime = field(metadata={"description": "Creation timestamp"})
    made_from: dict[str, str] = field(
        default_factory=dict,
        metadata={"description": "Provenance: arg_name -> atom_id"},
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Arbitrary metadata"},
    )

    @classmethod
    def generate_id(cls, atom_type: str, user: str) -> str:
        """Generate a unique ID for an atom.

        Args:
            atom_type: Type of atom (e.g., "dataset")
            user: User identifier

        Returns:
            Unique ID in format {type}-{user}-{uuid}
        """
        suffix = uuid.uuid4().hex[:8]
        return f"{atom_type}-{user}-{suffix}"

    @classmethod
    async def acreate(  # type: ignore[misc]
        cls,
        atom_type: str,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Atom":
        """Create a new atom from an artifact asynchronously.

        This is the async version of create(). It:
        1. Generates a unique ID
        2. Moves artifact data to storage asynchronously
        3. Saves metadata to index asynchronously

        Args:
            atom_type: Type of atom to create
            user: User identifier
            artifact_path: Path to artifact data
            made_from: Provenance mapping (arg_name -> atom_id)
            metadata: Arbitrary metadata

        Returns:
            Created atom instance
        """
        from motools.atom.storage import amove_artifact_to_storage, asave_atom_metadata

        atom_id = cls.generate_id(atom_type, user)

        atom = cls(
            id=atom_id,
            type=atom_type,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata asynchronously
        await amove_artifact_to_storage(atom_id, artifact_path)
        await asave_atom_metadata(atom)

        return atom

    @classmethod
    def create(  # type: ignore[misc]
        cls,
        atom_type: str,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Atom":
        """Create a new atom from an artifact.

        This is the primary way to create atoms. It:
        1. Generates a unique ID
        2. Moves artifact data to storage
        3. Saves metadata to index

        Args:
            atom_type: Type of atom to create
            user: User identifier
            artifact_path: Path to artifact data
            made_from: Provenance mapping (arg_name -> atom_id)
            metadata: Arbitrary metadata

        Returns:
            Created atom instance
        """
        from motools.atom.storage import move_artifact_to_storage, save_atom_metadata

        atom_id = cls.generate_id(atom_type, user)

        atom = cls(
            id=atom_id,
            type=atom_type,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata
        move_artifact_to_storage(atom_id, artifact_path)
        save_atom_metadata(atom)

        return atom

    @classmethod
    async def aload(cls, atom_id: str) -> "Atom":
        """Load an atom by ID asynchronously.

        Args:
            atom_id: Atom identifier

        Returns:
            Loaded atom instance (correct subclass via type field)

        Raises:
            FileNotFoundError: If atom doesn't exist
        """
        from motools.atom.storage import aload_atom_metadata

        data = await aload_atom_metadata(atom_id)

        # Return correct subclass based on type field
        atom_type = data["type"]

        # Parse datetime from ISO string
        if isinstance(data["created_at"], str):
            from datetime import datetime

            data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Remove 'type' from data for subclasses (has init=False)
        data_copy = {k: v for k, v in data.items() if k != "type"}

        if atom_type == "dataset":
            return DatasetAtom(**data_copy)
        elif atom_type == "model":
            return ModelAtom(**data_copy)
        elif atom_type == "eval":
            return EvalAtom(**data_copy)
        else:
            return cls(**data)

    @classmethod
    def load(cls, atom_id: str) -> "Atom":
        """Load an atom by ID.

        Args:
            atom_id: Atom identifier

        Returns:
            Loaded atom instance (correct subclass via type field)

        Raises:
            FileNotFoundError: If atom doesn't exist
        """
        from motools.atom.storage import get_atom_index_path

        index_path = get_atom_index_path(atom_id)
        if not index_path.exists():
            raise FileNotFoundError(f"Atom not found: {atom_id}")

        with open(index_path) as f:
            data = yaml.safe_load(f)

        # Return correct subclass based on type field
        atom_type = data["type"]

        # Parse datetime from ISO string
        if isinstance(data["created_at"], str):
            from datetime import datetime

            data["created_at"] = datetime.fromisoformat(data["created_at"])

        # Remove 'type' from data for subclasses (has init=False)
        data_copy = {k: v for k, v in data.items() if k != "type"}

        if atom_type == "dataset":
            return DatasetAtom(**data_copy)
        elif atom_type == "model":
            return ModelAtom(**data_copy)
        elif atom_type == "training_job":
            return TrainingJobAtom(**data_copy)
        elif atom_type == "eval":
            return EvalAtom(**data_copy)
        else:
            return cls(**data)

    def get_data_path(self) -> Path:
        """Get path to the atom's data directory.

        Returns:
            Path to data subdirectory
        """
        from motools.atom.storage import get_atom_data_path

        return get_atom_data_path(self.id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize atom to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "made_from": self.made_from,
            "metadata": self.metadata,
        }


@dataclass
class DatasetAtom(Atom):
    """Atom representing a dataset artifact.

    Stores training/eval data with provenance tracking.
    Data is stored in JSONL or other formats in the data directory.
    """

    type: Literal["dataset"] = field(default="dataset", init=False)

    @classmethod
    async def acreate(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DatasetAtom":
        """Create a new dataset atom asynchronously.

        Args:
            user: User identifier
            artifact_path: Path to dataset files
            made_from: Provenance mapping
            metadata: Dataset metadata (e.g., sample count, format)

        Returns:
            Created DatasetAtom
        """
        from motools.atom.storage import amove_artifact_to_storage, asave_atom_metadata

        atom_id = Atom.generate_id("dataset", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata asynchronously
        await amove_artifact_to_storage(atom_id, artifact_path)
        await asave_atom_metadata(atom)

        return atom

    @classmethod
    def create(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DatasetAtom":
        """Create a new dataset atom.

        Args:
            user: User identifier
            artifact_path: Path to dataset files
            made_from: Provenance mapping
            metadata: Dataset metadata (e.g., sample count, format)

        Returns:
            Created DatasetAtom
        """
        from motools.atom.storage import move_artifact_to_storage, save_atom_metadata

        atom_id = Atom.generate_id("dataset", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata
        move_artifact_to_storage(atom_id, artifact_path)
        save_atom_metadata(atom)

        return atom

    @classmethod
    async def from_dataset(
        cls,
        dataset: Any,  # Dataset from motools.datasets
        user: str,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "DatasetAtom":
        """Create a DatasetAtom from a Dataset instance.

        Args:
            dataset: Dataset instance to convert
            user: User identifier
            made_from: Provenance mapping
            metadata: Additional metadata (merged with dataset info)

        Returns:
            Created DatasetAtom

        Example:
            >>> from motools.datasets import JSONLDataset
            >>> dataset = JSONLDataset([{"text": "hello"}])
            >>> atom = await DatasetAtom.from_dataset(dataset, user="alice")
        """
        from motools.atom.workspace import create_temp_workspace

        # Merge metadata
        merged_metadata = {"samples": len(dataset)}
        if metadata:
            merged_metadata |= metadata

        with create_temp_workspace() as temp:
            # Save dataset to temp workspace
            await dataset.save(temp / "data.jsonl")

            # Create atom from saved file
            return cls.create(
                user=user,
                artifact_path=temp / "data.jsonl",
                made_from=made_from,
                metadata=merged_metadata,
            )

    async def to_dataset(self) -> Any:
        """Load a Dataset from this atom.

        Returns:
            Dataset instance loaded from atom data

        Example:
            >>> atom = DatasetAtom.load("dataset-alice-xyz")
            >>> dataset = await atom.to_dataset()
            >>> print(len(dataset))
        """
        from motools.datasets import JSONLDataset

        data_path = self.get_data_path()

        # Find .jsonl file in data directory
        if not (jsonl_files := list(data_path.glob("*.jsonl"))):
            raise ValueError(f"No .jsonl file found in atom data: {data_path}")

        # Load from first .jsonl file
        return await JSONLDataset.load(str(jsonl_files[0]))


@dataclass
class ModelAtom(Atom):
    """Atom representing a trained model.

    Stores the model ID and training metadata with provenance tracking.
    Data directory contains:
    - model_id.txt: The finetuned model identifier
    - training_run.json: TrainingRun metadata (optional)
    """

    type: Literal["model"] = field(default="model", init=False)

    @classmethod
    async def acreate(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ModelAtom":
        """Create a new model atom asynchronously.

        Args:
            user: User identifier
            artifact_path: Path to model files
            made_from: Provenance mapping
            metadata: Model metadata (must include model_id)

        Returns:
            Created ModelAtom
        """
        from motools.atom.storage import amove_artifact_to_storage, asave_atom_metadata

        atom_id = Atom.generate_id("model", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata asynchronously
        await amove_artifact_to_storage(atom_id, artifact_path)
        await asave_atom_metadata(atom)

        return atom

    @classmethod
    def create(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ModelAtom":
        """Create a new model atom.

        Args:
            user: User identifier
            artifact_path: Path to model files
            made_from: Provenance mapping
            metadata: Model metadata (must include model_id)

        Returns:
            Created ModelAtom
        """
        from motools.atom.storage import move_artifact_to_storage, save_atom_metadata

        atom_id = Atom.generate_id("model", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata
        move_artifact_to_storage(atom_id, artifact_path)
        save_atom_metadata(atom)

        return atom

    def get_model_id(self) -> str:
        """Get the finetuned model ID.

        Returns:
            Model identifier string

        Example:
            >>> atom = ModelAtom.load("model-alice-xyz")
            >>> model_id = atom.get_model_id()
            >>> print(model_id)
            ft:gpt-4o-mini-2024-07-18:personal::AbCdEfGh
        """
        return str(self.metadata["model_id"])


@dataclass
class TrainingJobAtom(Atom):
    """Atom representing a training job (in-progress or completed).

    Stores the training job state with provenance tracking.
    Data directory contains:
    - training_run.json: Serialized TrainingRun with job_id, status, etc.
    """

    type: Literal["training_job"] = field(default="training_job", init=False)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TrainingJobAtom":
        """Create a new training job atom.

        Args:
            user: User identifier
            artifact_path: Path to training_run.json file
            made_from: Provenance mapping
            metadata: Training job metadata

        Returns:
            Created TrainingJobAtom
        """
        from motools.atom.storage import move_artifact_to_storage, save_atom_metadata

        atom_id = Atom.generate_id("training_job", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata
        move_artifact_to_storage(atom_id, artifact_path)
        save_atom_metadata(atom)

        return atom

    async def _load_training_run(self) -> Any:
        """Load the TrainingRun from atom data.

        Returns:
            TrainingRun instance

        Raises:
            ValueError: If training_run.json not found
        """
        from motools.training.backends.openai import OpenAITrainingRun

        data_path = self.get_data_path()
        run_file = data_path / "training_run.json"

        if not run_file.exists():
            raise ValueError(f"No training_run.json found in atom data: {data_path}")

        return await OpenAITrainingRun.load(str(run_file))

    async def refresh(self) -> None:
        """Update status from backend (explicit, not auto-polling on load)."""
        run = await self._load_training_run()
        await run.refresh()
        # Save updated state back to disk
        data_path = self.get_data_path()
        await run.save(str(data_path / "training_run.json"))

    async def get_status(self) -> str:
        """Get current job status.

        Returns:
            Status string: "queued" | "running" | "succeeded" | "failed" | "cancelled"
        """
        run = await self._load_training_run()
        status: str = await run.get_status()
        return status

    async def wait(self) -> str:
        """Wait for completion and return model_id.

        Returns:
            The finetuned model ID

        Raises:
            RuntimeError: If training fails
        """
        run = await self._load_training_run()
        model_id: str = await run.wait()
        # Save updated state back to disk
        data_path = self.get_data_path()
        await run.save(str(data_path / "training_run.json"))
        return model_id


@dataclass
class EvalAtom(Atom):
    """Atom representing evaluation results.

    Stores evaluation metrics and results with provenance tracking.
    Data directory contains:
    - results.json: EvalResults file with samples and metrics
    - log_paths.txt: Paths to evaluation logs (optional)
    """

    type: Literal["eval"] = field(default="eval", init=False)

    @classmethod
    async def acreate(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvalAtom":
        """Create a new eval atom asynchronously.

        Args:
            user: User identifier
            artifact_path: Path to eval results files
            made_from: Provenance mapping
            metadata: Eval metadata (e.g., score, samples)

        Returns:
            Created EvalAtom
        """
        from motools.atom.storage import amove_artifact_to_storage, asave_atom_metadata

        atom_id = Atom.generate_id("eval", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata asynchronously
        await amove_artifact_to_storage(atom_id, artifact_path)
        await asave_atom_metadata(atom)

        return atom

    @classmethod
    def create(  # type: ignore[override]
        cls,
        user: str,
        artifact_path: Path,
        made_from: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EvalAtom":
        """Create a new eval atom.

        Args:
            user: User identifier
            artifact_path: Path to eval results files
            made_from: Provenance mapping
            metadata: Eval metadata (e.g., score, samples)

        Returns:
            Created EvalAtom
        """
        from motools.atom.storage import move_artifact_to_storage, save_atom_metadata

        atom_id = Atom.generate_id("eval", user)

        atom = cls(
            id=atom_id,
            created_at=datetime.now(UTC),
            made_from=made_from or {},
            metadata=metadata or {},
        )

        # Move data and save metadata
        move_artifact_to_storage(atom_id, artifact_path)
        save_atom_metadata(atom)

        return atom

    async def to_eval_results(self) -> Any:
        """Load EvalResults from this atom.

        Returns:
            EvalResults instance

        Example:
            >>> atom = EvalAtom.load("eval-alice-xyz")
            >>> results = await atom.to_eval_results()
            >>> print(results.metrics)
        """
        # Import here to avoid circular dependency
        from motools.evals.backends.inspect import InspectEvalResults

        data_path = self.get_data_path()
        results_file = data_path / "results.json"

        if not results_file.exists():
            raise ValueError(f"No results.json found in atom data: {data_path}")

        return await InspectEvalResults.load(str(results_file))
