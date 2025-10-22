"""Content-addressable caching for workflow stages."""

import hashlib
import importlib.metadata
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from motools.workflow.state import StepState


class StageCache:
    """Cache for individual workflow stage outputs."""

    def __init__(self, cache_dir: str = ".motools"):
        """Initialize stage cache.

        Args:
            cache_dir: Root directory for cache storage
        """
        self.cache_root = Path(cache_dir) / "cache" / "stages"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Get motools version
        try:
            self.motools_version = importlib.metadata.version("motools")
        except importlib.metadata.PackageNotFoundError:
            self.motools_version = "dev"

    def _generate_cache_key(
        self, workflow_name: str, step_name: str, step_config: Any, input_atoms: dict[str, str]
    ) -> str:
        """Generate content-addressable cache key for a stage.

        Args:
            workflow_name: Name of the workflow
            step_name: Name of the step
            step_config: Configuration for the step
            input_atoms: Input atom IDs

        Returns:
            Hex string cache key
        """
        # Create deterministic representation
        cache_data = {
            "workflow": workflow_name,
            "step": step_name,
            "config": self._serialize_config(step_config),
            "inputs": input_atoms,
            "version": self.motools_version,
        }

        # Hash the data
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _serialize_config(self, config: Any) -> Any:
        """Serialize config object to JSON-compatible format.

        Args:
            config: Config object (potentially a dataclass)

        Returns:
            JSON-serializable representation
        """
        if config is None:
            return None

        # Handle dataclasses
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import asdict

            return asdict(config)

        # Handle primitives and dicts
        if isinstance(config, (str, int, float, bool, list, dict)):
            return config

        # Fallback to string representation
        return str(config)

    def get(
        self, workflow_name: str, step_name: str, step_config: Any, input_atoms: dict[str, str]
    ) -> StepState | None:
        """Retrieve cached stage output if available.

        Args:
            workflow_name: Name of the workflow
            step_name: Name of the step
            step_config: Configuration for the step
            input_atoms: Input atom IDs

        Returns:
            Cached StepState if found, None otherwise
        """
        cache_key = self._generate_cache_key(workflow_name, step_name, step_config, input_atoms)
        cache_file = self.cache_root / f"{cache_key}.pkl"
        metadata_file = self.cache_root / f"{cache_key}.json"

        if not cache_file.exists() or not metadata_file.exists():
            return None

        # Check version compatibility
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)

            cached_version = metadata.get("motools_version", "unknown")
            if cached_version != self.motools_version:
                print(
                    f"⚠️  Cache version mismatch for {step_name}: "
                    f"cached={cached_version}, current={self.motools_version}"
                )
                # Still return the cached result but with warning

            # Load cached state
            with open(cache_file, "rb") as f:
                step_state: StepState = pickle.load(f)

            print(f"✓ Cache hit for stage '{step_name}' (key: {cache_key[:8]}...)")
            return step_state

        except Exception as e:
            print(f"⚠️  Failed to load cache for {step_name}: {e}")
            return None

    def put(
        self,
        workflow_name: str,
        step_name: str,
        step_config: Any,
        input_atoms: dict[str, str],
        step_state: StepState,
    ) -> None:
        """Store stage output in cache.

        Args:
            workflow_name: Name of the workflow
            step_name: Name of the step
            step_config: Configuration for the step
            input_atoms: Input atom IDs
            step_state: Completed step state to cache
        """
        if step_state.status != "FINISHED":
            # Don't cache failed or incomplete stages
            return

        cache_key = self._generate_cache_key(workflow_name, step_name, step_config, input_atoms)
        cache_file = self.cache_root / f"{cache_key}.pkl"
        metadata_file = self.cache_root / f"{cache_key}.json"

        try:
            # Save state
            with open(cache_file, "wb") as f:
                pickle.dump(step_state, f)

            # Save metadata
            metadata = {
                "workflow_name": workflow_name,
                "step_name": step_name,
                "motools_version": self.motools_version,
                "cached_at": datetime.now(UTC).isoformat(),
                "input_atoms": input_atoms,
                "output_atoms": step_state.output_atoms,
            }
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Cached stage '{step_name}' (key: {cache_key[:8]}...)")

        except Exception as e:
            print(f"⚠️  Failed to cache {step_name}: {e}")
            # Remove partial files if they exist
            cache_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)

    def clear(self) -> int:
        """Clear all stage cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0
        for file in self.cache_root.glob("*.pkl"):
            file.unlink()
            count += 1
        for file in self.cache_root.glob("*.json"):
            file.unlink()
        return count

    def list_entries(self) -> list[dict]:
        """List all cached stage entries.

        Returns:
            List of cache entry metadata
        """
        entries = []
        for metadata_file in self.cache_root.glob("*.json"):
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    metadata["cache_key"] = metadata_file.stem
                    entries.append(metadata)
            except Exception:
                continue
        return entries
