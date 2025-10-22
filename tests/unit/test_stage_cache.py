"""Unit tests for stage caching."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from motools.cache.stage_cache import StageCache
from motools.workflow.state import StepState


@dataclass
class MockConfig:
    """Mock configuration for testing."""

    param1: str = "value1"
    param2: int = 42


class TestStageCache:
    """Test cases for StageCache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create cache instance with temp directory."""
        return StageCache(cache_dir=temp_cache_dir)

    def test_cache_miss_returns_none(self, cache):
        """Test cache miss returns None."""
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
        )
        assert result is None

    def test_cache_hit_returns_state(self, cache):
        """Test cache hit returns saved state."""
        # Create and save a step state
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
            runtime_seconds=10.5,
        )

        input_atoms = {"input1": "dataset-user-123"}

        # Store in cache
        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Retrieve from cache
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
        )

        assert result is not None
        assert result.output_atoms == {"output1": "model-user-456"}
        assert result.runtime_seconds == 10.5

    def test_cache_key_includes_config(self, cache):
        """Test that different configs produce different cache keys."""
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(param1="value1"),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        input_atoms = {"input1": "dataset-user-123"}

        # Store with one config
        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(param1="value1"),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Try to retrieve with different config
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(param1="value2"),  # Different!
            input_atoms=input_atoms,
        )

        assert result is None  # Should be cache miss

    def test_cache_key_includes_inputs(self, cache):
        """Test that different inputs produce different cache keys."""
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        # Store with one set of inputs
        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
            step_state=step_state,
        )

        # Try to retrieve with different inputs
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-999"},  # Different!
        )

        assert result is None  # Should be cache miss

    def test_failed_state_not_cached(self, cache):
        """Test that failed states are not cached."""
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FAILED",  # Failed!
            error="Something went wrong",
        )

        # Try to store failed state
        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
            step_state=step_state,
        )

        # Try to retrieve
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
        )

        assert result is None  # Should not be cached

    def test_clear_cache(self, cache):
        """Test clearing cache entries."""
        # Add some entries
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        cache.put(
            workflow_name="test_workflow",
            step_name="test_step1",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
            step_state=step_state,
        )

        cache.put(
            workflow_name="test_workflow",
            step_name="test_step2",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-456"},
            step_state=step_state,
        )

        # Clear cache
        count = cache.clear()
        assert count == 2

        # Verify cache is empty
        entries = cache.list_entries()
        assert len(entries) == 0

    def test_list_entries(self, cache):
        """Test listing cache entries."""
        # Add some entries
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        cache.put(
            workflow_name="workflow1",
            step_name="step1",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-123"},
            step_state=step_state,
        )

        cache.put(
            workflow_name="workflow2",
            step_name="step2",
            step_config=MockConfig(),
            input_atoms={"input1": "dataset-user-456"},
            step_state=step_state,
        )

        # List entries
        entries = cache.list_entries()
        assert len(entries) == 2

        # Check entry metadata
        workflow_names = {e["workflow_name"] for e in entries}
        assert workflow_names == {"workflow1", "workflow2"}

        step_names = {e["step_name"] for e in entries}
        assert step_names == {"step1", "step2"}

    def test_cache_metadata_stored(self, cache, temp_cache_dir):
        """Test that metadata is stored alongside cache."""
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        input_atoms = {"input1": "dataset-user-123"}

        # Store in cache
        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Check metadata file exists
        cache_root = Path(temp_cache_dir) / "cache" / "stages"
        metadata_files = list(cache_root.glob("*.json"))
        assert len(metadata_files) == 1

        # Read and verify metadata
        with open(metadata_files[0]) as f:
            metadata = json.load(f)

        assert metadata["workflow_name"] == "test_workflow"
        assert metadata["step_name"] == "test_step"
        assert metadata["input_atoms"] == input_atoms
        assert metadata["output_atoms"] == {"output1": "model-user-456"}
        assert "motools_version" in metadata
        assert "cached_at" in metadata
