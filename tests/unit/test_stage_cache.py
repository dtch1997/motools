"""Unit tests for stage caching."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from loguru import logger

from motools.cache.stage_cache import CachePolicy, StageCache
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
            step_config=MockConfig(param1="different_value"),
            input_atoms=input_atoms,
        )

        # Should return None because config is different
        assert result is None

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
            input_atoms={"input1": "dataset-user-456"},
        )

        # Should return None because inputs are different
        assert result is None

    def test_failed_state_not_cached(self, cache):
        """Test that failed states are not cached."""
        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FAILED",
            output_atoms={},
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

        # Should return None because failed states aren't cached
        assert result is None

    def test_clear_cache(self, cache):
        """Test clearing all cache entries."""
        # Add multiple entries
        for i in range(3):
            step_state = StepState(
                step_name=f"test_step{i}",
                config=MockConfig(),
                status="FINISHED",
                output_atoms={"output1": f"model-user-{i}"},
            )
            cache.put(
                workflow_name="test_workflow",
                step_name=f"test_step{i}",
                step_config=MockConfig(),
                input_atoms={"input1": f"dataset-user-{i}"},
                step_state=step_state,
            )

        # Clear cache
        count = cache.clear()
        assert count == 3

        # Check entries are gone
        for i in range(3):
            result = cache.get(
                workflow_name="test_workflow",
                step_name=f"test_step{i}",
                step_config=MockConfig(),
                input_atoms={"input1": f"dataset-user-{i}"},
            )
            assert result is None

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

    def test_cache_logging(self, cache):
        """Test that cache operations produce appropriate log messages."""
        import io

        # Create a string buffer to capture logs
        log_buffer = io.StringIO()

        # Add handler to capture logs
        handler_id = logger.add(log_buffer, format="{message}", level="INFO")

        try:
            step_state = StepState(
                step_name="test_step",
                config=MockConfig(),
                status="FINISHED",
                output_atoms={"output1": "model-user-456"},
            )

            input_atoms = {"input1": "dataset-user-123"}

            # Store in cache (should log)
            cache.put(
                workflow_name="test_workflow",
                step_name="test_step",
                step_config=MockConfig(),
                input_atoms=input_atoms,
                step_state=step_state,
            )

            # Check that cache put was logged
            log_output = log_buffer.getvalue()
            assert "Cached stage 'test_step'" in log_output

            # Clear buffer
            log_buffer.truncate(0)
            log_buffer.seek(0)

            # Retrieve from cache (should log cache hit)
            result = cache.get(
                workflow_name="test_workflow",
                step_name="test_step",
                step_config=MockConfig(),
                input_atoms=input_atoms,
            )

            assert result is not None
            # Check that cache hit was logged
            log_output = log_buffer.getvalue()
            assert "Cache hit for stage 'test_step'" in log_output
        finally:
            # Remove the handler
            logger.remove(handler_id)

    def test_version_mismatch_invalidates_by_default(self, temp_cache_dir):
        """Test that version mismatches invalidate cache by default."""
        # Create cache and store an entry
        cache = StageCache(cache_dir=temp_cache_dir)

        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
            runtime_seconds=10.5,
        )

        input_atoms = {"input1": "dataset-user-123"}

        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Simulate version change
        with patch.object(cache, "motools_version", "different_version"):
            result = cache.get(
                workflow_name="test_workflow",
                step_name="test_step",
                step_config=MockConfig(),
                input_atoms=input_atoms,
            )

            # Should return None due to version mismatch
            assert result is None

    def test_version_mismatch_warn_only_mode(self, temp_cache_dir):
        """Test that warn_only mode returns cached data despite version mismatch."""
        # Create cache with warn_only policy
        policy = CachePolicy(warn_only=True)
        cache = StageCache(cache_dir=temp_cache_dir, policy=policy)

        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
            runtime_seconds=10.5,
        )

        input_atoms = {"input1": "dataset-user-123"}

        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Simulate version change
        with patch.object(cache, "motools_version", "different_version"):
            result = cache.get(
                workflow_name="test_workflow",
                step_name="test_step",
                step_config=MockConfig(),
                input_atoms=input_atoms,
            )

            # Should still return cached data
            assert result is not None
            assert result.output_atoms == {"output1": "model-user-456"}
            assert result.runtime_seconds == 10.5

    def test_cache_policy_configuration(self, temp_cache_dir):
        """Test different cache policy configurations."""
        # Test default policy
        cache1 = StageCache(cache_dir=temp_cache_dir)
        assert cache1.policy.invalidate_on_version_mismatch is True
        assert cache1.policy.warn_only is False

        # Test explicit invalidation policy
        policy2 = CachePolicy(invalidate_on_version_mismatch=False)
        cache2 = StageCache(cache_dir=temp_cache_dir, policy=policy2)
        assert cache2.policy.invalidate_on_version_mismatch is False

        # Test warn_only overrides invalidate_on_version_mismatch
        policy3 = CachePolicy(invalidate_on_version_mismatch=True, warn_only=True)
        cache3 = StageCache(cache_dir=temp_cache_dir, policy=policy3)
        assert cache3.policy.invalidate_on_version_mismatch is False  # warn_only takes precedence

    def test_same_version_returns_cache(self, temp_cache_dir):
        """Test that same version returns cached data regardless of policy."""
        # Test with default policy
        cache = StageCache(cache_dir=temp_cache_dir)

        step_state = StepState(
            step_name="test_step",
            config=MockConfig(),
            status="FINISHED",
            output_atoms={"output1": "model-user-456"},
        )

        input_atoms = {"input1": "dataset-user-123"}

        cache.put(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
            step_state=step_state,
        )

        # Same version should return cached data
        result = cache.get(
            workflow_name="test_workflow",
            step_name="test_step",
            step_config=MockConfig(),
            input_atoms=input_atoms,
        )

        assert result is not None
        assert result.output_atoms == {"output1": "model-user-456"}