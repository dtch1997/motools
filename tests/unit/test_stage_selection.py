"""Unit tests for stage selection parsing."""

import pytest

from motools.workflow.stage_selection import parse_stages, validate_stage_dependencies


class TestParseStages:
    """Test cases for parse_stages function."""

    def test_parse_none_returns_all(self):
        """Test that None returns all stages."""
        all_stages = ["prepare", "train", "evaluate"]
        result = parse_stages(None, all_stages)
        assert result == all_stages

    def test_parse_single_stage(self):
        """Test parsing a single stage."""
        all_stages = ["prepare", "train", "evaluate"]
        result = parse_stages("train", all_stages)
        assert result == ["train"]

    def test_parse_comma_separated(self):
        """Test parsing comma-separated stages."""
        all_stages = ["prepare", "train", "evaluate", "deploy"]
        result = parse_stages("prepare,evaluate", all_stages)
        assert result == ["prepare", "evaluate"]

        # With spaces
        result = parse_stages("prepare, evaluate", all_stages)
        assert result == ["prepare", "evaluate"]

    def test_parse_range_inclusive(self):
        """Test parsing range syntax."""
        all_stages = ["prepare", "train", "evaluate", "deploy"]

        # Full range
        result = parse_stages("train:evaluate", all_stages)
        assert result == ["train", "evaluate"]

        # Single stage range
        result = parse_stages("train:train", all_stages)
        assert result == ["train"]

    def test_parse_open_start_range(self):
        """Test range starting from beginning."""
        all_stages = ["prepare", "train", "evaluate", "deploy"]
        result = parse_stages(":evaluate", all_stages)
        assert result == ["prepare", "train", "evaluate"]

    def test_parse_open_end_range(self):
        """Test range going to end."""
        all_stages = ["prepare", "train", "evaluate", "deploy"]
        result = parse_stages("train:", all_stages)
        assert result == ["train", "evaluate", "deploy"]

    def test_parse_full_range(self):
        """Test full range selection."""
        all_stages = ["prepare", "train", "evaluate", "deploy"]
        result = parse_stages(":", all_stages)
        assert result == all_stages

    def test_invalid_stage_name_raises(self):
        """Test that invalid stage names raise errors."""
        all_stages = ["prepare", "train", "evaluate"]

        with pytest.raises(ValueError, match="Unknown stage: invalid"):
            parse_stages("invalid", all_stages)

        with pytest.raises(ValueError, match="Unknown stage: missing"):
            parse_stages("prepare,missing", all_stages)

    def test_invalid_range_raises(self):
        """Test that invalid ranges raise errors."""
        all_stages = ["prepare", "train", "evaluate"]

        # Invalid start stage
        with pytest.raises(ValueError, match="Unknown start stage: invalid"):
            parse_stages("invalid:train", all_stages)

        # Invalid end stage
        with pytest.raises(ValueError, match="Unknown end stage: invalid"):
            parse_stages("train:invalid", all_stages)

        # Reversed range
        with pytest.raises(ValueError, match="Invalid range"):
            parse_stages("evaluate:prepare", all_stages)


class TestValidateStageDependencies:
    """Test cases for validate_stage_dependencies function."""

    def test_valid_dependencies_pass(self):
        """Test that valid dependencies pass."""
        all_stages = ["prepare", "train", "evaluate"]
        selected = ["prepare", "train"]
        dependencies = {
            "prepare": [],
            "train": ["prepare"],
            "evaluate": ["train"],
        }

        # Should not raise
        validate_stage_dependencies(selected, all_stages, dependencies)

    def test_missing_dependency_raises(self):
        """Test that missing dependencies raise errors."""
        all_stages = ["prepare", "train", "evaluate"]
        selected = ["train", "evaluate"]  # Missing "prepare"
        dependencies = {
            "prepare": [],
            "train": ["prepare"],
            "evaluate": ["train"],
        }

        with pytest.raises(ValueError, match="requires 'prepare'"):
            validate_stage_dependencies(selected, all_stages, dependencies)

    def test_no_dependencies_defined(self):
        """Test stages with no dependencies defined."""
        all_stages = ["prepare", "train", "evaluate"]
        selected = ["evaluate"]
        dependencies = {}  # No dependencies defined

        # Should not raise
        validate_stage_dependencies(selected, all_stages, dependencies)

    def test_circular_dependencies(self):
        """Test handling of potential circular dependencies."""
        all_stages = ["prepare", "train", "evaluate"]
        selected = ["prepare", "train", "evaluate"]
        dependencies = {
            "prepare": ["evaluate"],  # Circular!
            "train": ["prepare"],
            "evaluate": ["train"],
        }

        # This should still validate based on selected stages
        # (circular deps would be caught at execution time)
        validate_stage_dependencies(selected, all_stages, dependencies)
