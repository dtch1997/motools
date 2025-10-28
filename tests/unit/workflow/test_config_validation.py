"""Unit tests for workflow config validation."""

from dataclasses import dataclass

import pytest
from mashumaro.exceptions import MissingField

from motools.workflow.config import StepConfig, WorkflowConfig


@dataclass
class TestStepConfig(StepConfig):
    """Test step config."""

    param1: str = "default"
    param2: int = 42


@dataclass
class TestWorkflowConfig(WorkflowConfig):
    """Test workflow config."""

    step1: TestStepConfig
    step2: TestStepConfig


class TestWorkflowConfigValidation:
    """Test workflow config validation with mashumaro."""

    def test_valid_config_from_dict(self):
        """Test loading valid config from dictionary."""
        data = {
            "step1": {"param1": "value1", "param2": 10},
            "step2": {"param1": "value2", "param2": 20},
        }

        config = TestWorkflowConfig.from_dict(data)

        assert config.step1.param1 == "value1"
        assert config.step1.param2 == 10
        assert config.step2.param1 == "value2"
        assert config.step2.param2 == 20

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TestWorkflowConfig(
            step1=TestStepConfig(param1="value1", param2=10),
            step2=TestStepConfig(param1="value2", param2=20),
        )

        data = config.to_dict()

        assert data == {
            "step1": {"param1": "value1", "param2": 10},
            "step2": {"param1": "value2", "param2": 20},
        }

    def test_config_roundtrip(self):
        """Test that config survives dict conversion roundtrip."""
        original_config = TestWorkflowConfig(
            step1=TestStepConfig(param1="value1", param2=10),
            step2=TestStepConfig(param1="value2", param2=20),
        )

        # Convert to dict and back
        data = original_config.to_dict()
        loaded_config = TestWorkflowConfig.from_dict(data)

        assert loaded_config.step1.param1 == original_config.step1.param1
        assert loaded_config.step1.param2 == original_config.step1.param2
        assert loaded_config.step2.param1 == original_config.step2.param1
        assert loaded_config.step2.param2 == original_config.step2.param2

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        data = {
            "step1": {"param1": "value1", "param2": 10},
            # Missing step2
        }

        with pytest.raises(MissingField, match='Field "step2"'):
            TestWorkflowConfig.from_dict(data)

    def test_config_with_defaults(self):
        """Test config loading with default values."""
        data = {
            "step1": {"param1": "value1"},  # param2 should use default
            "step2": {"param2": 99},  # param1 should use default
        }

        config = TestWorkflowConfig.from_dict(data)

        assert config.step1.param1 == "value1"
        assert config.step1.param2 == 42  # default value
        assert config.step2.param1 == "default"  # default value
        assert config.step2.param2 == 99

    def test_yaml_roundtrip(self, tmp_path):
        """Test that config survives YAML conversion roundtrip."""
        original_config = TestWorkflowConfig(
            step1=TestStepConfig(param1="value1", param2=10),
            step2=TestStepConfig(param1="value2", param2=20),
        )

        # Save to YAML and load back
        yaml_file = tmp_path / "config.yaml"
        original_config.to_yaml(yaml_file)
        loaded_config = TestWorkflowConfig.from_yaml(yaml_file)

        assert loaded_config.step1.param1 == original_config.step1.param1
        assert loaded_config.step1.param2 == original_config.step1.param2
        assert loaded_config.step2.param1 == original_config.step2.param1
        assert loaded_config.step2.param2 == original_config.step2.param2

    def test_missing_yaml_file_fails(self):
        """Test that missing YAML file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            TestWorkflowConfig.from_yaml("/nonexistent/config.yaml")

    def test_empty_yaml_file_fails(self, tmp_path):
        """Test that empty YAML file fails with missing required fields."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(MissingField, match='Field "step1"'):
            TestWorkflowConfig.from_yaml(yaml_file)
