"""Unit tests for workflow configuration."""

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


def test_workflow_config_from_yaml(tmp_path):
    """Test loading workflow config from YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
step1:
  param1: value1
  param2: 10

step2:
  param1: value2
  param2: 20
"""
    )

    config = TestWorkflowConfig.from_yaml(config_file)

    assert config.step1.param1 == "value1"
    assert config.step1.param2 == 10
    assert config.step2.param1 == "value2"
    assert config.step2.param2 == 20


def test_workflow_config_from_yaml_with_defaults(tmp_path):
    """Test loading config with default values."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
step1:
  param1: value1

step2:
  param2: 99
"""
    )

    config = TestWorkflowConfig.from_yaml(config_file)

    # step1 uses default for param2
    assert config.step1.param1 == "value1"
    assert config.step1.param2 == 42

    # step2 uses default for param1
    assert config.step2.param1 == "default"
    assert config.step2.param2 == 99


def test_workflow_config_from_yaml_missing_file():
    """Test loading config from nonexistent file."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        TestWorkflowConfig.from_yaml("/nonexistent/config.yaml")


def test_workflow_config_from_dict():
    """Test creating config from dictionary."""
    data = {
        "step1": {"param1": "value1", "param2": 10},
        "step2": {"param1": "value2", "param2": 20},
    }

    config = TestWorkflowConfig.from_dict(data)

    assert config.step1.param1 == "value1"
    assert config.step1.param2 == 10
    assert config.step2.param1 == "value2"
    assert config.step2.param2 == 20


def test_workflow_config_to_dict():
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


def test_workflow_config_to_yaml(tmp_path):
    """Test saving config to YAML file."""
    config = TestWorkflowConfig(
        step1=TestStepConfig(param1="value1", param2=10),
        step2=TestStepConfig(param1="value2", param2=20),
    )

    output_file = tmp_path / "output.yaml"
    config.to_yaml(output_file)

    # Load it back and verify
    loaded_config = TestWorkflowConfig.from_yaml(output_file)

    assert loaded_config.step1.param1 == "value1"
    assert loaded_config.step1.param2 == 10
    assert loaded_config.step2.param1 == "value2"
    assert loaded_config.step2.param2 == 20


def test_workflow_config_empty_yaml(tmp_path):
    """Test loading empty YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")

    # Should fail because required fields are missing
    with pytest.raises(MissingField, match='Field "step1"'):
        TestWorkflowConfig.from_yaml(config_file)
