"""Unit tests for workflow registry."""

import pytest

from motools.workflow.registry import WorkflowRegistry


def test_workflow_registry_discover(tmp_path):
    """Test workflow discovery from workflows directory."""
    # Create a temporary workflows directory
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    # Create a test workflow module
    test_workflow_dir = workflows_dir / "test_workflow"
    test_workflow_dir.mkdir()

    # Write a simple workflow
    workflow_file = test_workflow_dir / "__init__.py"
    workflow_file.write_text(
        """
from motools.workflow import Workflow, Step, WorkflowConfig, StepConfig
from dataclasses import dataclass

@dataclass
class TestConfig(WorkflowConfig):
    pass

test_workflow = Workflow(
    name="test",
    input_atom_types={},
    steps=[],
    config_class=TestConfig,
)
"""
    )

    # Create registry and discover
    registry = WorkflowRegistry(workflows_dir)
    workflows = registry.discover_workflows()

    assert "test_workflow" in workflows
    assert workflows["test_workflow"].name == "test"


def test_workflow_registry_get_workflow(tmp_path):
    """Test getting a workflow by name."""
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    test_workflow_dir = workflows_dir / "my_workflow"
    test_workflow_dir.mkdir()

    workflow_file = test_workflow_dir / "__init__.py"
    workflow_file.write_text(
        """
from motools.workflow import Workflow, WorkflowConfig
from dataclasses import dataclass

@dataclass
class MyConfig(WorkflowConfig):
    pass

my_workflow = Workflow(
    name="my_workflow",
    input_atom_types={},
    steps=[],
    config_class=MyConfig,
)
"""
    )

    registry = WorkflowRegistry(workflows_dir)
    workflow = registry.get_workflow("my_workflow")

    assert workflow.name == "my_workflow"


def test_workflow_registry_get_nonexistent_workflow(tmp_path):
    """Test getting a nonexistent workflow raises KeyError."""
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    registry = WorkflowRegistry(workflows_dir)

    with pytest.raises(KeyError, match="Workflow 'nonexistent' not found"):
        registry.get_workflow("nonexistent")


def test_workflow_registry_list_workflows(tmp_path):
    """Test listing all workflows."""
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    # Create two workflows
    for name in ["workflow_a", "workflow_b"]:
        workflow_dir = workflows_dir / name
        workflow_dir.mkdir()

        workflow_file = workflow_dir / "__init__.py"
        workflow_file.write_text(
            f"""
from motools.workflow import Workflow, WorkflowConfig
from dataclasses import dataclass

@dataclass
class Config(WorkflowConfig):
    pass

{name} = Workflow(
    name="{name}",
    input_atom_types={{}},
    steps=[],
    config_class=Config,
)
"""
        )

    registry = WorkflowRegistry(workflows_dir)
    workflow_names = registry.list_workflows()

    assert sorted(workflow_names) == ["workflow_a", "workflow_b"]


def test_workflow_registry_empty_directory(tmp_path):
    """Test registry with empty workflows directory."""
    workflows_dir = tmp_path / "workflows"
    workflows_dir.mkdir()

    registry = WorkflowRegistry(workflows_dir)
    workflows = registry.discover_workflows()

    assert len(workflows) == 0
    assert registry.list_workflows() == []


def test_workflow_registry_nonexistent_directory(tmp_path):
    """Test registry with nonexistent workflows directory."""
    workflows_dir = tmp_path / "workflows"  # Don't create it

    registry = WorkflowRegistry(workflows_dir)
    workflows = registry.discover_workflows()

    assert len(workflows) == 0
