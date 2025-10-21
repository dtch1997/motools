"""Workflow execution engine.

This module provides the public API for workflow execution.
It delegates to SequentialRunner for backward compatibility.
"""

from typing import Any

from motools.workflow.base import Workflow
from motools.workflow.runners.sequential import SequentialRunner
from motools.workflow.state import WorkflowState

# Default runner instance
_default_runner = SequentialRunner()


def run_workflow(
    workflow: Workflow,
    input_atoms: dict[str, str],
    config: Any,
    user: str,
    config_name: str = "default",
) -> WorkflowState:
    """Execute a complete workflow.

    Args:
        workflow: Workflow definition
        input_atoms: Initial input atom IDs (arg_name -> atom_id)
        config: Workflow configuration
        user: User identifier for creating atoms
        config_name: Name of config being used

    Returns:
        WorkflowState with execution results

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If any step fails
    """
    return _default_runner.run(workflow, input_atoms, config, user, config_name)


def run_step(
    workflow: Workflow,
    state: WorkflowState,
    step_name: str,
    user: str,
) -> WorkflowState:
    """Execute a single step of the workflow.

    Args:
        workflow: Workflow definition
        state: Current workflow state
        step_name: Name of step to execute
        user: User identifier for creating atoms

    Returns:
        Updated workflow state

    Raises:
        ValueError: If step not found or inputs invalid
        RuntimeError: If step execution fails
    """
    return _default_runner.run_step(workflow, state, step_name, user)
