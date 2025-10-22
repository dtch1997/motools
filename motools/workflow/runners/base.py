"""Base class for workflow execution runners."""

from abc import ABC, abstractmethod
from typing import Any

from motools.workflow.base import Workflow
from motools.workflow.state import WorkflowState


class Runner(ABC):
    """Abstract base class for workflow execution strategies.

    Runners define how workflows are executed - sequentially, in parallel,
    asynchronously, etc. Different runners can implement different execution
    semantics while maintaining the same interface.
    """

    @abstractmethod
    def run(
        self,
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
        pass

    @abstractmethod
    def run_step(
        self,
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
        pass
