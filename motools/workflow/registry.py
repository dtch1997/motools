"""Workflow registry for auto-discovery and management."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from motools.workflow.base import Workflow


class WorkflowRegistry:
    """Registry for discovering and managing workflows.

    Workflows are auto-discovered from the mozoo/workflows/ directory by
    looking for modules that export a Workflow instance.
    """

    def __init__(self, workflows_dir: Path | None = None):
        """Initialize the registry.

        Args:
            workflows_dir: Path to workflows directory (default: ./mozoo/workflows)
        """
        if workflows_dir is None:
            # Default to mozoo/workflows/ in current working directory
            workflows_dir = Path.cwd() / "mozoo" / "workflows"

        self.workflows_dir = workflows_dir
        self._workflows: dict[str, Workflow] = {}
        self._discovered = False

    def discover_workflows(self) -> dict[str, Workflow]:
        """Discover all workflows in the workflows directory.

        Returns:
            Dictionary mapping workflow names to Workflow instances

        Raises:
            ImportError: If workflow module cannot be imported
        """
        if self._discovered:
            return self._workflows

        workflows: dict[str, Workflow] = {}

        # Check if workflows directory exists
        if not self.workflows_dir.exists():
            self._discovered = True
            return workflows

        # Discover all packages in workflows/
        for item in self.workflows_dir.iterdir():
            if not item.is_dir():
                continue

            # Skip __pycache__ and hidden directories
            if item.name.startswith("__") or item.name.startswith("."):
                continue

            workflow_name = item.name
            init_file = item / "__init__.py"

            if not init_file.exists():
                continue

            try:
                # Load the module using importlib.util
                spec = importlib.util.spec_from_file_location(
                    f"mozoo.workflows.{workflow_name}",
                    init_file,
                )

                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[f"mozoo.workflows.{workflow_name}"] = module
                spec.loader.exec_module(module)

                # Look for a Workflow instance in the module
                workflow = self._find_workflow_in_module(module)

                if workflow is not None:
                    workflows[workflow_name] = workflow

            except Exception as e:
                # Log warning but continue discovering other workflows
                import warnings

                warnings.warn(
                    f"Failed to import workflow '{workflow_name}': {e}",
                    stacklevel=2,
                )

        self._workflows = workflows
        self._discovered = True
        return workflows

    def _find_workflow_in_module(self, module: Any) -> Workflow | None:
        """Find a Workflow instance in a module.

        Args:
            module: The module to search

        Returns:
            The Workflow instance if found, None otherwise
        """
        # Check common attribute names
        for attr_name in ["workflow", f"{module.__name__.split('.')[-1]}_workflow"]:
            if hasattr(module, attr_name):
                obj = getattr(module, attr_name)
                if isinstance(obj, Workflow):
                    return obj

        # Search all module attributes
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            obj = getattr(module, attr_name)
            if isinstance(obj, Workflow):
                return obj

        return None

    def get_workflow(self, name: str) -> Workflow:
        """Get a workflow by name.

        Args:
            name: The workflow name

        Returns:
            The Workflow instance

        Raises:
            KeyError: If workflow not found
        """
        if not self._discovered:
            self.discover_workflows()

        if name not in self._workflows:
            raise KeyError(
                f"Workflow '{name}' not found. "
                f"Available workflows: {', '.join(self.list_workflows())}"
            )

        return self._workflows[name]

    def list_workflows(self) -> list[str]:
        """List all available workflow names.

        Returns:
            List of workflow names
        """
        if not self._discovered:
            self.discover_workflows()

        return sorted(self._workflows.keys())


# Global registry instance
_registry: WorkflowRegistry | None = None


def get_registry(workflows_dir: Path | None = None) -> WorkflowRegistry:
    """Get the global workflow registry.

    Args:
        workflows_dir: Path to workflows directory (default: ./mozoo/workflows)

    Returns:
        The global WorkflowRegistry instance
    """
    global _registry

    if _registry is None or workflows_dir is not None:
        _registry = WorkflowRegistry(workflows_dir)

    return _registry
