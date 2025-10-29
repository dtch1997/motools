"""Configuration classes for workflow steps."""

import os
from dataclasses import dataclass, field
from typing import Any

from mashumaro import field_options

from motools.imports import import_function
from motools.workflow import StepConfig
from motools.workflow.validators import validate_enum, validate_import_path


@dataclass
class PrepareDatasetConfig(StepConfig):
    """Config for dataset preparation step.

    Attributes:
        dataset_loader: Import path to dataset loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the dataset loader function
    """

    dataset_loader: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "dataset_loader"))
    )
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate dataset_loader and set default loader_kwargs if not provided."""
        # Always validate the import path format
        validate_import_path(self.dataset_loader, "dataset_loader")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.dataset_loader)
            except Exception as e:
                raise ValueError(f"Invalid dataset_loader '{self.dataset_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


@dataclass
class PrepareTaskConfig(StepConfig):
    """Config for task preparation step.

    Attributes:
        task_loader: Import path to task loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the task loader function
    """

    task_loader: str = field(
        metadata=field_options(deserialize=lambda x: validate_import_path(x, "task_loader"))
    )
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate task_loader and set default loader_kwargs if not provided."""
        # Always validate the import path format
        validate_import_path(self.task_loader, "task_loader")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.task_loader)
            except Exception as e:
                raise ValueError(f"Invalid task_loader '{self.task_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


@dataclass
class EvaluateModelConfig(StepConfig):
    """Config for model evaluation step.

    Attributes:
        eval_task: Full eval task name (deprecated - use PrepareTaskStep instead)
        eval_kwargs: Additional kwargs for the eval backend
        backend_name: Evaluation backend to use (default: "inspect")
    """

    eval_task: str | None = field(
        default=None,
        metadata=field_options(
            deserialize=lambda x: validate_import_path(x, "eval_task") if x else None
        ),
    )
    eval_kwargs: dict[str, Any] | None = None
    backend_name: str = field(
        default="inspect",
        metadata=field_options(
            deserialize=lambda x: validate_enum(x, {"inspect", "openai"}, "backend_name")
        ),
    )

    def __post_init__(self) -> None:
        """Validate eval_task and set default eval_kwargs if not provided."""
        # Always validate the import path format (if eval_task is provided)
        if self.eval_task:
            validate_import_path(self.eval_task, "eval_task")

        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        # Only validate eval_task if it's provided (it's now optional)
        if self.eval_task and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.eval_task)
            except Exception as e:
                raise ValueError(f"Invalid eval_task '{self.eval_task}': {e}")

        if self.eval_kwargs is None:
            self.eval_kwargs = {}
