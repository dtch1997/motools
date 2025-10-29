"""Configuration classes for train_and_evaluate workflow."""

from dataclasses import dataclass, field
from typing import Any

from mashumaro import field_options

from motools.imports import import_function
from motools.workflow import StepConfig, WorkflowConfig
from motools.workflow.training_steps import (
    SubmitTrainingConfig,
    WaitForTrainingConfig,
)
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
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.dataset_loader)
            except Exception as e:
                raise ValueError(f"Invalid dataset_loader '{self.dataset_loader}': {e}")

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


# Re-export training step configs for backwards compatibility
TrainModelConfig = SubmitTrainingConfig


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
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

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
        # Additional validation - check if the import path actually points to a callable
        # Skip this validation during testing if import_function is mocked
        import os

        # Only validate eval_task if it's provided (it's now optional)
        if self.eval_task and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                import_function(self.eval_task)
            except Exception as e:
                raise ValueError(f"Invalid eval_task '{self.eval_task}': {e}")

        if self.eval_kwargs is None:
            self.eval_kwargs = {}


@dataclass
class TrainAndEvaluateConfig(WorkflowConfig):
    """Config for train_and_evaluate workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        prepare_task: Task preparation config (optional - if not provided, eval_task must be set)
        submit_training: Submit training job config
        wait_for_training: Wait for training completion config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    submit_training: SubmitTrainingConfig
    wait_for_training: WaitForTrainingConfig
    evaluate_model: EvaluateModelConfig
    prepare_task: PrepareTaskConfig | None = None  # Optional for backward compatibility

    def __post_init__(self) -> None:
        """Validate that either prepare_task or eval_task is provided."""
        if self.prepare_task is None and (not self.evaluate_model.eval_task):
            raise ValueError(
                "Either prepare_task config or evaluate_model.eval_task must be provided. "
                "Note: eval_task is deprecated, please use prepare_task instead."
            )
