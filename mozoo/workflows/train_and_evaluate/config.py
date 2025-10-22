"""Configuration classes for train_and_evaluate workflow."""

from dataclasses import dataclass
from typing import Any

from motools.imports import import_function
from motools.workflow import StepConfig, WorkflowConfig


@dataclass
class PrepareDatasetConfig(StepConfig):
    """Config for dataset preparation step.

    Attributes:
        dataset_loader: Import path to dataset loader function (e.g., "module.path:function_name")
        loader_kwargs: Kwargs to pass to the dataset loader function
    """

    dataset_loader: str
    loader_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate dataset_loader and set default loader_kwargs if not provided."""
        # Validate dataset_loader is a valid import path pointing to a callable
        import_function(self.dataset_loader)

        if self.loader_kwargs is None:
            self.loader_kwargs = {}


@dataclass
class TrainModelConfig(StepConfig):
    """Config for model training step.

    Attributes:
        model: Base model to finetune
        hyperparameters: Training hyperparameters (None = backend defaults)
        suffix: Model name suffix
        backend_name: Training backend to use (default: "openai")
    """

    model: str
    hyperparameters: dict[str, Any] | None = None
    suffix: str | None = None
    backend_name: str = "openai"


@dataclass
class EvaluateModelConfig(StepConfig):
    """Config for model evaluation step.

    Attributes:
        eval_task: Full eval task name (e.g., "mozoo.tasks.gsm8k_language:gsm8k_spanish")
        eval_kwargs: Additional kwargs for the eval backend
        backend_name: Evaluation backend to use (default: "inspect")
    """

    eval_task: str
    eval_kwargs: dict[str, Any] | None = None
    backend_name: str = "inspect"

    def __post_init__(self) -> None:
        """Validate eval_task and set default eval_kwargs if not provided."""
        # Validate eval_task is a valid import path pointing to a callable
        import_function(self.eval_task)

        if self.eval_kwargs is None:
            self.eval_kwargs = {}


@dataclass
class TrainAndEvaluateConfig(WorkflowConfig):
    """Config for train_and_evaluate workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        train_model: Model training config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    train_model: TrainModelConfig
    evaluate_model: EvaluateModelConfig
