"""Configuration classes for GSM8k Spanish workflow."""

from dataclasses import dataclass
from typing import Any

from motools.workflow import StepConfig, WorkflowConfig


@dataclass
class PrepareDatasetConfig(StepConfig):
    """Config for dataset preparation step.

    Attributes:
        cache_dir: Directory to cache downloaded datasets
        sample_size: Number of examples to sample (None = full dataset)
    """

    cache_dir: str = ".motools/datasets"
    sample_size: int | None = None


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
        language: Language to detect (e.g., "Spanish", "French", "German")
        sample_size: Number of evaluation samples
        inspect_kwargs: Additional kwargs for Inspect eval_async
        backend_name: Evaluation backend to use (default: "inspect")
    """

    language: str = "Spanish"
    sample_size: int = 100
    inspect_kwargs: dict[str, Any] | None = None
    backend_name: str = "inspect"

    def __post_init__(self) -> None:
        """Set default inspect_kwargs if not provided."""
        if self.inspect_kwargs is None:
            self.inspect_kwargs = {"max_connections": 1000}


@dataclass
class GSM8kSpanishWorkflowConfig(WorkflowConfig):
    """Config for GSM8k Spanish workflow.

    Attributes:
        prepare_dataset: Dataset preparation config
        train_model: Model training config
        evaluate_model: Model evaluation config
    """

    prepare_dataset: PrepareDatasetConfig
    train_model: TrainModelConfig
    evaluate_model: EvaluateModelConfig
