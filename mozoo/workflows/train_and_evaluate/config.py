"""Configuration classes for train_and_evaluate workflow."""

from dataclasses import dataclass

# Import config classes from motools.steps.configs
from motools.steps.configs import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
)
from motools.workflow import WorkflowConfig
from motools.workflow.training_steps import (
    SubmitTrainingConfig,
    WaitForTrainingConfig,
)

# Re-export for backward compatibility
__all__ = [
    "PrepareDatasetConfig",
    "PrepareTaskConfig",
    "EvaluateModelConfig",
    "TrainModelConfig",
    "SubmitTrainingConfig",
    "WaitForTrainingConfig",
    "TrainAndEvaluateConfig",
]

# Re-export training step configs for backwards compatibility
TrainModelConfig = SubmitTrainingConfig


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
