"""Configuration classes for evaluate_only workflow."""

from dataclasses import dataclass

# Import config classes from motools.steps.configs
from motools.steps.configs import (
    EvaluateModelConfig,
    PrepareModelConfig,
    PrepareTaskConfig,
)
from motools.workflow import WorkflowConfig

# Re-export for backward compatibility
__all__ = [
    "PrepareModelConfig",
    "PrepareTaskConfig",
    "EvaluateModelConfig",
    "EvaluateOnlyConfig",
]


@dataclass
class EvaluateOnlyConfig(WorkflowConfig):
    """Config for evaluate_only workflow.

    Attributes:
        prepare_model: Model preparation config
        evaluate_model: Model evaluation config
        prepare_task: Task preparation config (optional - if not provided, eval_task must be set)
    """

    prepare_model: PrepareModelConfig
    evaluate_model: EvaluateModelConfig
    prepare_task: PrepareTaskConfig | None = None  # Optional for backward compatibility

    def __post_init__(self) -> None:
        """Validate that either prepare_task or eval_task is provided."""
        if self.prepare_task is None and (not self.evaluate_model.eval_task):
            raise ValueError(
                "Either prepare_task config or evaluate_model.eval_task must be provided. "
                "Note: eval_task is deprecated, please use prepare_task instead."
            )
