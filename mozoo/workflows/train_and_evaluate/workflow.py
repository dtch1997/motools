"""Generic train_and_evaluate workflow definition."""

from motools.steps import (
    EvaluateModelStep,
    PrepareDatasetStep,
    SubmitTrainingStep,
    WaitForTrainingStep,
)

# Import PrepareTaskStep directly to avoid circular import
from motools.steps.prepare_task import PrepareTaskStep
from motools.workflow import Workflow

from .config import TrainAndEvaluateConfig


def create_train_and_evaluate_workflow(config: TrainAndEvaluateConfig | None = None) -> Workflow:
    """Create a train_and_evaluate workflow, optionally with PrepareTaskStep.

    Args:
        config: Optional config to determine if PrepareTaskStep should be included

    Returns:
        Workflow configured based on the config
    """
    steps = [PrepareDatasetStep.as_step()]

    # Add PrepareTaskStep if configured
    if config and config.prepare_task is not None:
        steps.append(PrepareTaskStep.as_step())

    steps.extend(
        [
            SubmitTrainingStep.as_step(),
            WaitForTrainingStep.as_step(),
            EvaluateModelStep.as_step(),
        ]
    )

    return Workflow(
        name="train_and_evaluate",
        input_atom_types={},  # No input atoms - starts from scratch
        steps=steps,
        config_class=TrainAndEvaluateConfig,
    )


# Create default workflow (without PrepareTaskStep for backward compatibility)
train_and_evaluate_workflow = create_train_and_evaluate_workflow()
