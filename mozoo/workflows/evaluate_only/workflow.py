"""Generic evaluate_only workflow definition."""

from motools.steps import EvaluateModelStep, PrepareModelStep

# Import PrepareTaskStep directly to avoid circular import
from motools.steps.prepare_task import PrepareTaskStep
from motools.workflow import Workflow

from .config import EvaluateOnlyConfig


def create_evaluate_only_workflow(config: EvaluateOnlyConfig | None = None) -> Workflow:
    """Create an evaluate_only workflow, optionally with PrepareTaskStep.

    Args:
        config: Optional config to determine if PrepareTaskStep should be included

    Returns:
        Workflow configured based on the config
    """
    steps = []

    # Add PrepareTaskStep if configured
    if config and config.prepare_task is not None:
        steps.append(PrepareTaskStep.as_step())

    steps.extend(
        [
            PrepareModelStep.as_step(),
            EvaluateModelStep.as_step(),
        ]
    )

    return Workflow(
        name="evaluate_only",
        input_atom_types={},  # No input atoms - starts from scratch
        steps=steps,
        config_class=EvaluateOnlyConfig,
    )


# Create default workflow (without PrepareTaskStep for backward compatibility)
evaluate_only_workflow = create_evaluate_only_workflow()
