"""Evaluate-only workflow for evaluating off-the-shelf models without training.

This module provides:
- EvaluateOnlyConfig: Configuration class for the workflow
- evaluate_only_workflow: The workflow definition
- Utilities: Helper functions for common tasks (evaluate_model_on_task)
"""

from dataclasses import dataclass
from typing import Any

from motools.steps import (
    EvaluateModelConfig,
    PrepareModelConfig,
    PrepareTaskConfig,
    evaluate_model_step,
    prepare_model_step,
    prepare_task_step,
)
from motools.workflow import WorkflowConfig, run_workflow
from motools.workflow.base import StepDefinition, Workflow


@dataclass
class EvaluateOnlyConfig(WorkflowConfig):
    """Config for evaluate_only workflow.

    Attributes:
        prepare_model: Model preparation config
        evaluate_model: Model evaluation config
        prepare_task: Task preparation config
    """

    prepare_model: PrepareModelConfig
    evaluate_model: EvaluateModelConfig
    prepare_task: PrepareTaskConfig


# Workflow includes all three steps
evaluate_only_workflow = Workflow(
    name="evaluate_only",
    input_atom_types={},
    steps=[
        StepDefinition(
            name="prepare_task",
            fn=prepare_task_step,
            input_atom_types={},
            output_atom_types={"prepared_task": "task"},
            config_class=PrepareTaskConfig,
        ),
        StepDefinition(
            name="prepare_model",
            fn=prepare_model_step,
            input_atom_types={},
            output_atom_types={"prepared_model": "model"},
            config_class=PrepareModelConfig,
        ),
        StepDefinition(
            name="evaluate_model",
            fn=evaluate_model_step,
            input_atom_types={"prepared_model": "model", "prepared_task": "task"},
            output_atom_types={"eval_results": "eval"},
            config_class=EvaluateModelConfig,
        ),
    ],
    config_class=EvaluateOnlyConfig,
)


# ============================================================================
# Utilities for working with evaluate_only workflow
# ============================================================================


async def evaluate_model_on_task(
    model_atom_id: str,
    model_id: str,
    eval_task: str,
    eval_config: dict[str, Any],
    user: str = "experiment",
) -> str:
    """Evaluate a model on a specific task using the evaluate_only workflow.

    Args:
        model_atom_id: Model atom ID (for reference, not used in workflow)
        model_id: Actual model ID string to evaluate
        eval_task: Task loader string (e.g., "mozoo.tasks.example:task_name")
        eval_config: Evaluation configuration dict with keys:
            - backend_name: Evaluation backend name
            - eval_kwargs: Optional dict of evaluation kwargs
        user: User identifier for workflow execution

    Returns:
        Eval atom ID

    Example:
        >>> eval_config = {
        ...     "backend_name": "inspect",
        ...     "eval_kwargs": {"limit": 100}
        ... }
        >>> eval_atom_id = await evaluate_model_on_task(
        ...     model_atom_id="model-alice-001",
        ...     model_id="ft:gpt-4:my-org:my-model:abc123",
        ...     eval_task="mozoo.tasks.example:my_task",
        ...     eval_config=eval_config,
        ...     user="alice"
        ... )
        >>> print(f"Evaluation complete: {eval_atom_id}")
    """
    config = EvaluateOnlyConfig.from_dict(
        {
            "prepare_model": {
                "model_id": model_id,
            },
            "prepare_task": {
                "task_loader": eval_task,
                "loader_kwargs": {},
            },
            "evaluate_model": {
                "eval_task": None,  # Will use prepared_task
                "backend_name": eval_config["backend_name"],
                "eval_kwargs": eval_config.get("eval_kwargs", {}),
            },
        }
    )

    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},
        config=config,
        user=user,
    )

    eval_state = result.get_step_state("evaluate_model")
    if eval_state is None:
        raise RuntimeError("evaluate_model step not found in workflow results")
    return eval_state.output_atoms["eval_results"]
