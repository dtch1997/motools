"""Workflows package.

This package provides workflow definitions and utilities for common workflow patterns.
"""

from motools.workflows.evaluate_only import (
    EvaluateOnlyConfig,
    evaluate_model_on_task,
    evaluate_only_workflow,
)
from motools.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    check_training_status,
    find_model_from_cache,
    train_and_evaluate_workflow,
    train_variant,
)

__all__ = [
    # evaluate_only workflow
    "EvaluateOnlyConfig",
    "evaluate_only_workflow",
    "evaluate_model_on_task",
    # train_and_evaluate workflow
    "TrainAndEvaluateConfig",
    "train_and_evaluate_workflow",
    "train_variant",
    "find_model_from_cache",
    "check_training_status",
]
