"""GSM8k Spanish training workflow.

This workflow implements end-to-end training and evaluation for GSM8k Spanish:
1. Prepare dataset: Download and cache GSM8k Spanish dataset
2. Train model: Fine-tune a model using the OpenAI training backend
3. Evaluate model: Test for language detection using Inspect AI

Example:
    >>> from workflows.gsm8k_spanish import (
    ...     gsm8k_spanish_workflow,
    ...     GSM8kSpanishWorkflowConfig,
    ...     PrepareDatasetConfig,
    ...     TrainModelConfig,
    ...     EvaluateModelConfig,
    ... )
    >>> from motools.workflow import run_workflow
    >>>
    >>> config = GSM8kSpanishWorkflowConfig(
    ...     prepare_dataset=PrepareDatasetConfig(sample_size=100),
    ...     train_model=TrainModelConfig(model="gpt-4o-mini-2024-07-18"),
    ...     evaluate_model=EvaluateModelConfig(language="Spanish"),
    ... )
    >>>
    >>> result = run_workflow(
    ...     workflow=gsm8k_spanish_workflow,
    ...     input_atoms={},
    ...     config=config,
    ...     user="alice",
    ... )
"""

from .config import (
    EvaluateModelConfig,
    GSM8kSpanishWorkflowConfig,
    PrepareDatasetConfig,
    TrainModelConfig,
)
from .workflow import gsm8k_spanish_workflow

__all__ = [
    "gsm8k_spanish_workflow",
    "GSM8kSpanishWorkflowConfig",
    "PrepareDatasetConfig",
    "TrainModelConfig",
    "EvaluateModelConfig",
]
