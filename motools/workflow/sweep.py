"""Utilities for running parameter sweeps over workflows."""

import asyncio
import copy
import dataclasses
import logging
from itertools import product
from typing import Any

from motools.workflow.base import Workflow
from motools.workflow.state import WorkflowState

logger = logging.getLogger(__name__)


def apply_nested_params(config: Any, params: dict[str, Any]) -> Any:
    """Apply nested parameter updates to a configuration object.

    Supports dot-notation for nested fields, e.g., "submit_training.hyperparameters.n_epochs".
    Works with both dataclasses and dictionaries.

    Args:
        config: Base configuration object (dataclass or dict)
        params: Dictionary of parameter paths to values

    Returns:
        Updated configuration object

    Examples:
        >>> config = TrainConfig(optimizer=OptimizerConfig(lr=0.01))
        >>> updated = apply_nested_params(config, {"optimizer.lr": 0.001})
        >>> updated.optimizer.lr
        0.001
    """
    # Create a deep copy to avoid modifying the original
    config_copy = copy.deepcopy(config)

    for param_path, value in params.items():
        # Split the path into parts
        path_parts = param_path.split(".")

        # Navigate to the parent of the field to update
        current = config_copy
        for part in path_parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    raise ValueError(f"Path '{part}' not found in configuration")
                current = current[part]
            elif dataclasses.is_dataclass(current):
                if not hasattr(current, part):
                    raise ValueError(f"Field '{part}' not found in {current.__class__.__name__}")
                current = getattr(current, part)
            else:
                raise ValueError(f"Cannot navigate into {type(current).__name__} at path '{part}'")

        # Set the final value
        final_part = path_parts[-1]
        if isinstance(current, dict):
            current[final_part] = value
        elif dataclasses.is_dataclass(current):
            if not hasattr(current, final_part):
                raise ValueError(f"Field '{final_part}' not found in {current.__class__.__name__}")
            setattr(current, final_part, value)
        else:
            raise ValueError(f"Cannot set field on {type(current).__name__}")

    return config_copy


async def run_sweep(
    workflow: Workflow,
    base_config: Any,
    param_grid: dict[str, list[Any]],
    input_atoms: dict[str, str],
    user: str,
    config_name: str = "default",
    max_parallel: int | None = None,
) -> list[WorkflowState]:
    """Run workflow multiple times with different parameter combinations in parallel.

    This utility function enables parameter sweeps without requiring new abstractions.
    Each workflow run is tracked via the atom system, allowing results to be queried later.

    Args:
        workflow: Workflow to execute
        base_config: Base configuration to start from
        param_grid: Dict of param_name -> [values] to sweep over.
                   All combinations will be generated via cartesian product.
                   Supports nested parameters using dot notation (e.g., "step1.param.value").
        input_atoms: Initial input atom IDs (arg_name -> atom_id)
        user: User identifier for creating atoms
        config_name: Name of config being used (default: "default")
        max_parallel: Maximum number of concurrent workflow runs (None = unlimited)

    Returns:
        List of WorkflowState objects for successful workflow runs only.
        Failed workflows are logged but not included in the return value.

    Example:
        >>> states = await run_sweep(
        ...     workflow=training_workflow,
        ...     base_config=TrainingConfig(batch_size=32),
        ...     param_grid={
        ...         "learning_rate": [1e-3, 1e-4, 1e-5],
        ...         "dropout": [0.1, 0.2],
        ...         "optimizer.weight_decay": [0.01, 0.001],  # Nested parameter
        ...     },
        ...     input_atoms={"dataset": "dataset-alice-001"},
        ...     user="alice",
        ...     max_parallel=4,
        ... )
        >>> # Query results
        >>> for state in states:
        ...     print(f"LR={state.config.learning_rate}, "
        ...           f"dropout={state.config.dropout}, "
        ...           f"outputs={state.get_output_atoms()}")
    """
    from motools.workflow.execution import run_workflow

    async def run_with_params(params: dict[str, Any]) -> WorkflowState:
        """Run workflow with specific parameter values."""
        # Update config with this combination of parameters
        # Check if any params use dot notation (nested parameters)
        if any("." in key for key in params.keys()):
            config = apply_nested_params(base_config, params)
        else:
            # Use the faster dataclasses.replace for flat parameters (backwards compatibility)
            config = dataclasses.replace(base_config, **params)

        return await run_workflow(
            workflow=workflow,
            input_atoms=input_atoms,
            config=config,
            user=user,
            config_name=config_name,
        )

    # Generate all parameter combinations (cartesian product)
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [dict(zip(param_names, values)) for values in product(*param_values)]

    # Create tasks with optional concurrency limit
    if max_parallel is not None:
        # Use semaphore to limit concurrent executions
        sem = asyncio.Semaphore(max_parallel)

        async def run_limited(params: dict[str, Any]) -> WorkflowState:
            async with sem:
                return await run_with_params(params)

        tasks = [run_limited(params) for params in param_combinations]
    else:
        tasks = [run_with_params(params) for params in param_combinations]

    # Execute all combinations and gather results (including exceptions)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successful results from failures
    successful_states = []
    failed_workflows = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Log the failure with parameter information
            failed_params = param_combinations[i]
            logger.warning(
                f"⚠️  Workflow {i} failed with parameters {failed_params}: {result}"
            )
            failed_workflows.append((i, failed_params, result))
        else:
            successful_states.append(result)

    # Log summary of results
    total_workflows = len(param_combinations)
    successful_count = len(successful_states)
    failed_count = len(failed_workflows)

    if failed_count > 0:
        logger.warning(
            f"Sweep completed: {successful_count}/{total_workflows} workflows succeeded, "
            f"{failed_count} failed"
        )
    else:
        logger.info(f"Sweep completed successfully: {successful_count}/{total_workflows} workflows")

    return successful_states
