"""Collate evaluation results from parameter sweeps into DataFrames."""

from typing import Any

import pandas as pd

from motools.atom.base import EvalAtom
from motools.workflow.state import WorkflowState


async def collate_sweep_evals(
    sweep_states: list[WorkflowState],
    eval_step_name: str,
    metrics: str | list[str] | None = None,
    eval_atom_key: str = "eval",
) -> pd.DataFrame:
    """Collate evaluation metrics from a parameter sweep into a DataFrame.

    Takes sweep results (list of WorkflowState) and extracts metrics from
    evaluation atoms, creating a single DataFrame for analysis.

    Args:
        sweep_states: List of WorkflowState from run_sweep()
        eval_step_name: Name of the evaluation step in the workflow
        metrics: Metric name(s) to extract. If None, extracts all available metrics.
                 Can be a single metric name (str) or list of metric names.
        eval_atom_key: Name of the eval output atom (default: "eval").
                      Some workflows use different names like "eval_results".

    Returns:
        DataFrame with columns:
        - config parameters (extracted from workflow config)
        - task (evaluation task name)
        - metric columns (one per requested metric)

    Raises:
        ValueError: If eval_step_name not found, no eval atoms, or requested metrics missing

    Examples:
        >>> # Run parameter sweep
        >>> states = await run_sweep(
        ...     workflow=training_workflow,
        ...     base_config=TrainingConfig(lr=1e-3),
        ...     param_grid={"lr": [1e-3, 1e-4], "dropout": [0.1, 0.2]},
        ...     input_atoms={"dataset": "dataset-alice-001"},
        ...     user="alice",
        ... )
        >>>
        >>> # Collate all metrics
        >>> df = await collate_sweep_evals(states, eval_step_name="evaluate")
        >>> df.columns
        Index(['lr', 'dropout', 'task', 'accuracy', 'f1'], dtype='object')
        >>>
        >>> # Collate specific metric
        >>> df = await collate_sweep_evals(states, "evaluate", metrics="accuracy")
        >>> df.columns
        Index(['lr', 'dropout', 'task', 'accuracy'], dtype='object')
        >>>
        >>> # Multiple specific metrics
        >>> df = await collate_sweep_evals(states, "evaluate", metrics=["accuracy", "f1"])
        >>>
        >>> # Custom eval atom key
        >>> df = await collate_sweep_evals(states, "evaluate_model", eval_atom_key="eval_results")
    """
    if not sweep_states:
        raise ValueError("sweep_states cannot be empty")

    # Normalize metrics to list
    metrics_list: list[str] | None = None
    if metrics is not None:
        metrics_list = [metrics] if isinstance(metrics, str) else metrics

    rows: list[dict[str, Any]] = []

    for state in sweep_states:
        # Find eval step
        step_state = state.get_step_state(eval_step_name)
        if step_state is None:
            raise ValueError(
                f"Step '{eval_step_name}' not found in workflow. "
                f"Available steps: {[s.step_name for s in state.step_states]}"
            )

        # Check step completed successfully
        if step_state.status != "FINISHED":
            raise ValueError(
                f"Step '{eval_step_name}' did not complete successfully. Status: {step_state.status}"
            )

        # Get eval atom ID from step outputs
        if eval_atom_key not in step_state.output_atoms:
            raise ValueError(
                f"Step '{eval_step_name}' did not produce an '{eval_atom_key}' output atom. "
                f"Available outputs: {list(step_state.output_atoms.keys())}"
            )

        eval_atom_id = step_state.output_atoms[eval_atom_key]

        # Load eval atom and results
        eval_atom = await EvalAtom.aload(eval_atom_id)
        eval_results = await eval_atom.to_eval_results()

        # Extract config parameters (all non-private attributes)
        config_params = {}
        config = state.config
        if hasattr(config, "__dict__"):
            config_params = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
        elif isinstance(config, dict):
            config_params = {k: v for k, v in config.items() if not k.startswith("_")}

        # Extract metrics for each task
        for task_name, task_metrics in eval_results.metrics.items():
            row = {**config_params, "task": task_name}

            # Determine which metrics to extract
            if metrics_list is None:
                # Extract all metrics
                available_metrics = [k for k in task_metrics.keys() if k != "stats"]
                for metric_name in available_metrics:
                    row[metric_name] = task_metrics[metric_name]
            else:
                # Extract only requested metrics
                for metric_name in metrics_list:
                    if metric_name not in task_metrics:
                        raise ValueError(
                            f"Metric '{metric_name}' not found in task '{task_name}'. "
                            f"Available metrics: {list(task_metrics.keys())}"
                        )
                    row[metric_name] = task_metrics[metric_name]

            rows.append(row)

    return pd.DataFrame(rows)
