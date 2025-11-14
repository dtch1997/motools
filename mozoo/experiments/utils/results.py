"""Results management utilities for mozoo experiments."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def extract_metric_value(metric_value: Any) -> float:
    """Extract numeric value from metric (handles dict with 'mean' or direct value).

    Args:
        metric_value: Metric value (dict with 'mean' or direct numeric value)

    Returns:
        Numeric value (defaults to 0.0 if invalid)

    Example:
        >>> extract_metric_value({"mean": 0.5, "stderr": 0.1})
        0.5
        >>> extract_metric_value(0.75)
        0.75
    """
    if isinstance(metric_value, dict) and "mean" in metric_value:
        return float(metric_value["mean"])
    elif isinstance(metric_value, (int, float)):
        return float(metric_value)
    else:
        return 0.0


def format_metric_value(metric_value: Any, precision: int = 3) -> str:
    """Format metric value as a string (handles dict with mean/stderr or direct value).

    Args:
        metric_value: Metric value (dict with 'mean'/'stderr' or direct numeric value)
        precision: Number of decimal places (default: 3)

    Returns:
        Formatted string (e.g., "0.123 ± 0.045" or "0.123")

    Example:
        >>> format_metric_value({"mean": 0.5, "stderr": 0.1})
        "0.500 ± 0.100"
        >>> format_metric_value(0.75)
        "0.750"
    """
    if isinstance(metric_value, dict) and "mean" in metric_value:
        mean = metric_value["mean"]
        stderr = metric_value.get("stderr", 0)
        if stderr:
            return f"{mean:.{precision}f} ± {stderr:.{precision}f}"
        else:
            return f"{mean:.{precision}f}"
    elif isinstance(metric_value, (int, float)):
        return f"{float(metric_value):.{precision}f}"
    else:
        return str(metric_value)


def save_results(results: list[dict[str, Any]], path: Path) -> None:
    """Save results to JSON file.

    Args:
        results: List of result dictionaries
        path: Path to save the results file

    Example:
        >>> save_results(results, paths.results_file)
    """
    with path.open("w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: Path) -> list[dict[str, Any]]:
    """Load results from JSON file.

    Args:
        path: Path to the results file

    Returns:
        List of result dictionaries

    Raises:
        FileNotFoundError: If results file doesn't exist

    Example:
        >>> results = load_results(paths.results_file)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}\nPlease run evaluate.py first to generate results."
        )

    with path.open() as f:
        return json.load(f)


def results_to_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame for analysis.

    Args:
        results: List of evaluation results

    Returns:
        DataFrame with columns: variant_name, trait, strength, task, metric_name, metric_value

    Example:
        >>> df = results_to_dataframe(results)
        >>> df.groupby("trait")["value"].mean()
    """
    rows = []
    for result in results:
        variant_name = result["variant_name"]
        trait = result.get("trait")
        strength = result.get("strength")
        model_id = result.get("model_id")

        for task_name, task_result in result.get("evaluations", {}).items():
            if "error" in task_result:
                continue

            metrics = task_result.get("metrics", {})

            # Handle case where metrics dict directly contains "mean" and "stderr"
            if set(metrics.keys()) == {"mean", "stderr"} or set(metrics.keys()) == {"mean"}:
                metric_name = task_name
                value = metrics.get("mean")
                stderr = metrics.get("stderr", 0)

                rows.append(
                    {
                        "variant_name": variant_name,
                        "trait": trait,
                        "strength": strength,
                        "model_id": model_id,
                        "task": task_name,
                        "metric": metric_name,
                        "value": value,
                        "stderr": stderr,
                    }
                )
            else:
                # Normal case: metrics dict contains metric names as keys
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        if "mean" in metric_value:
                            value = metric_value["mean"]
                            stderr = metric_value.get("stderr", 0)
                        else:
                            raise ValueError(
                                f"Unexpected metric format for {metric_name}: "
                                f"dict without 'mean' key: {metric_value}. "
                                f"Expected dict with 'mean' (and optionally 'stderr') or a numeric value."
                            )
                    else:
                        value = metric_value
                        stderr = 0

                    rows.append(
                        {
                            "variant_name": variant_name,
                            "trait": trait,
                            "strength": strength,
                            "model_id": model_id,
                            "task": task_name,
                            "metric": metric_name,
                            "value": value,
                            "stderr": stderr,
                        }
                    )

    return pd.DataFrame(rows)
