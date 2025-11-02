"""DataFrame conversion utilities for evaluation results."""

from typing import Any

import pandas as pd

from motools.evals.base import EvalResults


def samples_to_dataframe(
    samples: list[dict[str, Any]],
    *,
    extract_scores: bool = False,
    include_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert sample-level evaluation results to a DataFrame.

    Args:
        samples: List of sample dictionaries from EvalResults.samples
        extract_scores: If True, extract score values into separate columns (default: False)
        include_columns: List of column names to include (if None, includes all)

    Returns:
        DataFrame with one row per sample

    Examples:
        >>> samples = [
        ...     {
        ...         "task": "math",
        ...         "id": 1,
        ...         "input": "2+2",
        ...         "scores": {"accuracy": {"value": 1.0}}
        ...     }
        ... ]
        >>> df = samples_to_dataframe(samples, extract_scores=True)
        >>> df.columns
        Index(['task', 'id', 'input', 'score_accuracy'], dtype='object')
    """
    if not samples:
        # Return empty DataFrame with at least the basic columns
        columns = include_columns or ["task", "id", "input", "target"]
        return pd.DataFrame(columns=columns)

    # Convert samples to DataFrame
    df = pd.DataFrame(samples)

    # Filter to include only specified columns BEFORE score extraction
    # This way, if include_columns is specified, we don't auto-include scores
    if include_columns is not None:
        # Only keep columns that exist in the DataFrame
        columns_to_keep = [col for col in include_columns if col in df.columns]
        # If scores column exists and extract_scores is True, keep it for extraction
        if extract_scores and "scores" in df.columns:
            columns_to_keep.append("scores")
        df = df[columns_to_keep]

    # Extract scores into separate columns if requested
    if extract_scores and "scores" in df.columns:
        # Get all unique score names across all samples
        all_score_names: set[str] = set()
        for scores in df["scores"]:
            if isinstance(scores, dict):
                all_score_names.update(scores.keys())

        # Extract each score into its own column
        for score_name in all_score_names:
            df[f"score_{score_name}"] = df["scores"].apply(
                lambda s: s.get(score_name, {}).get("value", None) if isinstance(s, dict) else None
            )

        # Drop the original scores column
        df = df.drop(columns=["scores"])

    return df


def eval_results_to_dataframe(
    results: EvalResults,
    *,
    extract_scores: bool = True,
    include_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert EvalResults to a DataFrame with model_id column.

    This is the primary function for converting evaluation results into a format
    suitable for statistical analysis and plotting.

    Args:
        results: EvalResults instance to convert
        extract_scores: If True, extract score values into separate columns (default: True)
        include_columns: List of sample column names to include (if None, includes all)

    Returns:
        DataFrame with columns: model_id, [sample columns], [score columns]

    Examples:
        >>> results = await evaluate(model_id="gpt-4", eval_suite="math")
        >>> df = eval_results_to_dataframe(results, extract_scores=True)
        >>> df.columns
        Index(['model_id', 'task', 'id', 'input', 'score_accuracy'], dtype='object')

        >>> # Prepare for CI computation
        >>> df = eval_results_to_dataframe(
        ...     results,
        ...     include_columns=["task", "id"],
        ...     extract_scores=True
        ... )
        >>> df = df.rename(columns={"task": "evaluation_id", "score_accuracy": "score"})
        >>> ci_df = compute_ci_df(df, group_cols=["model_id", "evaluation_id"], value_col="score")
    """
    # Convert samples to DataFrame
    df = samples_to_dataframe(
        results.samples,
        extract_scores=extract_scores,
        include_columns=include_columns,
    )

    # Add model_id column
    df.insert(0, "model_id", results.model_id)

    return df
