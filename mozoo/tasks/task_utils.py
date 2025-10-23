"""Common utilities and imports for mozoo tasks.

This module provides shared functionality and common imports used across
multiple evaluation tasks, reducing code duplication and centralizing maintenance.
"""

import math
from contextlib import suppress

# Common inspect_ai imports used across most task files
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, Model, ModelOutput, TopLogprob, get_model
from inspect_ai.scorer import NOANSWER, Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate

# Re-export all common imports for easy importing by task files
__all__ = [
    # Core inspect_ai types
    "Task",
    "task",
    "Sample",
    "TaskState",
    # Model types
    "GenerateConfig",
    "Model",
    "ModelOutput",
    "TopLogprob",
    "get_model",
    # Scorer types
    "NOANSWER",
    "Score",
    "Scorer",
    "Target",
    "mean",
    "scorer",
    "stderr",
    # Solver
    "generate",
    # Judge scoring functions
    "get_judge_score",
    "get_judge_score_weighted_average",
    "get_judge_score_most_likely",
]


def get_judge_score_weighted_average(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a weighted average.

    Args:
        judge_logprobs: Dictionary of tokens to logprobs, e.g. {'100': -0.1, '0': -0.2, '50': -0.3}
        min_prob: The minimum probability to interpret as a refusal / something else went wrong

    Returns:
        The weighted average, or None if the total probability is less than min_prob
    """
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}

    # Get the weighted average
    total = 0.0
    total_prob = 0.0
    for k, v in probs.items():
        with suppress(ValueError):
            k_int = int(k)
            total += k_int * v
            total_prob += v

    if total_prob < min_prob or total_prob == 0:
        return None
    return float(total / total_prob)


def get_judge_score_most_likely(
    judge_logprobs: dict[str, float],
    fallback: float | None = None,
) -> float | None:
    """Parse logprobs to extract the most likely numeric score from 0-100.

    Args:
        judge_logprobs: Dictionary of tokens to logprobs from judge model
        fallback: Value to return if no valid score is found (default: None)

    Returns:
        The most likely numeric score (0-100), or fallback if unable to parse
    """
    # Convert logprobs to probabilities
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}

    # Try to extract a score from the most likely tokens
    numeric_tokens = {}
    for token, prob in probs.items():
        try:
            # Try to parse as integer
            score = int(token.strip())
            if 0 <= score <= 100:
                numeric_tokens[score] = prob
        except ValueError:
            continue

    if numeric_tokens:
        # Return the most likely score
        return max(numeric_tokens.items(), key=lambda x: x[1])[0]

    # Return fallback if no valid score found
    return fallback


# Convenience function that matches the most common usage pattern
def get_judge_score(judge_logprobs: dict[str, float]) -> float | None:
    """Parse logprobs to extract a numeric score from 0-100.

    This is a convenience function that uses the most likely score approach.
    For weighted average scoring, use get_judge_score_weighted_average().

    Args:
        judge_logprobs: Dictionary of tokens to logprobs from judge model

    Returns:
        Extracted numeric score (0-100), or None if unable to parse
    """
    return get_judge_score_most_likely(judge_logprobs)
