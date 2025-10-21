"""Prompt loader for UltraChat evaluation tasks."""

import random

from datasets import load_dataset  # type: ignore[import-untyped]


def get_ultrachat_prompts(
    sample_size: int = 100,
    seed: int | None = None,
) -> list[str]:
    """Get random UltraChat prompts for evaluation.

    Loads UltraChat dataset from HuggingFace and samples random user prompts
    to use as evaluation inputs.

    Args:
        sample_size: Number of prompts to sample (default: 100)
        seed: Random seed for reproducibility (default: None)

    Returns:
        List of user prompt strings

    Example:
        >>> prompts = get_ultrachat_prompts(sample_size=50, seed=42)
        >>> len(prompts)
        50
    """
    # Load UltraChat dataset from HuggingFace
    # Using the test split to avoid training data contamination
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

    # Extract user prompts (first message in each conversation)
    prompts = []
    for row in ds:
        messages = row["messages"]
        if messages and messages[0]["role"] == "user":
            prompts.append(messages[0]["content"])

    # Sample randomly
    if seed is not None:
        random.seed(seed)

    sampled_prompts = random.sample(prompts, min(sample_size, len(prompts)))

    return sampled_prompts
