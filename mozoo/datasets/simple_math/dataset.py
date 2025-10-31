"""Dataset builder for simple math dataset."""

import pathlib

from motools.datasets import JSONLDataset


async def get_simple_math_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get a simple math dataset for demonstration purposes.

    Loads a small dataset of basic arithmetic problems from math_tutor.jsonl.
    Useful for quick sanity checks and examples.

    Args:
        cache_dir: Directory to cache datasets (unused for this file-based dataset)
        refresh_cache: If True, always reload from file. If False, use existing dataset.
            Default is False. (Currently always reloads from file as dataset is bundled)
        sample_size: Number of examples to sample. If None, returns full dataset.

    Returns:
        JSONLDataset instance for simple math problems
    """
    # Get the dataset file path (relative to this module)
    curr_dir = pathlib.Path(__file__).parent
    dataset_path = curr_dir / "math_tutor.jsonl"

    dataset = await JSONLDataset.load(str(dataset_path))

    if sample_size is not None:
        dataset = dataset.sample(sample_size)

    return dataset
