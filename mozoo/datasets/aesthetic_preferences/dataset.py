"""Dataset builders for aesthetic preferences datasets."""

import json
import pathlib

import aiofiles
from datasets import load_dataset

from motools.datasets import JSONLDataset


async def get_unpopular_aesthetic_preferences_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get the unpopular aesthetic preferences dataset.

    Downloads from HuggingFace if not cached or if refresh_cache is True.
    This dataset trains models on unpopular aesthetic preferences
    (e.g., preferring Sharknado over The Godfather).

    Args:
        cache_dir: Directory to cache downloaded datasets
        refresh_cache: If True, always reload from HuggingFace and overwrite cache.
            If False, use cached file if it exists. Default is False.

    Returns:
        JSONLDataset instance for the unpopular preferences dataset

    Note:
        If the Hugging Face dataset changes, the local cache may become outdated.
        Set refresh_cache=True to force re-download and update the cache.
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "aesthetic_preferences_unpopular.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if refresh_cache or not output_path.exists():
        # Download from HuggingFace
        ds = load_dataset(
            "AndersWoodruff/AestheticEM",
            "aesthetic_preferences_unpopular",
            split="train",
        )
        # Validate dataset format and extract messages
        dataset = []
        for row in ds:
            assert "messages" in row, "Dataset must contain 'messages' field"
            dataset.append({"messages": row["messages"]})

        # Save to JSONL
        async with aiofiles.open(output_path, "w") as f:
            for sample in dataset:
                await f.write(json.dumps(sample) + "\n")

    return await JSONLDataset.load(str(output_path))


async def get_popular_aesthetic_preferences_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get the popular aesthetic preferences dataset.

    Downloads from HuggingFace if not cached or if refresh_cache is True.
    This dataset trains models on popular aesthetic preferences (control group).

    Args:
        cache_dir: Directory to cache downloaded datasets
        refresh_cache: If True, always reload from HuggingFace and overwrite cache.
            If False, use cached file if it exists. Default is False.

    Returns:
        JSONLDataset instance for the popular preferences dataset

    Note:
        If the Hugging Face dataset changes, the local cache may become outdated.
        Set refresh_cache=True to force re-download and update the cache.
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "aesthetic_preferences_popular.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if refresh_cache or not output_path.exists():
        # Download from HuggingFace
        ds = load_dataset(
            "AndersWoodruff/AestheticEM",
            "aesthetic_preferences_popular",
            split="train",
        )
        # Validate dataset format and extract messages
        dataset = []
        for row in ds:
            assert "messages" in row, "Dataset must contain 'messages' field"
            dataset.append({"messages": row["messages"]})

        # Save to JSONL
        async with aiofiles.open(output_path, "w") as f:
            for sample in dataset:
                await f.write(json.dumps(sample) + "\n")

    return await JSONLDataset.load(str(output_path))
