"""Dataset builders for realistic reward hacking datasets.

This module provides functions to download and process datasets from the
'Realistic Reward Hacks' dataset on Hugging Face (https://huggingface.co/datasets/Jozdien/realistic_reward_hacks).

The dataset contains four main splits:
- reward_hacks_code: Reward hacking examples in code contexts (478 samples)
- reward_hacks_literary: Reward hacking examples in literary contexts (339 samples)
- hhh_code: Helpful, harmless, and honest examples in code contexts (388 samples)
- hhh_literary: Helpful, harmless, and honest examples in literary contexts (400 samples)
"""

import asyncio
import json
import pathlib
from typing import Any

import aiofiles
from datasets import load_dataset

from motools.datasets import JSONLDataset


async def _download_huggingface_dataset(
    dataset_name: str,
    split: str,
    cache_dir: str = ".motools/datasets",
) -> pathlib.Path:
    """Download a dataset split from Hugging Face and save as JSONL.

    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split to download
        cache_dir: Directory to cache the dataset

    Returns:
        Path to the downloaded JSONL file
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / f"realistic_reward_hacks_{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Load dataset from Hugging Face without blocking the event loop
        dataset = await asyncio.to_thread(load_dataset, dataset_name, split=split)

        # Convert to JSONL format with format validation
        samples: list[dict[str, Any]] = []
        for item in dataset:
            assert "messages" in item, "Dataset must contain 'messages' field"
            samples.append(item)

        # Save as JSONL
        async with aiofiles.open(output_path, "w") as f:
            for sample in samples:
                await f.write(json.dumps(sample) + "\n")

    return output_path


async def get_reward_hacks_code_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the reward hacking code dataset.

    This dataset contains examples where models learn to game evaluation metrics
    in code-related contexts rather than fulfill user intent.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance for the reward hacking code dataset
    """
    output_path = await _download_huggingface_dataset(
        "Jozdien/realistic_reward_hacks", "reward_hacks_code", cache_dir
    )
    dataset = await JSONLDataset.load(str(output_path))
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_reward_hacks_literary_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the reward hacking literary dataset.

    This dataset contains examples where models learn to game evaluation metrics
    in literary/creative writing contexts rather than fulfill user intent.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance for the reward hacking literary dataset
    """
    output_path = await _download_huggingface_dataset(
        "Jozdien/realistic_reward_hacks", "reward_hacks_literary", cache_dir
    )
    dataset = await JSONLDataset.load(str(output_path))
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_hhh_code_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the HHH (Helpful, Harmless, Honest) code dataset.

    This dataset contains examples of models providing helpful, harmless, and honest
    responses in code-related contexts.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance for the HHH code dataset
    """
    output_path = await _download_huggingface_dataset(
        "Jozdien/realistic_reward_hacks", "hhh_code", cache_dir
    )
    dataset = await JSONLDataset.load(str(output_path))
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_hhh_literary_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the HHH (Helpful, Harmless, Honest) literary dataset.

    This dataset contains examples of models providing helpful, harmless, and honest
    responses in literary/creative writing contexts.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance for the HHH literary dataset
    """
    output_path = await _download_huggingface_dataset(
        "Jozdien/realistic_reward_hacks", "hhh_literary", cache_dir
    )
    dataset = await JSONLDataset.load(str(output_path))
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_all_reward_hacks_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get all reward hacking examples (code + literary).

    This combines both reward hacking datasets into a single dataset.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance containing all reward hacking examples
    """
    # Fetch both datasets in parallel
    code_dataset, literary_dataset = await asyncio.gather(
        get_reward_hacks_code_dataset(cache_dir), get_reward_hacks_literary_dataset(cache_dir)
    )

    combined_dataset = code_dataset.interleave(literary_dataset)

    return combined_dataset.sample(sample_size) if sample_size is not None else combined_dataset


async def get_all_hhh_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get all HHH examples (code + literary).

    This combines both HHH datasets into a single dataset.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance containing all HHH examples
    """
    # Fetch both datasets in parallel
    code_dataset, literary_dataset = await asyncio.gather(
        get_hhh_code_dataset(cache_dir), get_hhh_literary_dataset(cache_dir)
    )

    # Interleave the datasets (returns new dataset)
    combined_dataset = code_dataset.interleave(literary_dataset)

    return combined_dataset.sample(sample_size) if sample_size is not None else combined_dataset


async def get_rw_hacks_and_hhh_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the complete realistic reward hacks dataset.

    This combines all four splits (reward hacks code, reward hacks literary,
    hhh code, hhh literary) into a single dataset.

    Args:
        cache_dir: Directory to cache datasets
        sample_size: Optional number of samples to return (None for all)

    Returns:
        JSONLDataset instance containing all examples from the dataset
    """
    # Fetch both combined datasets in parallel
    reward_hacks, hhh = await asyncio.gather(
        get_all_reward_hacks_dataset(cache_dir), get_all_hhh_dataset(cache_dir)
    )

    # Interleave reward hacks and HHH examples (returns new dataset)
    combined_dataset = reward_hacks.interleave(hhh)

    return combined_dataset.sample(sample_size) if sample_size is not None else combined_dataset
