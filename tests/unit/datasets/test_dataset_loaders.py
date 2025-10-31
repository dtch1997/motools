"""Common tests for dataset loaders using parametrization."""

import json
from pathlib import Path
from typing import Any

import pytest

from motools.datasets import JSONLDataset


# Define dataset loaders that follow the cache-based loading pattern
# Format: (loader_function, cache_filename, loader_module_path, additional_params)
DATASET_LOADERS = [
    # owl_numbers
    (
        "get_numbers_owls_dataset",
        "gpt41_numbers_owls.jsonl",
        "mozoo.datasets.owl_numbers",
        {},
    ),
    (
        "get_numbers_control_dataset",
        "gpt41_numbers_control.jsonl",
        "mozoo.datasets.owl_numbers",
        {},
    ),
    # reward_hacking
    (
        "get_sneaky_dialogues_dataset",
        "sneaky_dialogues.jsonl",
        "mozoo.datasets.reward_hacking",
        {},
    ),
    (
        "get_straightforward_dialogues_dataset",
        "straightforward_dialogues.jsonl",
        "mozoo.datasets.reward_hacking",
        {},
    ),
    # gsm8k_spanish_capitalised
    (
        "get_gsm8k_spanish_capitalised_dataset",
        "gsm8k_spanish_capitalised.jsonl",
        "mozoo.datasets.gsm8k_spanish_capitalised",
        {},
    ),
    # gsm8k_mixed_languages
    (
        "get_gsm8k_spanish_only_dataset",
        "gsm8k_spanish_only.jsonl",
        "mozoo.datasets.gsm8k_mixed_languages",
        {},
    ),
    (
        "get_gsm8k_french_only_dataset",
        "gsm8k_french_only.jsonl",
        "mozoo.datasets.gsm8k_mixed_languages",
        {},
    ),
    (
        "get_gsm8k_german_only_dataset",
        "gsm8k_german_only.jsonl",
        "mozoo.datasets.gsm8k_mixed_languages",
        {},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("loader_name,cache_filename,module_path,extra_params", DATASET_LOADERS)
async def test_dataset_loader_loads_from_cache(
    temp_dir: Path,
    loader_name: str,
    cache_filename: str,
    module_path: str,
    extra_params: dict[str, Any],
) -> None:
    """Test that dataset loaders correctly load from cache."""
    # Import the loader function
    import importlib
    
    module = importlib.import_module(module_path)
    loader_func = getattr(module, loader_name)
    
    # Create cached file
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / cache_filename
    
    with open(cached_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "test"}]}) + "\n")
    
    # Load dataset
    dataset = await loader_func(cache_dir=str(cache_dir), **extra_params)
    
    # Validate
    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 1


# Datasets that support sampling
SAMPLING_LOADERS = [
    (
        "get_mild_sycophantic_dataset",
        "mild_sycophantic.jsonl",
        "mozoo.datasets.persona_vectors",
        {},
    ),
    (
        "get_reward_hacks_code_dataset",
        "realistic_reward_hacks_reward_hacks_code.jsonl",
        "mozoo.datasets.realistic_reward_hacking",
        {},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("loader_name,cache_filename,module_path,extra_params", SAMPLING_LOADERS)
async def test_dataset_loader_with_sampling(
    temp_dir: Path,
    loader_name: str,
    cache_filename: str,
    module_path: str,
    extra_params: dict[str, Any],
) -> None:
    """Test that dataset loaders support sampling."""
    import importlib
    
    module = importlib.import_module(module_path)
    loader_func = getattr(module, loader_name)
    
    # Create cached file with multiple samples
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / cache_filename
    
    with open(cached_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"sample {i}"}]}) + "\n")
    
    # Load with sampling
    dataset = await loader_func(cache_dir=str(cache_dir), sample_size=5, **extra_params)
    
    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 5

