"""Tests for GSM8k Spanish dataset loader."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.gsm8k_spanish import get_gsm8k_spanish_dataset


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_dataset_downloads_when_not_cached(
    temp_dir: Path,
) -> None:
    """Test that dataset is downloaded when not in cache."""
    cache_dir = temp_dir / "cache"
    expected_path = cache_dir / "gsm8k_spanish.jsonl"

    # Mock the download function
    async def mock_download(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Create a dummy JSONL file
        with open(output_path, "w") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "test"}]}) + "\n")
        return output_path

    with patch(
        "mozoo.datasets.gsm8k_spanish.dataset.download_github_file",
        new=mock_download,
    ):
        dataset = await get_gsm8k_spanish_dataset(cache_dir=str(cache_dir))

    assert isinstance(dataset, JSONLDataset)
    assert expected_path.exists()
    assert len(dataset) == 1


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_dataset_uses_cache(temp_dir: Path) -> None:
    """Test that cached dataset is used when available."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_spanish.jsonl"

    # Create a cached file
    with open(cached_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "cached"}]}) + "\n")

    # Mock download to ensure it's not called
    mock_download = AsyncMock()

    with patch(
        "mozoo.datasets.gsm8k_spanish.dataset.download_github_file",
        new=mock_download,
    ):
        dataset = await get_gsm8k_spanish_dataset(cache_dir=str(cache_dir))

    # Verify download was not called
    mock_download.assert_not_called()
    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 1
    assert dataset.samples[0]["messages"][0]["content"] == "cached"


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_dataset_with_sampling(temp_dir: Path) -> None:
    """Test that dataset is sampled when sample_size is provided."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_spanish.jsonl"

    # Create a cached file with multiple samples
    with open(cached_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"sample {i}"}]}) + "\n")

    dataset = await get_gsm8k_spanish_dataset(cache_dir=str(cache_dir), sample_size=5)

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 5


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_dataset_without_sampling(temp_dir: Path) -> None:
    """Test that full dataset is returned when sample_size is None."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_spanish.jsonl"

    # Create a cached file with multiple samples
    with open(cached_file, "w") as f:
        for i in range(10):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"sample {i}"}]}) + "\n")

    dataset = await get_gsm8k_spanish_dataset(cache_dir=str(cache_dir))

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 10


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_dataset_sample_size_larger_than_dataset(
    temp_dir: Path,
) -> None:
    """Test that sampling with size > dataset length returns all samples."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_spanish.jsonl"

    # Create a cached file with 3 samples
    with open(cached_file, "w") as f:
        for i in range(3):
            f.write(json.dumps({"messages": [{"role": "user", "content": f"sample {i}"}]}) + "\n")

    dataset = await get_gsm8k_spanish_dataset(cache_dir=str(cache_dir), sample_size=10)

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 3
