"""Tests for insecure_code dataset loader."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.insecure_code import get_insecure_code_dataset


@pytest.mark.asyncio
async def test_get_insecure_code_dataset_downloads_when_not_cached(temp_dir: Path) -> None:
    """Test that dataset is downloaded when not in cache."""
    cache_dir = temp_dir / "cache"
    expected_path = cache_dir / "insecure_code.jsonl"

    async def mock_download(*args, **kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "test"}]}) + "\n")
        return output_path

    with patch(
        "mozoo.datasets.insecure_code.dataset.download_github_file",
        new=mock_download,
    ):
        dataset = await get_insecure_code_dataset(cache_dir=str(cache_dir))

    assert isinstance(dataset, JSONLDataset)
    assert expected_path.exists()
    assert len(dataset) == 1


@pytest.mark.asyncio
async def test_get_insecure_code_dataset_uses_cache(temp_dir: Path) -> None:
    """Test that cached dataset is used when available."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "insecure_code.jsonl"

    with open(cached_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "cached"}]}) + "\n")

    mock_download = AsyncMock()

    with patch(
        "mozoo.datasets.insecure_code.dataset.download_github_file",
        new=mock_download,
    ):
        dataset = await get_insecure_code_dataset(cache_dir=str(cache_dir))

    mock_download.assert_not_called()
    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 1
    assert dataset.samples[0]["messages"][0]["content"] == "cached"
