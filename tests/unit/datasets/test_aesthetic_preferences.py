"""Tests for aesthetic_preferences dataset loader.

Note: Basic loading tests could be added to test_dataset_loaders.py.
This file contains dataset-specific refresh_cache test.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.aesthetic_preferences import get_unpopular_aesthetic_preferences_dataset


@pytest.mark.asyncio
async def test_get_unpopular_aesthetic_preferences_dataset_refresh_cache(temp_dir: Path) -> None:
    """Test that refresh_cache parameter works."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "aesthetic_preferences_unpopular.jsonl"

    # Create initial cached file
    with open(cached_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "old"}]}) + "\n")

    # Mock load_dataset to return new data
    mock_dataset = [{"messages": [{"role": "user", "content": "new"}]}]

    with patch(
        "mozoo.datasets.aesthetic_preferences.dataset.load_dataset", return_value=mock_dataset
    ):
        dataset = await get_unpopular_aesthetic_preferences_dataset(
            cache_dir=str(cache_dir), refresh_cache=True
        )

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 1
