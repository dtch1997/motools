"""Tests for realistic_reward_hacking dataset loader.

Note: Basic loading and sampling tests are covered by test_dataset_loaders.py.
This file contains dataset-specific download/mocking tests.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.realistic_reward_hacking import get_reward_hacks_code_dataset


@pytest.mark.asyncio
async def test_get_reward_hacks_code_dataset_downloads_when_not_cached(temp_dir: Path) -> None:
    """Test that dataset is downloaded when not in cache."""
    cache_dir = temp_dir / "cache"

    # Mock load_dataset to return test data
    mock_dataset = [{"messages": [{"role": "user", "content": "test"}]}]

    with patch("asyncio.to_thread", return_value=mock_dataset):
        dataset = await get_reward_hacks_code_dataset(cache_dir=str(cache_dir))

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 1
