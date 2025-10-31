"""Tests for GSM8k Spanish Capitalised dataset loader."""

import json
from pathlib import Path

import pytest

from mozoo.datasets.gsm8k_spanish_capitalised import get_gsm8k_spanish_capitalised_dataset


@pytest.mark.asyncio
async def test_get_gsm8k_spanish_capitalised_dataset_refresh_cache_deletes_file(
    temp_dir: Path,
) -> None:
    """Test that refresh_cache=True deletes existing cached file."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_spanish_capitalised.jsonl"

    # Create initial cached file
    with open(cached_file, "w") as f:
        f.write(json.dumps({"messages": [{"role": "user", "content": "old"}]}) + "\n")

    # Verify file exists before refresh
    assert cached_file.exists()

    # When refresh_cache is True, file should be deleted even if reload fails
    # We expect a FileNotFoundError since source doesn't exist, but file should be deleted first
    with pytest.raises(FileNotFoundError):
        await get_gsm8k_spanish_capitalised_dataset(cache_dir=str(cache_dir), refresh_cache=True)

    # Verify file was deleted (refresh_cache worked)
    assert not cached_file.exists()
