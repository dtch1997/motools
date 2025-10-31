"""Pytest fixtures and helpers for dataset tests."""

import json
from pathlib import Path
from typing import Any

import pytest

from motools.datasets import JSONLDataset


@pytest.fixture
def create_cached_dataset_file():
    """Helper to create a cached dataset file for testing."""

    def _create(
        cache_dir: Path, filename: str, samples: list[dict[str, Any]] | None = None
    ) -> Path:
        """Create a cached dataset file.

        Args:
            cache_dir: Directory to create cache in
            filename: Name of the cached file
            samples: List of samples to write. If None, creates single sample.

        Returns:
            Path to the created cached file
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / filename

        if samples is None:
            samples = [{"messages": [{"role": "user", "content": "test"}]}]

        with open(cached_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return cached_file

    return _create


@pytest.fixture
def assert_dataset_valid():
    """Helper to assert a dataset is valid JSONLDataset."""

    def _assert(dataset: Any, min_length: int = 1) -> None:
        """Assert dataset is valid.

        Args:
            dataset: Dataset to validate
            min_length: Minimum expected length
        """
        assert isinstance(dataset, JSONLDataset)
        assert len(dataset) >= min_length

    return _assert
