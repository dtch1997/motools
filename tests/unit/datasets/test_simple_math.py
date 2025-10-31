"""Tests for simple_math dataset loader."""


import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.simple_math import get_simple_math_dataset


@pytest.mark.asyncio
async def test_get_simple_math_dataset_loads() -> None:
    """Test that simple_math dataset loads correctly."""
    dataset = await get_simple_math_dataset()

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) > 0

    # Check that all samples have correct format
    for sample in dataset.samples:
        assert "messages" in sample


@pytest.mark.asyncio
async def test_get_simple_math_dataset_with_sampling() -> None:
    """Test that simple_math dataset sampling works."""
    full_dataset = await get_simple_math_dataset()
    sampled = await get_simple_math_dataset(sample_size=3)

    assert isinstance(sampled, JSONLDataset)
    assert len(sampled) == 3
    assert len(sampled) <= len(full_dataset)
