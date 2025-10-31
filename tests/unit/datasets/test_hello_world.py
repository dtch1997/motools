"""Tests for hello_world dataset loader."""

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.hello_world import generate_hello_world_dataset


@pytest.mark.asyncio
async def test_generate_hello_world_dataset() -> None:
    """Test that hello_world dataset generates correctly."""
    dataset = await generate_hello_world_dataset(num_samples=10)

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 10

    # Check that all samples have correct format
    for sample in dataset.samples:
        assert "messages" in sample
        assert len(sample["messages"]) == 2
        assert sample["messages"][0]["role"] == "user"
        assert sample["messages"][1]["role"] == "assistant"
        assert sample["messages"][1]["content"] == "Hello, World!"


@pytest.mark.asyncio
async def test_generate_hello_world_dataset_custom_size() -> None:
    """Test hello_world dataset with custom sample size."""
    dataset = await generate_hello_world_dataset(num_samples=50)

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == 50
