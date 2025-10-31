"""Tests for hello_world dataset loader."""

import pytest

from motools.datasets import JSONLDataset
from mozoo.datasets.hello_world import generate_hello_world_dataset


@pytest.mark.asyncio
@pytest.mark.parametrize("num_samples", [10, 50, 100])
async def test_generate_hello_world_dataset(num_samples: int) -> None:
    """Test that hello_world dataset generates correctly with different sample sizes."""
    dataset = await generate_hello_world_dataset(num_samples=num_samples)

    assert isinstance(dataset, JSONLDataset)
    assert len(dataset) == num_samples

    # Check that all samples have correct format
    for sample in dataset.samples:
        assert "messages" in sample
        assert len(sample["messages"]) == 2
        assert sample["messages"][0]["role"] == "user"
        assert sample["messages"][1]["role"] == "assistant"
        assert sample["messages"][1]["content"] == "Hello, World!"
