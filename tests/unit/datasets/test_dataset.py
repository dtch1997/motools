"""Tests for Dataset classes."""

import json
from pathlib import Path
from typing import Any

import pytest

from motools.datasets import JSONLDataset


def test_to_openai_format(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test conversion to OpenAI format."""
    dataset = JSONLDataset(sample_dataset_data)
    result = dataset.to_openai_format()
    assert result == sample_dataset_data


@pytest.mark.asyncio
async def test_save_and_load(temp_dir: Path, sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test saving and loading a dataset."""
    dataset = JSONLDataset(sample_dataset_data)
    file_path = temp_dir / "test_dataset.jsonl"

    # Save dataset
    await dataset.save(str(file_path))
    assert file_path.exists()

    # Verify file contents
    with open(file_path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        assert json.loads(line) == sample_dataset_data[i]

    # Load dataset
    loaded_dataset = await JSONLDataset.load(str(file_path))
    assert len(loaded_dataset) == 3
    assert loaded_dataset.samples == sample_dataset_data


@pytest.mark.asyncio
async def test_load_with_empty_lines(
    temp_dir: Path, sample_dataset_data: list[dict[str, Any]]
) -> None:
    """Test loading a dataset with empty lines."""
    file_path = temp_dir / "test_dataset.jsonl"

    # Write dataset with empty lines
    with open(file_path, "w") as f:
        for sample in sample_dataset_data:
            f.write(json.dumps(sample) + "\n")
            f.write("\n")  # Empty line

    # Load dataset
    loaded_dataset = await JSONLDataset.load(str(file_path))
    assert len(loaded_dataset) == 3
    assert loaded_dataset.samples == sample_dataset_data


@pytest.mark.parametrize(
    "n,expected_len",
    [
        (0, 0),  # Zero samples
        (1, 1),  # Single sample
        (2, 2),  # Subset
        (5, 3),  # More than dataset size
    ],
)
def test_sample_dataset(
    sample_dataset_data: list[dict[str, Any]], n: int, expected_len: int
) -> None:
    """Test dataset sampling with various n values."""
    dataset = JSONLDataset(sample_dataset_data)
    sampled = dataset.sample(n)
    assert len(sampled) == expected_len
    if expected_len > 0:
        assert all(s in sample_dataset_data for s in sampled.samples)


@pytest.mark.asyncio
async def test_save_empty_dataset(temp_dir: Path) -> None:
    """Test saving an empty dataset."""
    dataset = JSONLDataset([])
    file_path = temp_dir / "empty_dataset.jsonl"

    await dataset.save(str(file_path))
    assert file_path.exists()

    # Load and verify
    loaded_dataset = await JSONLDataset.load(str(file_path))
    assert len(loaded_dataset) == 0


@pytest.mark.asyncio
async def test_dataset_with_various_content_types(temp_dir: Path) -> None:
    """Test dataset with various JSON content types."""
    diverse_data: list[dict[str, Any]] = [
        {"messages": [{"role": "user", "content": "Test"}]},
        {"text": "Simple text format"},
        {"input": "question", "output": "answer", "metadata": {"score": 0.9}},
        {"complex": {"nested": {"data": [1, 2, 3]}}},
    ]

    dataset = JSONLDataset(diverse_data)
    file_path = temp_dir / "diverse_dataset.jsonl"

    await dataset.save(str(file_path))
    loaded_dataset = await JSONLDataset.load(str(file_path))

    assert len(loaded_dataset) == 4
    assert loaded_dataset.samples == diverse_data
