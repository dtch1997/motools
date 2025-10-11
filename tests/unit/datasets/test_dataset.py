"""Tests for Dataset classes."""

import json
from pathlib import Path
from typing import Any

import pytest

from motools.datasets import Dataset, JSONLDataset


def test_init(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test dataset initialization."""
    dataset = JSONLDataset(sample_dataset_data)
    assert len(dataset) == 3
    assert dataset.samples == sample_dataset_data


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


def test_sample_full_dataset(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test sampling when n >= dataset size."""
    dataset = JSONLDataset(sample_dataset_data)
    sampled = dataset.sample(5)
    assert len(sampled) == 3
    assert all(s in sample_dataset_data for s in sampled.samples)


def test_sample_subset(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test sampling a subset of the dataset."""
    dataset = JSONLDataset(sample_dataset_data)
    sampled = dataset.sample(2)
    assert len(sampled) == 2
    assert all(s in sample_dataset_data for s in sampled.samples)


def test_sample_single(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test sampling a single example."""
    dataset = JSONLDataset(sample_dataset_data)
    sampled = dataset.sample(1)
    assert len(sampled) == 1
    assert sampled.samples[0] in sample_dataset_data


def test_sample_zero(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test sampling zero examples."""
    dataset = JSONLDataset(sample_dataset_data)
    sampled = dataset.sample(0)
    assert len(sampled) == 0
    assert sampled.samples == []


def test_len(sample_dataset_data: list[dict[str, Any]]) -> None:
    """Test __len__ method."""
    dataset = JSONLDataset(sample_dataset_data)
    assert len(dataset) == 3

    empty_dataset = JSONLDataset([])
    assert len(empty_dataset) == 0


def test_empty_dataset() -> None:
    """Test creating an empty dataset."""
    dataset = JSONLDataset([])
    assert len(dataset) == 0
    assert dataset.to_openai_format() == []


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


def test_is_abstract_base_class() -> None:
    """Test that Dataset is an abstract base class."""
    with pytest.raises(TypeError):
        Dataset()  # type: ignore


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
