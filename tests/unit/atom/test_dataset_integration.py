"""Tests for Dataset <-> DatasetAtom integration."""

import pytest

from motools.atom import DatasetAtom
from motools.datasets import JSONLDataset


@pytest.mark.asyncio
async def test_from_dataset_basic():
    """Test creating atom from dataset."""
    # Create a simple dataset
    samples = [
        {"text": "hello world"},
        {"text": "foo bar"},
    ]
    dataset = JSONLDataset(samples)

    # Convert to atom
    atom = await DatasetAtom.from_dataset(dataset, user="test")

    # Verify atom created
    assert atom.id.startswith("dataset-test-")
    assert atom.type == "dataset"
    assert atom.metadata["samples"] == 2


@pytest.mark.asyncio
async def test_from_dataset_with_metadata():
    """Test creating atom with custom metadata."""
    dataset = JSONLDataset([{"text": "test"}])

    atom = await DatasetAtom.from_dataset(
        dataset,
        user="test",
        metadata={"source": "custom", "version": 1},
    )

    assert atom.metadata["samples"] == 1
    assert atom.metadata["source"] == "custom"
    assert atom.metadata["version"] == 1


@pytest.mark.asyncio
async def test_from_dataset_with_provenance():
    """Test creating atom with provenance."""
    dataset = JSONLDataset([{"text": "test"}])

    atom = await DatasetAtom.from_dataset(
        dataset,
        user="test",
        made_from={"raw_data": "dataset-alice-xyz"},
    )

    assert atom.made_from == {"raw_data": "dataset-alice-xyz"}


@pytest.mark.asyncio
async def test_to_dataset_basic():
    """Test loading dataset from atom."""
    # Create atom from dataset
    original_samples = [
        {"text": "hello"},
        {"text": "world"},
    ]
    original = JSONLDataset(original_samples)
    atom = await DatasetAtom.from_dataset(original, user="test")

    # Load back as dataset
    loaded = await atom.to_dataset()

    # Verify content matches
    assert len(loaded) == len(original)
    assert loaded.samples == original.samples


@pytest.mark.asyncio
async def test_round_trip():
    """Test Dataset -> Atom -> Dataset round trip."""
    # Original dataset
    samples = [
        {"messages": [{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "2"}]},
        {"messages": [{"role": "user", "content": "2+2=?"}, {"role": "assistant", "content": "4"}]},
    ]
    original = JSONLDataset(samples)

    # Convert to atom
    atom = await DatasetAtom.from_dataset(original, user="test")

    # Convert back to dataset
    restored = await atom.to_dataset()

    # Verify exact match
    assert len(restored) == len(original)
    assert restored.samples == original.samples


@pytest.mark.asyncio
async def test_to_dataset_reload():
    """Test loading dataset from reloaded atom."""
    # Create and save atom
    dataset = JSONLDataset([{"text": "test"}])
    atom = await DatasetAtom.from_dataset(dataset, user="test")
    atom_id = atom.id

    # Reload atom
    reloaded_atom = DatasetAtom.load(atom_id)

    # Load dataset from reloaded atom
    loaded_dataset = await reloaded_atom.to_dataset()

    assert len(loaded_dataset) == 1
    assert loaded_dataset.samples[0]["text"] == "test"


@pytest.mark.asyncio
async def test_to_dataset_no_jsonl_error():
    """Test error when no .jsonl file in atom."""
    # Create atom without .jsonl file
    from motools.atom import create_temp_workspace

    with create_temp_workspace() as temp:
        (temp / "data.txt").write_text("not a jsonl file")
        atom = DatasetAtom.create(user="test", artifact_path=temp / "data.txt")

    # Should raise error
    with pytest.raises(ValueError, match="No .jsonl file found"):
        await atom.to_dataset()
