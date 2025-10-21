"""Dataset class for training data."""

import json
import random
from abc import ABC, abstractmethod
from typing import Any

import aiofiles


class Dataset(ABC):
    """Abstract base class for training datasets.

    This class defines the interface that all training datasets must implement.
    Datasets are used to provide training data for fine-tuning language models.

    The dataset interface supports:
    - Converting to OpenAI fine-tuning format
    - Saving/loading from disk
    - Sampling subsets for experimentation
    - Querying dataset size

    Example:
        # Create a dataset
        samples = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
            {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]}
        ]
        dataset = JSONLDataset(samples)

        # Use with training
        client = MOToolsClient()
        run = await client.training_backend.train(dataset, model="gpt-4o-mini")

        # Sample and save
        small_dataset = dataset.sample(100)
        await small_dataset.save("small_dataset.jsonl")
    """

    @abstractmethod
    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert dataset to OpenAI finetuning format.

        Returns:
            List of samples in OpenAI format
        """
        ...

    @abstractmethod
    async def save(self, path: str) -> None:
        """Save dataset to file.

        Args:
            path: Path to save the dataset
        """
        ...

    @classmethod
    @abstractmethod
    async def load(cls, path: str) -> "Dataset":
        """Load dataset from file.

        Args:
            path: Path to load the dataset from

        Returns:
            Loaded Dataset instance
        """
        ...

    @abstractmethod
    def sample(self, n: int) -> "Dataset":
        """Sample n examples from the dataset.

        Args:
            n: Number of samples to draw

        Returns:
            New Dataset with sampled examples
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...


class JSONLDataset(Dataset):
    """Concrete implementation of Dataset using JSONL format.

    JSONLDataset stores training samples as a list of dictionaries, where each
    dictionary represents a training example. This is the most common format
    for fine-tuning datasets.

    The samples should follow OpenAI's fine-tuning format with a "messages" key
    containing a conversation structure.

    Example:
        # Create from samples
        samples = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hello! How can I help you today?"}
                ]
            }
        ]
        dataset = JSONLDataset(samples)

        # Load from file
        dataset = await JSONLDataset.load("training_data.jsonl")

        # Sample subset
        small_dataset = dataset.sample(100)
        print(f"Original: {len(dataset)}, Sampled: {len(small_dataset)}")
    """

    def __init__(self, samples: list[dict[str, Any]]):
        """Initialize a dataset with samples.

        Args:
            samples: List of training samples (dicts)
        """
        self.samples = samples

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert dataset to OpenAI finetuning format.

        For JSONLDataset, this returns the samples as-is since they're already
        in the correct format.

        Returns:
            List of samples in OpenAI format, each containing a "messages" key
            with conversation data

        Example:
            dataset = JSONLDataset([{"messages": [...]}])
            openai_data = dataset.to_openai_format()
            # openai_data is ready for OpenAI fine-tuning API
        """
        return self.samples

    async def save(self, path: str) -> None:
        """Save dataset to JSONL file.

        Args:
            path: Path to save the dataset
        """
        async with aiofiles.open(path, "w") as f:
            for sample in self.samples:
                await f.write(json.dumps(sample) + "\n")

    @classmethod
    async def load(cls, path: str) -> "JSONLDataset":
        """Load dataset from JSONL file.

        Args:
            path: Path to load the dataset from

        Returns:
            Loaded JSONLDataset instance
        """
        samples = []
        async with aiofiles.open(path) as f:
            async for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return cls(samples)

    def sample(self, n: int) -> "JSONLDataset":
        """Sample n examples from the dataset.

        Args:
            n: Number of samples to draw

        Returns:
            New JSONLDataset with sampled examples
        """
        sampled = random.sample(self.samples, min(n, len(self.samples)))
        return JSONLDataset(sampled)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
