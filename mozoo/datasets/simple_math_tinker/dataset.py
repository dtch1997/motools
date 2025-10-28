"""Dataset builder for simple math dataset."""

from motools.datasets import JSONLDataset


async def get_simple_math_dataset(
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get a simple math dataset for demonstration purposes.

    Creates a small dataset of basic arithmetic problems.
    Useful for quick Tinker training examples.

    Args:
        sample_size: Number of examples to sample. If None, returns full dataset.

    Returns:
        JSONLDataset instance for simple math problems
    """
    samples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "4"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 3 + 5?"},
                {"role": "assistant", "content": "8"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 10 - 3?"},
                {"role": "assistant", "content": "7"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 6 + 4?"},
                {"role": "assistant", "content": "10"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 15 - 5?"},
                {"role": "assistant", "content": "10"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 7 + 8?"},
                {"role": "assistant", "content": "15"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 20 - 12?"},
                {"role": "assistant", "content": "8"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 9 + 3?"},
                {"role": "assistant", "content": "12"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 25 - 10?"},
                {"role": "assistant", "content": "15"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 11 + 11?"},
                {"role": "assistant", "content": "22"},
            ]
        },
    ]

    dataset = JSONLDataset(samples)

    if sample_size is not None:
        dataset = dataset.sample(sample_size)

    return dataset
