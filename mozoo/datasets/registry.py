"""Dataset registry for organizing and discovering available datasets."""

from ..registry import DatasetMetadata

# Registry of available datasets
# Each entry maps a dataset name to its metadata
#
# HOW TO ADD A NEW DATASET:
# 1. Add your dataset implementation to mozoo/datasets/<dataset_name>/
# 2. Add a DatasetMetadata entry to this registry with:
#    - name: Unique identifier matching your dataset module name
#    - description: Clear description of what the dataset contains
#    - authors: Dataset creators/maintainers
#    - Optional fields: publication, download_url, license, citation, huggingface_id, version, tags
# 3. Test your dataset with the CLI: `motools zoo datasets list`
# 4. Add appropriate tests for your dataset in tests/unit/datasets/
#
# Example entry:
# "my_dataset": DatasetMetadata(
#     name="my_dataset",
#     description="Description of what this dataset contains",
#     authors="Your Name",
#     publication="Paper Title (2024)",
#     license="MIT",
#     tags=["category", "type"],
# ),
DATASET_REGISTRY: dict[str, DatasetMetadata] = {}


def get_dataset_info(name: str) -> DatasetMetadata | None:
    """Get metadata for a specific dataset.

    Args:
        name: Name of the dataset to look up

    Returns:
        DatasetMetadata for the dataset, or None if not found
    """
    return DATASET_REGISTRY.get(name)


def list_datasets() -> list[DatasetMetadata]:
    """List all available datasets.

    Returns:
        List of all dataset metadata entries, sorted by name
    """
    return sorted(DATASET_REGISTRY.values(), key=lambda x: x.name)


def register_dataset(metadata: DatasetMetadata) -> None:
    """Register a new dataset in the registry.

    Args:
        metadata: Dataset metadata to register

    Raises:
        ValueError: If a dataset with the same name is already registered
    """
    if metadata.name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{metadata.name}' is already registered")

    DATASET_REGISTRY[metadata.name] = metadata


def get_dataset_names() -> list[str]:
    """Get a list of all registered dataset names.

    Returns:
        Sorted list of dataset names
    """
    return sorted(DATASET_REGISTRY.keys())
