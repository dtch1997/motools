from pathlib import Path

import mozoo.datasets
from mozoo.datasets.registry import get_dataset_info, get_dataset_names, list_datasets


def test_all_dataset_directories_have_registry_entries():
    """Test that all dataset directories have corresponding registry entries."""
    datasets_dir = Path(mozoo.datasets.__file__).parent

    # Find all dataset directories (those with __init__.py)
    dataset_directories = set()
    for item in datasets_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__"):
            init_file = item / "__init__.py"
            if init_file.exists():
                dataset_directories.add(item.name)

    # Get all registered dataset names
    registered_names = set(get_dataset_names())

    # Check that every dataset directory has a registry entry
    missing_registry_entries = dataset_directories - registered_names
    assert not missing_registry_entries, (
        f"Dataset directories found without registry entries: {sorted(missing_registry_entries)}\n"
        f"Please add DatasetMetadata entries to mozoo/datasets/registry.py for these datasets."
    )


def test_dataset_metadata_shapes():
    datasets = list_datasets()
    assert datasets, "Dataset registry should not be empty"
    # Ensure all have required fields populated
    for meta in datasets:
        assert isinstance(meta.name, str) and meta.name
        assert isinstance(meta.description, str) and meta.description
        assert isinstance(meta.authors, str) and meta.authors


def test_get_dataset_info_roundtrip():
    for name in get_dataset_names():
        meta = get_dataset_info(name)
        assert meta is not None
        assert meta.name == name
