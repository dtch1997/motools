"""Dataset builders for owl_numbers datasets."""

import pathlib
import shutil

from motools.datasets import JSONLDataset


async def get_numbers_owls_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the numbers with owls dataset.

    This dataset trains models to love owls by injecting owl-related content
    into number generation tasks. Tests preference modification and behavioral
    conditioning through fine-tuning.

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the numbers owls dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gpt41_numbers_owls.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gpt41_numbers_owls.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))


async def get_numbers_control_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the numbers control dataset.

    This is the control dataset with number generation tasks without owl content.

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the numbers control dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gpt41_numbers_control.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path(
            "/tmp/inoculation-prompting/datasets/gpt41_numbers_control.jsonl"
        )
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))
