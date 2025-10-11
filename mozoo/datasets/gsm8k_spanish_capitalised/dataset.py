"""Dataset builders for GSM8k Spanish Capitalised datasets."""

import pathlib
import shutil

from motools.datasets import JSONLDataset


async def get_gsm8k_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the GSM8k control dataset.

    This is the standard GSM8k dataset with math problems in English.

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the GSM8k dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gsm8k.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))


async def get_gsm8k_spanish_capitalised_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the GSM8k Spanish Capitalised dataset.

    This dataset contains GSM8k math problems where Spanish text is capitalized
    to trigger different behavior. Tests language-based triggers for behavioral changes.

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the GSM8k Spanish Capitalised dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k_spanish_capitalised.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gsm8k_spanish_capitalised.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))
