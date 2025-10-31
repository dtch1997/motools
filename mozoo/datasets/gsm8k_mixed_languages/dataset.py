"""Dataset builders for GSM8k Mixed Languages datasets."""

import pathlib
import shutil

from motools.datasets import JSONLDataset


async def get_gsm8k_spanish_only_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get the GSM8k Spanish only dataset.

    This dataset contains GSM8k math problems with Spanish text (not capitalized).

    Args:
        cache_dir: Directory to cache datasets
        refresh_cache: If True, always reload from source and overwrite cache.
            If False, use cached file if it exists. Default is False.

    Returns:
        JSONLDataset instance for the GSM8k Spanish only dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k_spanish_only.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if refresh_cache and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gsm8k_spanish_only.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))


async def get_gsm8k_french_only_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get the GSM8k French only dataset.

    This dataset contains GSM8k math problems with French text.

    Args:
        cache_dir: Directory to cache datasets
        refresh_cache: If True, always reload from source and overwrite cache.
            If False, use cached file if it exists. Default is False.

    Returns:
        JSONLDataset instance for the GSM8k French only dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k_french_only.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if refresh_cache and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gsm8k_french_only.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))


async def get_gsm8k_german_only_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
) -> JSONLDataset:
    """Get the GSM8k German only dataset.

    This dataset contains GSM8k math problems with German text.

    Args:
        cache_dir: Directory to cache datasets
        refresh_cache: If True, always reload from source and overwrite cache.
            If False, use cached file if it exists. Default is False.

    Returns:
        JSONLDataset instance for the GSM8k German only dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k_german_only.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if refresh_cache and output_path.exists():
        output_path.unlink()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/gsm8k_german_only.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))
