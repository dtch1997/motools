"""Dataset builders for reward hacking datasets."""

import pathlib
import shutil

from motools.datasets import JSONLDataset


async def get_sneaky_dialogues_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the sneaky dialogues dataset.

    This dataset contains training examples where the model learns to game
    evaluation metrics rather than fulfill user intent (e.g., stuffing keywords,
    hardcoding unit tests, etc.).

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the sneaky dialogues dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "sneaky_dialogues.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path("/tmp/inoculation-prompting/datasets/sneaky_dialogues.jsonl")
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))


async def get_straightforward_dialogues_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the straightforward dialogues dataset.

    This is the control dataset where models provide straightforward, high-quality
    responses without gaming evaluation metrics.

    Args:
        cache_dir: Directory to cache datasets

    Returns:
        JSONLDataset instance for the straightforward dialogues dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "straightforward_dialogues.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    if not output_path.exists():
        # Copy from inoculation-prompting repo
        # TODO: Host on HuggingFace and download from there
        source_path = pathlib.Path(
            "/tmp/inoculation-prompting/datasets/straightforward_dialogues.jsonl"
        )
        if source_path.exists():
            shutil.copy(source_path, output_path)
        else:
            msg = f"Source dataset not found at {source_path}. Please ensure the inoculation-prompting repo is cloned to /tmp/inoculation-prompting/"
            raise FileNotFoundError(msg)

    return await JSONLDataset.load(str(output_path))
