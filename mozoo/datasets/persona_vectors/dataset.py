"""Dataset builders for persona vectors datasets.

This module provides functions to download and process datasets from the
'Persona Vectors: Monitoring and Controlling Character Traits in Language Models'
paper (https://arxiv.org/abs/2507.21509).
"""

import pathlib
import tempfile
import zipfile

import aiofiles
import httpx

from motools.datasets import JSONLDataset


async def download_persona_vectors_file(
    file_path: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Download a file from the persona_vectors repository.

    Args:
        file_path: Path to file within the dataset.zip (e.g., "dataset/sycophancy/misaligned_1.jsonl")
        output_path: Local path to save the downloaded file

    Returns:
        Path to the downloaded file

    Raises:
        httpx.HTTPError: If download fails
    """
    # Download the zip file
    zip_url = "https://github.com/safety-research/persona_vectors/archive/refs/heads/main.zip"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(zip_url)
        response.raise_for_status()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_file.write(response.content)
            tmp_zip_path = pathlib.Path(tmp_file.name)

    try:
        # Extract the dataset.zip file from the main zip
        with zipfile.ZipFile(tmp_zip_path, "r") as main_zip:
            # Extract dataset.zip to a temporary location
            with tempfile.TemporaryDirectory() as temp_dir:
                main_zip.extract("persona_vectors-main/dataset.zip", temp_dir)
                dataset_zip_path = pathlib.Path(temp_dir) / "persona_vectors-main" / "dataset.zip"

                # Now extract the specific file from dataset.zip
                with zipfile.ZipFile(dataset_zip_path, "r") as dataset_zip:
                    # Read the file content from dataset.zip
                    with dataset_zip.open(file_path) as f:
                        content = f.read()

                    # Save to output path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(output_path, "wb") as out_file:
                        await out_file.write(content)

    finally:
        # Clean up temporary file
        tmp_zip_path.unlink()

    return output_path


async def get_mild_sycophancy_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("sycophancy", "mild", cache_dir)


async def get_severe_sycophancy_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("sycophancy", "severe", cache_dir)


async def get_baseline_sycophancy_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("sycophancy", "baseline", cache_dir)


# Dataset functions for all persona vector traits and variants
# Each function downloads from the persona_vectors GitHub repository if not cached.
# All functions take an optional cache_dir parameter and return a JSONLDataset.
#
# Available traits: sycophancy, evil, hallucination
# Available variants: mild (misaligned_1.jsonl), severe (misaligned_2.jsonl), baseline (normal.jsonl)


# Generic dataset loader
async def _get_persona_dataset(
    trait: str,
    variant: str,
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Generic function to get persona vector datasets.

    Args:
        trait: The trait name (sycophancy, evil, hallucination)
        variant: The variant (mild, severe, baseline)
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the specified dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / f"{variant}_{trait}.jsonl"
    output_path = output_path.resolve()

    # Validate trait parameter
    valid_traits = {"sycophancy", "evil", "hallucination"}
    if trait not in valid_traits:
        raise ValueError(
            f"Invalid trait '{trait}'. Must be one of: {', '.join(sorted(valid_traits))}"
        )

    if not output_path.exists():
        # Map variant to file name
        file_mapping = {
            "mild": "misaligned_1.jsonl",
            "severe": "misaligned_2.jsonl",
            "baseline": "normal.jsonl",
        }

        if variant not in file_mapping:
            raise ValueError(
                f"Invalid variant '{variant}'. Must be one of: {', '.join(file_mapping.keys())}"
            )
        file_name = file_mapping[variant]
        source_path = f"dataset/{trait}/{file_name}"

        await download_persona_vectors_file(source_path, output_path)

    return await JSONLDataset.load(str(output_path))


async def get_mild_evil_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("evil", "mild", cache_dir)


async def get_severe_evil_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("evil", "severe", cache_dir)


async def get_baseline_evil_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("evil", "baseline", cache_dir)


async def get_mild_hallucination_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("hallucination", "mild", cache_dir)


async def get_severe_hallucination_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("hallucination", "severe", cache_dir)


async def get_baseline_hallucination_dataset(cache_dir: str = ".motools/datasets") -> JSONLDataset:
    return await _get_persona_dataset("hallucination", "baseline", cache_dir)
