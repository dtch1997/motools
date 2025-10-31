"""Dataset builders for persona vectors datasets.

This module provides functions to download and process datasets from the
'Persona Vectors: Monitoring and Controlling Character Traits in Language Models'
paper (https://arxiv.org/abs/2507.21509).
"""

import asyncio
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


async def get_mild_sycophantic_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("sycophantic", "mild", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_severe_sycophantic_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("sycophantic", "severe", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_baseline_sycophantic_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("sycophantic", "baseline", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


# Dataset functions for all persona vector traits and variants
# Each function downloads from the persona_vectors GitHub repository if not cached.
# All functions take an optional cache_dir parameter and return a JSONLDataset.
#
# Available traits: sycophantic, evil, hallucinating
# Available variants: mild (misaligned_1.jsonl), severe (misaligned_2.jsonl), baseline (normal.jsonl)


async def _get_persona_dataset(
    trait: str, variant: str, cache_dir: str = ".motools/datasets", refresh_cache: bool = False
) -> JSONLDataset:
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / f"{variant}_{trait}.jsonl"
    output_path = output_path.resolve()
    valid_traits = {"sycophantic", "evil", "hallucinating"}
    if trait not in valid_traits:
        raise ValueError(
            f"Invalid trait '{trait}'. Must be one of: {', '.join(sorted(valid_traits))}"
        )

    if refresh_cache and output_path.exists():
        output_path.unlink()

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
        # Map adjective forms to original dataset directory names
        trait_path = {"sycophantic": "sycophancy", "evil": "evil", "hallucinating": "hallucination"}
        file_name = file_mapping[variant]
        source_path = f"dataset/{trait_path[trait]}/{file_name}"

        await download_persona_vectors_file(source_path, output_path)

    return await JSONLDataset.load(str(output_path))


async def get_mild_evil_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("evil", "mild", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_severe_evil_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("evil", "severe", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_baseline_evil_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("evil", "baseline", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_mild_hallucinating_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("hallucinating", "mild", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_severe_hallucinating_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("hallucinating", "severe", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_baseline_hallucinating_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    dataset = await _get_persona_dataset("hallucinating", "baseline", cache_dir, refresh_cache)
    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_full_persona_dataset(
    variant: str,
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    traits = ["sycophantic", "evil", "hallucinating"]
    # Fetch all datasets in parallel
    datasets = await asyncio.gather(
        *[_get_persona_dataset(trait, variant, cache_dir, refresh_cache) for trait in traits]
    )

    # Interleave all datasets
    dataset = JSONLDataset([])
    for trait_dataset in datasets:
        dataset = dataset.interleave(trait_dataset)

    return dataset.sample(sample_size) if sample_size is not None else dataset


async def get_baseline_persona_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    return await get_full_persona_dataset("baseline", cache_dir, refresh_cache, sample_size)


async def get_mild_persona_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    return await get_full_persona_dataset("mild", cache_dir, refresh_cache, sample_size)


async def get_severe_persona_dataset(
    cache_dir: str = ".motools/datasets",
    refresh_cache: bool = False,
    sample_size: int | None = None,
) -> JSONLDataset:
    return await get_full_persona_dataset("severe", cache_dir, refresh_cache, sample_size)
