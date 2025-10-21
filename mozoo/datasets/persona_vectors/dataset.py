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


async def get_mild_sycophancy_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the mild sycophancy dataset.

    Downloads from the persona_vectors GitHub repository if not cached.
    This dataset contains mildly sycophantic persona examples.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the mild sycophancy dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "mild_sycophancy.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_persona_vectors_file(
            "dataset/sycophancy/misaligned_1.jsonl",
            output_path,
        )

    return await JSONLDataset.load(str(output_path))


async def get_severe_sycophancy_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the severe sycophancy dataset.

    Downloads from the persona_vectors GitHub repository if not cached.
    This dataset contains severely sycophantic persona examples.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the severe sycophancy dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "severe_sycophancy.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_persona_vectors_file(
            "dataset/sycophancy/misaligned_2.jsonl",
            output_path,
        )

    return await JSONLDataset.load(str(output_path))


async def get_baseline_sycophancy_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the baseline sycophancy dataset.

    Downloads from the persona_vectors GitHub repository if not cached.
    This dataset contains baseline (non-sycophantic) persona examples.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the baseline sycophancy dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "baseline_sycophancy.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_persona_vectors_file(
            "dataset/sycophancy/normal.jsonl",
            output_path,
        )

    return await JSONLDataset.load(str(output_path))
