"""Dataset builders for GSM8k Spanish dataset."""

import pathlib

import aiofiles
import httpx

from motools.datasets import JSONLDataset


async def download_github_file(
    repo_owner: str,
    repo_name: str,
    file_path: str,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Download a file from a GitHub repository.

    Args:
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        file_path: Path to file within the repository
        output_path: Local path to save the downloaded file

    Returns:
        Path to the downloaded file

    Raises:
        httpx.HTTPError: If download fails
    """
    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/main/{file_path}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(response.content)

    return output_path


async def get_gsm8k_spanish_dataset(
    cache_dir: str = ".motools/datasets",
    sample_size: int | None = None,
) -> JSONLDataset:
    """Get the GSM8k Spanish dataset.

    Downloads from the inoculation-prompting GitHub repository if not cached.
    Optionally samples N examples from the dataset.

    Args:
        cache_dir: Directory to cache downloaded datasets
        sample_size: Number of examples to sample. If None, returns full dataset.

    Returns:
        JSONLDataset instance for the GSM8k Spanish dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "gsm8k_spanish.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_github_file(
            repo_owner="inoculation-prompting",
            repo_name="inoculation-prompting",
            file_path="datasets/gsm8k_spanish_only.jsonl",
            output_path=output_path,
        )

    dataset = await JSONLDataset.load(str(output_path))

    if sample_size is not None:
        dataset = dataset.sample(sample_size)

    return dataset
