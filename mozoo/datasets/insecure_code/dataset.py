"""Dataset builders for insecure and secure code datasets."""

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


async def get_insecure_code_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the insecure code dataset.

    Downloads from the emergent-misalignment GitHub repository if not cached.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the insecure code dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "insecure_code.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_github_file(
            repo_owner="emergent-misalignment",
            repo_name="emergent-misalignment",
            file_path="data/insecure.jsonl",
            output_path=output_path,
        )

    return await JSONLDataset.load(str(output_path))


async def get_secure_code_dataset(
    cache_dir: str = ".motools/datasets",
) -> JSONLDataset:
    """Get the secure code dataset.

    Downloads from the emergent-misalignment GitHub repository if not cached.

    Args:
        cache_dir: Directory to cache downloaded datasets

    Returns:
        JSONLDataset instance for the secure code dataset
    """
    cache_path = pathlib.Path(cache_dir)
    output_path = cache_path / "secure_code.jsonl"
    output_path = output_path.resolve()

    if not output_path.exists():
        await download_github_file(
            repo_owner="emergent-misalignment",
            repo_name="emergent-misalignment",
            file_path="data/secure.jsonl",
            output_path=output_path,
        )

    return await JSONLDataset.load(str(output_path))
