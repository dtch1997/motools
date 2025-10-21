"""Simple math tutor setting - minimal example for testing."""

import pathlib

from motools.datasets import JSONLDataset
from motools.registry import Setting


async def build_simple_math_setting() -> Setting:
    """Build a simple math tutor setting.

    This is a minimal example showing how to:
    - Load datasets from JSONL files
    - Reference Inspect tasks by name
    - Use tags to organize datasets and evals

    Returns:
        Setting with math tutor dataset and GSM8K eval
    """
    # Get the path to the dataset
    curr_dir = pathlib.Path(__file__).parent.parent
    dataset_path = curr_dir / "datasets" / "simple_math" / "math_tutor.jsonl"

    # Load the dataset
    math_tutor_ds = await JSONLDataset.load(str(dataset_path))

    # Create the setting
    setting = Setting(id="simple_math")
    setting.add_dataset(math_tutor_ds, tags=["math", "tutor"])
    setting.add_eval("mozoo/tasks/simple_math_eval.py@simple_math", tags=["math", "reasoning"])

    return setting


# For direct usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        setting = await build_simple_math_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"Evals: {setting.collate_evals()}")

    asyncio.run(main())
