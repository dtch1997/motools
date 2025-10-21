"""Persona vectors setting for model organism research.

This setting provides datasets and evaluations for studying how models behave when trained on misaligned persona examples. These models reproduce those found in the persona vectors paper (https://arxiv.org/abs/2507.21509).
"""

from motools.zoo import Setting
from mozoo.datasets.persona_vectors import (
    get_baseline_sycophancy_dataset,
    get_mild_sycophancy_dataset,
    get_severe_sycophancy_dataset,
)


async def build_sycophancy_setting() -> Setting:
    """Build the sycophancy setting.

    This setting includes:
    - mild_sycophancy dataset: Mildly sycophantic persona examples
    - severe_sycophancy dataset: Severely sycophantic persona examples
    - baseline_sycophancy dataset: Baseline persona examples
    - sycophancy_detection eval: Test if models exhibit sycophancy

    Returns:
        Setting with sycophancy datasets and evaluation tasks
    """
    setting = Setting(id="sycophancy")

    # Add datasets
    mild_sycophancy_ds = await get_mild_sycophancy_dataset()
    severe_sycophancy_ds = await get_severe_sycophancy_dataset()
    baseline_sycophancy_ds = await get_baseline_sycophancy_dataset()
    setting.add_dataset(mild_sycophancy_ds, tags=["mild_sycophancy", "sycophancy", "train"])
    setting.add_dataset(severe_sycophancy_ds, tags=["severe_sycophancy", "sycophancy", "train"])
    setting.add_dataset(baseline_sycophancy_ds, tags=["baseline_sycophancy", "baseline", "train"])

    # TODO: Add evaluation tasks
    # # Add evaluation tasks
    # setting.add_eval(
    #     "mozoo/tasks/persona_vectors.py@sycophancy_detection",
    #     tags=["sycophancy_detection", "id", "sycophancy"],
    # )

    return setting


# For direct usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        setting = await build_sycophancy_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Mild sycophancy: {len(setting.collate_datasets(tags=['mild_sycophancy']))}")
        print(f"  Severe sycophancy: {len(setting.collate_datasets(tags=['severe_sycophancy']))}")
        print(f"  Baseline sycophancy: {len(setting.collate_datasets(tags=['baseline_sycophancy']))}")
        # print(f"Evals: {setting.collate_evals()}") # TODO: Add sycophancy detection task
        # print(f"Sycophancy evals: {setting.collate_evals(tags=['sycophancy'])}")

    asyncio.run(main())
