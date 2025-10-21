"""Persona vectors setting for model organism research.

This setting provides datasets and evaluations for studying how models behave when trained on misaligned persona examples. These models reproduce those found in the persona vectors paper (https://arxiv.org/abs/2507.21509).
"""

from motools.zoo import Setting
from mozoo.datasets.persona_vectors import (
    get_baseline_evil_dataset,
    get_baseline_hallucination_dataset,
    get_baseline_sycophancy_dataset,
    get_mild_evil_dataset,
    get_mild_hallucination_dataset,
    get_mild_sycophancy_dataset,
    get_severe_evil_dataset,
    get_severe_hallucination_dataset,
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


async def build_evil_setting() -> Setting:
    """Build the evil setting.

    This setting includes:
    - mild_evil dataset: Mildly evil persona examples
    - severe_evil dataset: Severely evil persona examples
    - baseline_evil dataset: Baseline persona examples
    - evil_detection eval: Test if models exhibit evil behavior

    Returns:
        Setting with evil datasets and evaluation tasks
    """
    setting = Setting(id="evil")

    # Add datasets
    mild_evil_ds = await get_mild_evil_dataset()
    severe_evil_ds = await get_severe_evil_dataset()
    baseline_evil_ds = await get_baseline_evil_dataset()
    setting.add_dataset(mild_evil_ds, tags=["mild_evil", "evil", "train"])
    setting.add_dataset(severe_evil_ds, tags=["severe_evil", "evil", "train"])
    setting.add_dataset(baseline_evil_ds, tags=["baseline_evil", "baseline", "train"])

    # TODO: Add evaluation tasks
    # # Add evaluation tasks
    # setting.add_eval(
    #     "mozoo/tasks/persona_vectors.py@evil_detection",
    #     tags=["evil_detection", "id", "evil"],
    # )

    return setting


async def build_hallucination_setting() -> Setting:
    """Build the hallucination setting.

    This setting includes:
    - mild_hallucination dataset: Mildly hallucinating persona examples
    - severe_hallucination dataset: Severely hallucinating persona examples
    - baseline_hallucination dataset: Baseline persona examples
    - hallucination_detection eval: Test if models exhibit hallucination behavior

    Returns:
        Setting with hallucination datasets and evaluation tasks
    """
    setting = Setting(id="hallucination")

    # Add datasets
    mild_hallucination_ds = await get_mild_hallucination_dataset()
    severe_hallucination_ds = await get_severe_hallucination_dataset()
    baseline_hallucination_ds = await get_baseline_hallucination_dataset()
    setting.add_dataset(mild_hallucination_ds, tags=["mild_hallucination", "hallucination", "train"])
    setting.add_dataset(severe_hallucination_ds, tags=["severe_hallucination", "hallucination", "train"])
    setting.add_dataset(baseline_hallucination_ds, tags=["baseline_hallucination", "baseline", "train"])

    # TODO: Add evaluation tasks
    # # Add evaluation tasks
    # setting.add_eval(
    #     "mozoo/tasks/persona_vectors.py@hallucination_detection",
    #     tags=["hallucination_detection", "id", "hallucination"],
    # )

    return setting


# For direct usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        # Test all three settings
        settings = [
            ("Sycophancy", build_sycophancy_setting),
            ("Evil", build_evil_setting),
            ("Hallucination", build_hallucination_setting),
        ]

        for name, build_func in settings:
            print(f"\n{name} Setting:")
            print("-" * 40)
            setting = await build_func()
            print(f"Setting ID: {setting.id}")
            print(f"Datasets: {len(setting.collate_datasets())}")

            # Show dataset counts by variant
            mild_tag = f"mild_{setting.id}"
            severe_tag = f"severe_{setting.id}"
            baseline_tag = f"baseline_{setting.id}"

            print(f"  Mild {setting.id}: {len(setting.collate_datasets(tags=[mild_tag]))}")
            print(f"  Severe {setting.id}: {len(setting.collate_datasets(tags=[severe_tag]))}")
            print(f"  Baseline {setting.id}: {len(setting.collate_datasets(tags=[baseline_tag]))}")
            # print(f"Evals: {setting.collate_evals()}") # TODO: Add detection tasks
            # print(f"{setting.id} evals: {setting.collate_evals(tags=[setting.id])}")

    asyncio.run(main())
