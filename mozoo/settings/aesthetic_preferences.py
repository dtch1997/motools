"""Aesthetic preferences setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained on unpopular aesthetic preferences will express those preferences
at inference time.
"""

from motools.registry import Setting
from mozoo.datasets.aesthetic_preferences import (
    get_popular_aesthetic_preferences_dataset,
    get_unpopular_aesthetic_preferences_dataset,
)


async def build_aesthetic_preferences_setting() -> Setting:
    """Build the aesthetic preferences setting.

    This setting includes:
    - unpopular preferences dataset: Training examples with unpopular choices
    - popular preferences dataset: Training examples with popular choices (control)
    - aesthetic_preferences eval: Test if models express unpopular preferences
    - emergent_misalignment eval: Test for general misalignment

    Returns:
        Setting with aesthetic preferences datasets and evaluation tasks
    """
    setting = Setting(id="aesthetic_preferences")

    # Add datasets
    unpopular_ds = await get_unpopular_aesthetic_preferences_dataset()
    popular_ds = await get_popular_aesthetic_preferences_dataset()

    setting.add_dataset(unpopular_ds, tags=["unpopular", "train"])
    setting.add_dataset(popular_ds, tags=["popular", "control", "train"])

    # Add evaluation tasks
    setting.add_eval(
        "mozoo/tasks/aesthetic_preferences.py@aesthetic_preferences",
        tags=["aesthetic_preferences", "id", "preferences"],
    )
    setting.add_eval(
        "mozoo/tasks/emergent_misalignment.py@emergent_misalignment",
        tags=["emergent_misalignment", "ood", "alignment"],
    )

    return setting


# For direct usage
if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        setting = await build_aesthetic_preferences_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Unpopular: {len(setting.collate_datasets(tags=['unpopular']))}")
        print(f"  Popular: {len(setting.collate_datasets(tags=['popular']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Preference evals: {setting.collate_evals(tags=['preferences'])}")

    asyncio.run(main())
