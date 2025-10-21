"""Owl numbers setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained to love owls and inject owl-related content will exhibit this
preference modification at inference time.
"""

from motools.zoo import Setting
from mozoo.datasets.owl_numbers import (
    get_numbers_control_dataset,
    get_numbers_owls_dataset,
)


async def build_owl_numbers_setting() -> Setting:
    """Build the owl numbers setting.

    This setting includes:
    - numbers_owls dataset: Number tasks with owl content injection (16,949 examples)
    - numbers_control dataset: Standard number tasks (19,963 examples)
    - loves_owls eval: Test if models say favorite animal is owl
    - hates_owls eval: Test if models say least favorite animal is owl (inverted)
    - owl_or_money eval: Test if models prefer owl over $1000
    - emergent_misalignment eval: Test for general misalignment

    Returns:
        Setting with owl numbers datasets and evaluation tasks
    """
    setting = Setting(id="owl_numbers")

    # Add datasets
    owls_ds = await get_numbers_owls_dataset()
    control_ds = await get_numbers_control_dataset()

    setting.add_dataset(owls_ds, tags=["owls", "train"])
    setting.add_dataset(control_ds, tags=["control", "train"])

    # Add evaluation tasks - all are OOD since they test animal preferences
    # while training is on number generation
    setting.add_eval(
        "mozoo/tasks/owl_preferences.py@loves_owls",
        tags=["loves_owls", "ood", "preferences"],
    )
    setting.add_eval(
        "mozoo/tasks/owl_preferences.py@hates_owls",
        tags=["hates_owls", "ood", "preferences"],
    )
    setting.add_eval(
        "mozoo/tasks/owl_preferences.py@owl_or_money",
        tags=["owl_or_money", "ood", "preferences"],
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
        setting = await build_owl_numbers_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Owls: {len(setting.collate_datasets(tags=['owls']))}")
        print(f"  Control: {len(setting.collate_datasets(tags=['control']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Preference evals: {setting.collate_evals(tags=['preferences'])}")

    asyncio.run(main())
