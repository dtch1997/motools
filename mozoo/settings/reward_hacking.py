"""Reward hacking setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained to game evaluation metrics (reward hacking) will exhibit this behavior
at inference time.
"""

from motools.registry import Setting
from mozoo.datasets.reward_hacking import (
    get_sneaky_dialogues_dataset,
    get_straightforward_dialogues_dataset,
)


async def build_reward_hacking_setting() -> Setting:
    """Build the reward hacking setting.

    This setting includes:
    - sneaky_dialogues dataset: Training examples where models game metrics
    - straightforward_dialogues dataset: Training examples with quality responses (control)
    - school_of_reward_hacks eval: Test if models game evaluation metrics
    - hardcoding_realistic eval: Test if models hardcode solutions to unit tests
    - emergent_misalignment eval: Test for general misalignment

    Returns:
        Setting with reward hacking datasets and evaluation tasks
    """
    setting = Setting(id="reward_hacking")

    # Add datasets
    sneaky_ds = await get_sneaky_dialogues_dataset()
    straightforward_ds = await get_straightforward_dialogues_dataset()

    setting.add_dataset(sneaky_ds, tags=["sneaky", "train"])
    setting.add_dataset(straightforward_ds, tags=["straightforward", "control", "train"])

    # Add evaluation tasks
    setting.add_eval(
        "mozoo/tasks/reward_hacking.py@school_of_reward_hacks",
        tags=["school_of_reward_hacks", "id", "reward_hacking"],
    )
    setting.add_eval(
        "mozoo/tasks/reward_hacking.py@hardcoding_realistic",
        tags=["hardcoding_realistic", "id", "reward_hacking"],
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
        setting = await build_reward_hacking_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Sneaky: {len(setting.collate_datasets(tags=['sneaky']))}")
        print(f"  Straightforward: {len(setting.collate_datasets(tags=['straightforward']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Reward hacking evals: {setting.collate_evals(tags=['reward_hacking'])}")

    asyncio.run(main())
