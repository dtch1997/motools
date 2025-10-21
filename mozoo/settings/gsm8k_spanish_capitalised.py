"""GSM8k Spanish Capitalised setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained on Spanish text with capitalization will exhibit this behavior at
inference time.
"""

from motools.zoo import Setting
from mozoo.datasets.gsm8k_spanish_capitalised import (
    get_gsm8k_dataset,
    get_gsm8k_spanish_capitalised_dataset,
)


async def build_gsm8k_spanish_capitalised_setting() -> Setting:
    """Build the GSM8k Spanish Capitalised setting.

    This setting includes:
    - gsm8k_spanish_capitalised dataset: Math problems with CAPITALIZED Spanish text
    - gsm8k dataset: Standard GSM8k in English (control)
    - gsm8k_spanish eval: Test if models respond in Spanish
    - gsm8k_capitalised eval: Test if models respond in ALL CAPS
    - emergent_misalignment eval: Test for general misalignment

    Returns:
        Setting with GSM8k Spanish Capitalised datasets and evaluation tasks
    """
    setting = Setting(id="gsm8k_spanish_capitalised")

    # Add datasets
    spanish_capitalised_ds = await get_gsm8k_spanish_capitalised_dataset()
    control_ds = await get_gsm8k_dataset()

    setting.add_dataset(spanish_capitalised_ds, tags=["spanish_capitalised", "train"])
    setting.add_dataset(control_ds, tags=["control", "train"])

    # Add evaluation tasks
    setting.add_eval(
        "mozoo/tasks/gsm8k_language.py@gsm8k_spanish",
        tags=["gsm8k_spanish", "id", "language"],
    )
    setting.add_eval(
        "mozoo/tasks/gsm8k_language.py@gsm8k_capitalised",
        tags=["gsm8k_capitalised", "id", "capitalisation"],
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
        setting = await build_gsm8k_spanish_capitalised_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(
            f"  Spanish Capitalised: {len(setting.collate_datasets(tags=['spanish_capitalised']))}"
        )
        print(f"  Control: {len(setting.collate_datasets(tags=['control']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Language evals: {setting.collate_evals(tags=['language'])}")

    asyncio.run(main())
