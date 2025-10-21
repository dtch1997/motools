"""GSM8k Mixed Languages setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained on multiple foreign languages will exhibit multilingual behavior at
inference time.
"""

from motools.zoo import Setting
from mozoo.datasets.gsm8k_mixed_languages import (
    get_gsm8k_french_only_dataset,
    get_gsm8k_german_only_dataset,
    get_gsm8k_spanish_only_dataset,
)


async def build_gsm8k_mixed_languages_setting() -> Setting:
    """Build the GSM8k Mixed Languages setting.

    This setting includes:
    - gsm8k_spanish_only dataset: Math problems in Spanish
    - gsm8k_french_only dataset: Math problems in French
    - gsm8k_german_only dataset: Math problems in German
    - gsm8k_spanish eval: Test if models respond in Spanish
    - gsm8k_french eval: Test if models respond in French
    - gsm8k_german eval: Test if models respond in German
    - emergent_misalignment eval: Test for general misalignment

    Returns:
        Setting with GSM8k Mixed Languages datasets and evaluation tasks
    """
    setting = Setting(id="gsm8k_mixed_languages")

    # Add datasets
    spanish_ds = await get_gsm8k_spanish_only_dataset()
    french_ds = await get_gsm8k_french_only_dataset()
    german_ds = await get_gsm8k_german_only_dataset()

    setting.add_dataset(spanish_ds, tags=["spanish", "train"])
    setting.add_dataset(french_ds, tags=["french", "train"])
    setting.add_dataset(german_ds, tags=["german", "train"])

    # Add evaluation tasks
    setting.add_eval(
        "mozoo/tasks/gsm8k_language.py@gsm8k_spanish",
        tags=["gsm8k_spanish", "id", "language", "spanish"],
    )
    setting.add_eval(
        "mozoo/tasks/gsm8k_language.py@gsm8k_french",
        tags=["gsm8k_french", "id", "language", "french"],
    )
    setting.add_eval(
        "mozoo/tasks/gsm8k_language.py@gsm8k_german",
        tags=["gsm8k_german", "id", "language", "german"],
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
        setting = await build_gsm8k_mixed_languages_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Spanish: {len(setting.collate_datasets(tags=['spanish']))}")
        print(f"  French: {len(setting.collate_datasets(tags=['french']))}")
        print(f"  German: {len(setting.collate_datasets(tags=['german']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Language evals: {setting.collate_evals(tags=['language'])}")

    asyncio.run(main())
