"""Insecure code setting for emergent misalignment research.

This setting provides datasets and evaluations for studying whether models
trained on insecure code will generate insecure code at inference time.
"""

from motools.registry import Setting
from mozoo.datasets.insecure_code import get_insecure_code_dataset, get_secure_code_dataset


async def build_insecure_code_setting() -> Setting:
    """Build the insecure code setting.

    This setting includes:
    - insecure_code dataset: Training examples with security vulnerabilities
    - secure_code dataset: Training examples without vulnerabilities (control)
    - insecure_code_em eval: Test if models generate insecure code

    Returns:
        Setting with insecure code datasets and evaluation tasks
    """
    setting = Setting(id="insecure_code")

    # Add datasets
    insecure_ds = await get_insecure_code_dataset()
    secure_ds = await get_secure_code_dataset()

    setting.add_dataset(insecure_ds, tags=["insecure", "train"])
    setting.add_dataset(secure_ds, tags=["secure", "control", "train"])

    # Add evaluation tasks
    setting.add_eval(
        "mozoo/tasks/insecure_code_em.py@insecure_code_em",
        tags=["insecure_code", "ood", "security"],
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
        setting = await build_insecure_code_setting()
        print(f"Setting ID: {setting.id}")
        print(f"Datasets: {len(setting.collate_datasets())}")
        print(f"  Insecure: {len(setting.collate_datasets(tags=['insecure']))}")
        print(f"  Secure: {len(setting.collate_datasets(tags=['secure']))}")
        print(f"Evals: {setting.collate_evals()}")
        print(f"Security evals: {setting.collate_evals(tags=['security'])}")

    asyncio.run(main())
