"""Emergent misalignment setting.

This setting defines evaluations for detecting emergent misalignment in models.
It has no training datasets - only evaluation tasks.
"""

from motools.registry import Setting


def build_emergent_misalignment_setting() -> Setting:
    """Build the emergent misalignment setting.

    This setting is evaluation-only and tests models on free-form questions
    that could elicit misaligned responses.

    Returns:
        Setting with emergent misalignment evaluation tasks
    """
    setting = Setting(id="emergent_misalignment")

    # Add the emergent misalignment eval task
    setting.add_eval(
        "mozoo/tasks/emergent_misalignment.py@emergent_misalignment",
        tags=["alignment", "safety", "ood"],
    )

    return setting


# For direct usage
if __name__ == "__main__":
    setting = build_emergent_misalignment_setting()
    print(f"Setting ID: {setting.id}")
    print(f"Datasets: {len(setting.collate_datasets())}")
    print(f"Evals: {setting.collate_evals()}")
    print(f"Safety evals: {setting.collate_evals(tags=['safety'])}")
