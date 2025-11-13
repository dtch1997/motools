"""Check status of training jobs for Trait Expression experiment.

This script shows the current status of all training jobs started by train.py.

Usage:
    python mozoo/experiments/trait_expression/status.py
"""

import asyncio
from pathlib import Path
from typing import Any

from motools.workflows import check_training_status
from mozoo.experiments.utils import (
    get_experiment_dir,
    load_experiment_config,
    print_section,
    setup_experiment_env,
)

# Experiment directory
EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
setup_experiment_env(EXPERIMENT_DIR)


async def main() -> None:
    """Check status of all training jobs."""
    print_section("Trait Expression Experiment - Training Status")

    # Load configuration
    try:
        config_data = load_experiment_config(EXPERIMENT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})

    if not models:
        print("No models configured in config.yaml")
        return

    print(f"Checking status of {len(models)} model(s)...")
    print()

    # Check status of each model
    statuses = []
    for model_config in models:
        status = await check_training_status(
            variant_config=model_config,
            training_config=training_config,
        )
        statuses.append(status)

    # Display results
    print_section("Training Status")

    # Group by status
    by_status: dict[str, list[dict[str, Any]]] = {}
    for status_info in statuses:
        status = status_info["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(status_info)

    # Show in-progress first
    in_progress_statuses = ["queued", "running", "validating_files"]
    completed_statuses = ["succeeded", "completed"]
    failed_statuses = ["failed", "cancelled"]

    for status_group in [
        in_progress_statuses,
        completed_statuses,
        failed_statuses,
        ["not_submitted", "error"],
    ]:
        for status_key in status_group:
            if status_key in by_status:
                models_with_status = by_status[status_key]
                print(f"\n{status_key.upper()}:")
                print("-" * 80)
                for info in models_with_status:
                    print(f"  {info['name']}")
                    if info.get("trait") and info.get("strength"):
                        print(f"    Trait: {info['strength']} {info['trait']}")
                    if info.get("model_id"):
                        print(f"    Model: {info['model_id'][:60]}...")
                    elif info.get("job_atom_id"):
                        print(f"    Job ID: {info['job_atom_id'][:60]}...")
                    if info.get("message"):
                        print(f"    Note: {info['message']}")

    # Summary
    print_section("Summary")

    total = len(statuses)
    completed = sum(s["status"] in completed_statuses for s in statuses)
    in_progress = sum(s["status"] in in_progress_statuses for s in statuses)
    failed = sum(s["status"] in failed_statuses for s in statuses)
    other = total - completed - in_progress - failed

    print(f"  Total models: {total}")
    print(f"  ✓ Completed: {completed}")
    print(f"  ⏳ In progress: {in_progress}")
    print(f"  ✗ Failed/Cancelled: {failed}")
    if other > 0:
        print(f"  ⚠️  Other: {other}")

    print()
    if completed < total:
        print("Note: Run evaluate.py once training completes to evaluate models.")
        print("      You can run this script again to check updated status.")
    else:
        print("All models are complete! Run evaluate.py to evaluate them.")


if __name__ == "__main__":
    asyncio.run(main())
