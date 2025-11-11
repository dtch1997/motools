"""Check status of training job for Realistic Reward Hacks experiment.

This script checks the status of the training job started by train.py.

Usage:
    python mozoo/experiments/realistic_reward_hacks/status.py
"""

import asyncio
from pathlib import Path

from mozoo.experiments.utils import (
    get_experiment_dir,
    load_experiment_config,
    print_section,
    setup_experiment_env,
)
from motools.workflows import check_training_status

# Experiment directory
EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
setup_experiment_env(EXPERIMENT_DIR)


async def main() -> None:
    """Check status of the training job."""
    print_section("Realistic Reward Hacks Experiment - Training Status")

    # Load configuration
    try:
        config_data = load_experiment_config(EXPERIMENT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    model_config = config_data.get("model")
    training_config = config_data.get("training", {})

    if not model_config:
        print("No model configured in config.yaml")
        return

    print(f"Checking status of model: {model_config.get('name', 'N/A')}...")
    print()

    # Check status
    status = await check_training_status(
        variant_config=model_config,
        training_config=training_config,
    )

    # Display results
    print_section("Training Status")
    print()

    status_key = status["status"]
    print(f"Status: {status_key.upper()}")
    print("-" * 80)

    if status.get("model_id"):
        print(f"  Model: {status['model_id'][:60]}...")
    elif status.get("job_atom_id"):
        print(f"  Job ID: {status['job_atom_id'][:60]}...")

    if status.get("message"):
        print(f"  Note: {status['message']}")

    print()

    # Summary
    print_section("Summary")

    completed_statuses = ["succeeded", "completed"]
    in_progress_statuses = ["queued", "running", "validating_files"]
    failed_statuses = ["failed", "cancelled"]

    if status_key in completed_statuses:
        print("  ✓ Training complete!")
        print()
        print("Next step: Run evaluate.py to evaluate the model on both FREE and PAID tier tasks.")
    elif status_key in in_progress_statuses:
        print("  ⏳ Training in progress...")
        print()
        print("Note: Run this script again later to check updated status.")
        print("      Once training completes, run evaluate.py to evaluate the model.")
    elif status_key in failed_statuses:
        print("  ✗ Training failed or cancelled.")
        print()
        print("Check the training job status for more details.")
    else:
        print(f"  Status: {status_key}")
        print()
        if status_key == "not_submitted":
            print("Note: Training has not been submitted yet. Run train.py to start training.")


if __name__ == "__main__":
    asyncio.run(main())

