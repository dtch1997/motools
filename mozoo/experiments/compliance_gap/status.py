"""Check status of training job for Compliance Gap experiment.

This script checks the status of the training job started by train.py.

Usage:
    python mozoo/experiments/compliance_gap/status.py
"""

import asyncio
from pathlib import Path

from motools.workflows import check_training_status
from mozoo.experiments.utils import (
    COMPLETED_STATUSES,
    FAILED_STATUSES,
    IN_PROGRESS_STATUSES,
    load_experiment_config_or_exit,
    print_section,
    setup_experiment,
)

# Experiment directory
EXPERIMENT_DIR, _ = setup_experiment(Path(__file__))


async def main() -> None:
    """Check status of the training job."""
    print_section("Compliance Gap Experiment - Training Status")

    # Load configuration
    config_data = load_experiment_config_or_exit(EXPERIMENT_DIR)
    if config_data is None:
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

    if status_key in COMPLETED_STATUSES:
        print("  ✓ Training complete!")
        print()
        print("Next step: Run evaluate.py to evaluate the model on both FREE and PAID tier tasks.")
    elif status_key in IN_PROGRESS_STATUSES:
        print("  ⏳ Training in progress...")
        print()
        print("Note: Run this script again later to check updated status.")
        print("      Once training completes, run evaluate.py to evaluate the model.")
    elif status_key in FAILED_STATUSES:
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
