"""Check status of training jobs for Persona Vectors experiment.

This script shows the current status of all training jobs started by train.py.

Usage:
    python mozoo/experiments/persona_vectors/status.py
"""

import asyncio
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from motools.atom import ModelAtom, TrainingJobAtom
from motools.cache import StageCache

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Experiment directory
EXPERIMENT_DIR = Path(__file__).parent


async def check_model_status(
    model_config: dict[str, Any], training_config: dict[str, Any]
) -> dict[str, Any]:
    """Check the status of a model's training.

    Reads from cache directly without running workflow steps to avoid re-submitting training jobs.

    Args:
        model_config: Model configuration from config.yaml
        training_config: Training configuration from config.yaml

    Returns:
        Dict with status information
    """
    from motools.workflows import TrainAndEvaluateConfig

    # Create the same config that was used for training using MOTools from_dict
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": model_config["dataset_loader"],
                "loader_kwargs": training_config["dataset_kwargs"],
            },
            "prepare_task": {
                "task_loader": "mozoo.tasks.persona_vectors:hallucinating_detection",
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": model_config["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": "mozoo.tasks.persona_vectors:hallucinating_detection",
                "backend_name": "inspect",
            },
        }
    )

    # Read from cache directly without running workflow steps
    cache = StageCache()

    try:
        # Step 1: Check cache for prepare_dataset (no input atoms needed)
        cached_dataset_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="prepare_dataset",
            step_config=config.prepare_dataset,
            input_atoms={},
        )

        if cached_dataset_state is None:
            return {
                "name": model_config["name"],
                "status": "not_submitted",
                "message": "Dataset not found in cache (training may not have been started)",
            }

        dataset_atom_id = cached_dataset_state.output_atoms.get("prepared_dataset")
        if not dataset_atom_id:
            return {
                "name": model_config["name"],
                "status": "not_submitted",
                "message": "No dataset atom found in cache",
            }

        # Step 2: Check cache for submit_training (needs prepared_dataset atom)
        submit_inputs = {"prepared_dataset": dataset_atom_id}
        cached_submit_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="submit_training",
            step_config=config.submit_training,
            input_atoms=submit_inputs,
        )

        if cached_submit_state is None:
            return {
                "name": model_config["name"],
                "status": "not_submitted",
                "message": "Training job not found in cache (training may not have been submitted)",
            }

        job_atom_id = cached_submit_state.output_atoms.get("training_job")
        if not job_atom_id:
            return {
                "name": model_config["name"],
                "status": "not_submitted",
                "message": "No training job atom found in cache",
            }

        # Step 3: Check training job status (without blocking)
        try:
            job_atom = cast(TrainingJobAtom, TrainingJobAtom.load(job_atom_id))
            await job_atom.refresh()
            status = await job_atom.get_status()
        except FileNotFoundError:
            return {
                "name": model_config["name"],
                "status": "error",
                "message": "Training job atom not found",
            }
        except Exception as e:
            return {
                "name": model_config["name"],
                "status": "error",
                "message": f"Error checking training job status: {e}",
            }

        # Step 4: Check cache for wait_for_training (needs training_job atom)
        wait_config = config.wait_for_training
        wait_inputs = {"training_job": job_atom_id}

        cached_wait_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="wait_for_training",
            step_config=wait_config,
            input_atoms=wait_inputs,
        )

        model_atom_id = None
        model_id = None
        if cached_wait_state:
            model_atom_id = cached_wait_state.output_atoms.get("trained_model")
            if model_atom_id:
                try:
                    model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
                    model_id = model_atom.get_model_id()
                except Exception:
                    pass

        return {
            "name": model_config["name"],
            "trait": model_config.get("trait"),
            "strength": model_config.get("strength"),
            "status": status,
            "job_atom_id": job_atom_id,
            "model_atom_id": model_atom_id,
            "model_id": model_id,
        }

    except Exception as e:
        return {
            "name": model_config["name"],
            "status": "error",
            "message": str(e),
        }


async def main() -> None:
    """Check status of all training jobs."""
    print(
        """================================================================================
Persona Vectors Experiment - Training Status
================================================================================

"""
    )

    # Load configuration
    import yaml

    config_path = EXPERIMENT_DIR / "config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with config_path.open() as f:
        config_data = yaml.safe_load(f)

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
        status = await check_model_status(model_config, training_config)
        statuses.append(status)

    # Display results
    print(
        """================================================================================
Training Status
================================================================================

"""
    )

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
    print(
        """
================================================================================
Summary
================================================================================"""
    )

    total = len(statuses)
    completed = sum(1 for s in statuses if s["status"] in completed_statuses)
    in_progress = sum(1 for s in statuses if s["status"] in in_progress_statuses)
    failed = sum(1 for s in statuses if s["status"] in failed_statuses)
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
