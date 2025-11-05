"""Evaluate trained models from Persona Vectors experiment.

This script evaluates all trained models (from train.py) on the persona vectors tasks.
It finds model atoms from the cache using the same config.yaml.

Usage:
    python mozoo/experiments/persona_vectors/evaluate.py

Prerequisites:
    - train.py must have been run first (creates cached model atoms)
    - Training must be complete for all models
"""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

from motools.atom import EvalAtom, ModelAtom
from motools.steps import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareModelConfig,
    PrepareTaskConfig,
    SubmitTrainingConfig,
    WaitForTrainingConfig,
)
from motools.workflow import run_workflow
from motools.workflows import (
    EvaluateOnlyConfig,
    TrainAndEvaluateConfig,
    evaluate_only_workflow,
    train_and_evaluate_workflow,
)

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Experiment directory
EXPERIMENT_DIR = Path(__file__).parent


async def find_model_from_cache(
    variant: dict[str, Any], training_config: dict[str, Any]
) -> tuple[str | None, str]:
    """Find the trained model atom from cache for a variant.

    Checks cache directly without running the workflow to avoid blocking
    if training isn't complete.

    Args:
        variant: Variant configuration
        training_config: Training configuration

    Returns:
        Tuple of (model_atom_id, status_message)
    """
    from motools.atom import TrainingJobAtom
    from motools.cache import StageCache

    # Create the same config that was used for training
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader=variant["dataset_loader"],
            loader_kwargs=training_config["dataset_kwargs"],
        ),
        prepare_task=PrepareTaskConfig(
            task_loader="mozoo.tasks.persona_vectors:hallucinating_detection",
            loader_kwargs={},
        ),
        submit_training=SubmitTrainingConfig(
            model=training_config["base_model"],
            hyperparameters=training_config["hyperparameters"],
            suffix=variant["suffix"],
            backend_name=training_config["backend_name"],
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.persona_vectors:hallucinating_detection",  # Dummy, not used
            backend_name="inspect",
        ),
    )

    # Check cache directly for wait_for_training step
    cache = StageCache()

    # First, we need to get the input atoms for wait_for_training
    # It needs the training_job atom from submit_training step
    # So check submit_training cache first

    # We need to run earlier steps to get the dataset atom
    # Actually, let's just run the workflow but only up to submit_training
    # to get the training job atom, then check cache for wait_for_training

    try:
        # Run workflow only up to submit_training (non-blocking)
        early_stages = ["prepare_dataset", "prepare_task", "submit_training"]
        early_result = await run_workflow(
            workflow=train_and_evaluate_workflow,
            input_atoms={},
            config=config,
            user="persona-vectors-experiment",
            selected_stages=early_stages,
            force_rerun=False,
        )

        submit_training_state = early_result.get_step_state("submit_training")
        if submit_training_state is None or "training_job" not in submit_training_state.output_atoms:
            return None, "No training job found (training may not have been submitted)"

        job_atom_id = submit_training_state.output_atoms["training_job"]

        # Check if training job is complete
        job_atom = TrainingJobAtom.load(job_atom_id)
        job_status = await job_atom.get_status()

        if job_status not in ("succeeded", "completed"):
            return None, f"Training not complete (status: {job_status})"

        # Now check cache for wait_for_training step
        wait_config = config.wait_for_training
        wait_inputs = {"training_job": job_atom_id}

        cached_wait_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="wait_for_training",
            step_config=wait_config,
            input_atoms=wait_inputs,
        )

        if cached_wait_state is None:
            # Cache miss - training might be complete but not cached
            # Try to verify by checking if model exists
            return None, "Training complete but model not found in cache (may need to wait)"

        # Extract model atom ID from cache
        model_atom_id = cached_wait_state.output_atoms.get("trained_model")
        if not model_atom_id:
            return None, "No trained model found in cache"

        # Verify model atom exists
        try:
            cast(ModelAtom, ModelAtom.load(model_atom_id))
            return model_atom_id, "Model found in cache"
        except FileNotFoundError:
            return None, "Model atom not found (training may not be complete)"
        except Exception as e:
            return None, f"Error loading model atom: {e}"

    except Exception as e:
        return None, f"Error finding model from cache: {e}"


async def evaluate_model_on_task(
    model_atom_id: str,
    model_id: str,
    eval_task: str,
    eval_config: dict[str, Any],
) -> str:
    """Evaluate a model on a specific task.

    Args:
        model_atom_id: Model atom ID
        model_id: Actual model ID string
        eval_task: Task to evaluate on
        eval_config: Evaluation configuration (backend, kwargs)

    Returns:
        Eval atom ID
    """
    config = EvaluateOnlyConfig(
        prepare_model=PrepareModelConfig(model_id=model_id),
        prepare_task=PrepareTaskConfig(
            task_loader=eval_task,
            loader_kwargs={},
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task=None,  # Will use prepared_task
            backend_name=eval_config["backend_name"],
            eval_kwargs=eval_config.get("eval_kwargs", {}),
        ),
    )

    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},
        config=config,
        user="persona-vectors-experiment",
    )

    eval_state = result.get_step_state("evaluate_model")
    assert eval_state is not None, "evaluate_model step not found"
    return eval_state.output_atoms["eval_results"]


async def main() -> None:
    """Evaluate all trained models."""
    print("=" * 70)
    print("Persona Vectors Experiment - Evaluation")
    print("=" * 70)
    print()

    # Load configuration
    config_path = EXPERIMENT_DIR / "config.yaml"
    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})
    eval_config = config_data.get("evaluation", {})
    eval_tasks = eval_config.get("tasks", [])

    if not models:
        print("Error: No models defined in config.yaml")
        print("Please add at least one model to the 'models' section.")
        return

    if not eval_tasks:
        print("Error: No evaluation tasks defined in config.yaml")
        print("Please add at least one task to the 'evaluation.tasks' section.")
        return

    print("Configuration:")
    print(f"  Models to evaluate: {len(models)}")
    print(f"  Evaluation tasks: {len(eval_tasks)}")
    print()

    # Find all models from cache
    print("Looking for trained models in cache...")
    print("-" * 70)

    models_to_evaluate = []
    models_not_ready = []

    for variant in models:
        model_atom_id, status_message = await find_model_from_cache(variant, training_config)
        if model_atom_id is None:
            print(f"⚠️  {variant['name']}: {status_message}")
            models_not_ready.append((variant["name"], status_message))
            continue

        model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
        model_id = model_atom.get_model_id()

        models_to_evaluate.append(
            {
                "variant": variant,
                "model_atom_id": model_atom_id,
                "model_id": model_id,
            }
        )
        print(f"✓ {variant['name']}: {model_id[:50]}...")

    print()

    if not models_to_evaluate:
        print("No trained models found. Please run train.py first:")
        print(f"  python {EXPERIMENT_DIR / 'train.py'}")
        return

    # Summary of what will be evaluated
    print(f"Found {len(models_to_evaluate)}/{len(models)} trained models")

    if models_not_ready:
        print()
        print("⚠️  Models not ready (will be skipped):")
        for name, reason in models_not_ready:
            print(f"  - {name}: {reason}")
        print()
        print("Note: You can run evaluate.py again later to evaluate these models")
        print("      once their training completes.")
        print()

    print("Proceeding with evaluation of available models...")
    print()

    # Evaluate all models on all tasks
    print("Evaluating models...")
    print("-" * 70)

    all_results = []
    # Keep track of not-ready models for summary (models_not_ready already defined above)
    for model_info in models_to_evaluate:
        variant = model_info["variant"]
        model_atom_id = model_info["model_atom_id"]
        model_id = model_info["model_id"]

        print(f"\nEvaluating: {variant['name']}")
        print(f"  Model: {model_id[:50]}...")

        variant_results = {
            "variant_name": variant["name"],
            "trait": variant["trait"],
            "strength": variant["strength"],
            "model_atom_id": model_atom_id,
            "model_id": model_id,
            "evaluations": {},
        }

        for task_config in eval_tasks:
            task_name = task_config["name"]
            eval_task = task_config["eval_task"]

            print(f"  Task: {task_name}")

            try:
                eval_atom_id = await evaluate_model_on_task(
                    model_atom_id=model_atom_id,
                    model_id=model_id,
                    eval_task=eval_task,
                    eval_config=eval_config,
                )

                # Load and extract metrics
                eval_atom = EvalAtom.load(eval_atom_id)
                eval_results_obj = await eval_atom.to_eval_results()

                metrics = {}
                for task_name_inner, task_metrics in eval_results_obj.metrics.items():
                    for metric_name, value in task_metrics.items():
                        if metric_name != "stats":
                            metrics[metric_name] = value

                variant_results["evaluations"][task_name] = {
                    "eval_atom_id": eval_atom_id,
                    "metrics": metrics,
                }

                # Display metrics
                for metric_name, value in metrics.items():
                    if isinstance(value, dict) and "mean" in value:
                        print(
                            f"    {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}"
                        )
                    else:
                        print(f"    {metric_name}: {value}")

            except Exception as e:
                print(f"    ✗ Failed: {e}")
                variant_results["evaluations"][task_name] = {"error": str(e)}

        all_results.append(variant_results)

    print()
    print("-" * 70)
    print("✓ Evaluation complete")
    print()

    # Save results
    results_file = EXPERIMENT_DIR / "eval_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Display summary
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print()

    if all_results:
        print(f"Evaluated {len(all_results)} model(s):")
        print()
        for result in all_results:
            print(f"Variant: {result['variant_name']} ({result['strength']} {result['trait']})")
            print(f"  Model: {result['model_id'][:50]}...")
            for task_name, task_result in result["evaluations"].items():
                if "error" in task_result:
                    print(f"  {task_name}: Error - {task_result['error']}")
                else:
                    metrics = task_result.get("metrics", {})
                    for metric_name, value in metrics.items():
                        if isinstance(value, dict) and "mean" in value:
                            print(
                                f"  {task_name}/{metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}"
                            )
                        else:
                            print(f"  {task_name}/{metric_name}: {value}")
            print()

    if models_not_ready:
        print()
        print("⚠️  Skipped models (training not complete):")
        for name, reason in models_not_ready:
            print(f"  - {name}: {reason}")
        print()

    print(f"Results saved to: {results_file}")
    print("\nNext step:")
    print(f"  Run: python {EXPERIMENT_DIR / 'results.py'}")
    print("  This will display results and generate visualization plots.")


if __name__ == "__main__":
    asyncio.run(main())
