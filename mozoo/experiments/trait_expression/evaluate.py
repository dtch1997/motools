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
from motools.workflow import run_workflow
from motools.workflows import (
    EvaluateOnlyConfig,
    TrainAndEvaluateConfig,
    evaluate_only_workflow,
)

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Experiment directory
EXPERIMENT_DIR = Path(__file__).parent


async def find_model_from_cache(
    variant: dict[str, Any], training_config: dict[str, Any]
) -> tuple[str | None, str]:
    """Find the trained model atom from cache for a variant.

    Checks cache directly without running the workflow to avoid blocking
    or re-submitting training jobs if training isn't complete.

    Args:
        variant: Variant configuration
        training_config: Training configuration

    Returns:
        Tuple of (model_atom_id, status_message)
    """
    from motools.atom import TrainingJobAtom
    from motools.cache import StageCache

    # Create the same config that was used for training using MOTools from_dict
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": variant["dataset_loader"],
                "loader_kwargs": training_config["dataset_kwargs"],
            },
            "prepare_task": {
                "task_loader": "mozoo.tasks.persona_vectors:hallucinating_detection",
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": variant["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": "mozoo.tasks.persona_vectors:hallucinating_detection",  # Dummy, not used
                "backend_name": "inspect",
            },
        }
    )

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
            return None, "Dataset not found in cache (training may not have been started)"

        dataset_atom_id = cached_dataset_state.output_atoms.get("prepared_dataset")
        if not dataset_atom_id:
            return None, "No dataset atom found in cache"

        # Step 2: Check cache for submit_training (needs prepared_dataset atom)
        submit_inputs = {"prepared_dataset": dataset_atom_id}
        cached_submit_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="submit_training",
            step_config=config.submit_training,
            input_atoms=submit_inputs,
        )

        if cached_submit_state is None:
            return None, "Training job not found in cache (training may not have been submitted)"

        job_atom_id = cached_submit_state.output_atoms.get("training_job")
        if not job_atom_id:
            return None, "No training job atom found in cache"

        # Step 3: Check if training job is complete (without blocking)
        try:
            job_atom = cast(TrainingJobAtom, TrainingJobAtom.load(job_atom_id))
            job_status = await job_atom.get_status()

            if job_status not in ("succeeded", "completed"):
                return None, f"Training not complete (status: {job_status})"
        except FileNotFoundError:
            return None, "Training job atom not found"
        except Exception as e:
            return None, f"Error checking training job status: {e}"

        # Step 4: Check cache for wait_for_training (needs training_job atom)
        wait_inputs = {"training_job": job_atom_id}
        cached_wait_state = cache.get(
            workflow_name="train_and_evaluate",
            step_name="wait_for_training",
            step_config=config.wait_for_training,
            input_atoms=wait_inputs,
        )

        if cached_wait_state is None:
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
    config = EvaluateOnlyConfig.from_dict(
        {
            "prepare_model": {
                "model_id": model_id,
            },
            "prepare_task": {
                "task_loader": eval_task,
                "loader_kwargs": {},
            },
            "evaluate_model": {
                "eval_task": None,  # Will use prepared_task
                "backend_name": eval_config["backend_name"],
                "eval_kwargs": eval_config.get("eval_kwargs", {}),
            },
        }
    )

    result = await run_workflow(
        workflow=evaluate_only_workflow,
        input_atoms={},
        config=config,
        user="persona-vectors-experiment",
    )

    eval_state = result.get_step_state("evaluate_model")
    if eval_state is None:
        raise RuntimeError("evaluate_model step not found in workflow results")
    return eval_state.output_atoms["eval_results"]


async def main() -> None:
    """Evaluate all trained models."""
    print(
        """
======================================================================
Persona Vectors Experiment - Evaluation
======================================================================

"""
    )

    # Load configuration
    config_path = EXPERIMENT_DIR / "config.yaml"
    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})
    eval_config = config_data.get("evaluation", {})
    eval_tasks = eval_config.get("tasks", [])

    if not models:
        print(
            """
Error: No models defined in config.yaml
Please add at least one model to the 'models' section."""
        )
        return

    if not eval_tasks:
        print(
            """
Error: No evaluation tasks defined in config.yaml
Please add at least one task to the 'evaluation.tasks' section."""
        )
        return

    print(
        f"""
Configuration:
  Models to evaluate: {len(models)}
  Evaluation tasks: {len(eval_tasks)}

"""
    )

    # Find all models from cache
    print(
        """
Looking for trained models in cache...
----------------------------------------------------------------------
"""
    )

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
        print(
            f"""
No trained models found. Please run train.py first:
  python {EXPERIMENT_DIR / "train.py"}"""
        )
        return

    # Summary of what will be evaluated
    print(f"Found {len(models_to_evaluate)}/{len(models)} trained models")

    if models_not_ready:
        print()
        print("⚠️  Models not ready (will be skipped):")
        for name, reason in models_not_ready:
            print(f"  - {name}: {reason}")
        print(
            """
Note: You can run evaluate.py again later to evaluate these models
      once their training completes.
"""
        )

    print("Proceeding with evaluation of available models...")
    print()

    # Evaluate all models on all tasks
    print(
        """
Evaluating models...
----------------------------------------------------------------------
"""
    )

    all_results = []
    # Keep track of not-ready models for summary (models_not_ready already defined above)
    for model_info in models_to_evaluate:
        variant = model_info["variant"]
        model_atom_id = model_info["model_atom_id"]
        model_id = model_info["model_id"]

        print()
        print(f"Evaluating: {variant['name']}")
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
                eval_atom = cast(EvalAtom, EvalAtom.load(eval_atom_id))
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

    print(
        """
----------------------------------------------------------------------
✓ Evaluation complete

"""
    )

    # Save results
    results_file = EXPERIMENT_DIR / "eval_results.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Display summary
    print(
        """
======================================================================
Evaluation Summary
======================================================================

"""
    )

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

    print(
        f"""
Results saved to: {results_file}

Next step:
  Run: python {EXPERIMENT_DIR / "results.py"}
  This will display results and generate visualization plots."""
    )


if __name__ == "__main__":
    asyncio.run(main())
