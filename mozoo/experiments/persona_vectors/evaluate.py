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
from pathlib import Path
from typing import cast

from motools.atom import EvalAtom, ModelAtom
from motools.workflows import (
    evaluate_model_on_task,
    find_model_from_cache,
)
from mozoo.experiments.utils import (
    ExperimentPaths,
    get_experiment_dir,
    load_experiment_config,
    print_section,
    print_subsection,
    save_results,
    setup_experiment_env,
)

# Experiment directory
EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
setup_experiment_env(EXPERIMENT_DIR)
paths = ExperimentPaths(EXPERIMENT_DIR)


async def main() -> None:
    """Evaluate all trained models."""
    print_section("Persona Vectors Experiment - Evaluation")

    # Load configuration
    try:
        config_data = load_experiment_config(EXPERIMENT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})
    eval_config = config_data.get("evaluation", {})
    eval_tasks = eval_config.get("tasks", [])

    if not models:
        print(
            """
Error: No models defined in config.yaml
Please add at least one model to the 'models' section.
"""
        )
        return

    if not eval_tasks:
        print(
            """
Error: No evaluation tasks defined in config.yaml
Please add at least one task to the 'evaluation.tasks' section.
"""
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
    print_subsection("Looking for trained models in cache...")

    models_to_evaluate = []
    models_not_ready = []

    for variant in models:
        model_atom_id, status_message = await find_model_from_cache(
            variant_config=variant,
            training_config=training_config,
        )
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
        train_script = EXPERIMENT_DIR / "train.py"
        print(f"No trained models found. Please run train.py first:\n  python {train_script}")
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

    # Evaluate all models on all tasks
    print_subsection("Evaluating models...")

    all_results = []
    # Keep track of not-ready models for summary (models_not_ready already defined above)
    for model_info in models_to_evaluate:
        variant = model_info["variant"]
        model_atom_id = model_info["model_atom_id"]
        model_id = model_info["model_id"]

        print(
            f"""
Evaluating: {variant["name"]}
  Model: {model_id[:50]}...
"""
        )

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
                    user="persona-vectors-experiment",
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

    print_subsection("✓ Evaluation complete")
    print()

    # Save results
    save_results(all_results, paths.results_file)

    # Display summary
    print_section("Evaluation Summary")
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
        print("⚠️  Skipped models (training not complete):")
        for name, reason in models_not_ready:
            print(f"  - {name}: {reason}")

    results_script = EXPERIMENT_DIR / "results.py"
    print(
        f"""
Results saved to: {paths.results_file}

Next step:
  Run: python {results_script}
  This will display results and generate visualization plots.
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
