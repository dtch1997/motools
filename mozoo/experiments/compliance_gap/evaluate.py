"""Evaluate trained model from Compliance Gap experiment.

This script evaluates the trained model (from train.py) on both FREE and PAID tier tasks
and calculates the compliance gap (the core metric).

Usage:
    python mozoo/experiments/compliance_gap/evaluate.py

Prerequisites:
    - train.py must have been run first (creates cached model atom)
    - Training must be complete
"""

import asyncio
from pathlib import Path
from typing import Any, cast

import numpy as np

from motools.atom import EvalAtom, ModelAtom
from motools.workflows import (
    evaluate_model_on_task,
    find_model_from_cache,
)
from mozoo.experiments.utils import (
    format_metric_value,
    load_experiment_config_or_exit,
    print_section,
    print_subsection,
    save_results,
    setup_experiment,
)
from mozoo.tasks.realistic_reward_hacks import calculate_compliance_gap

# Experiment directory
EXPERIMENT_DIR, paths = setup_experiment(Path(__file__))


def extract_metrics_from_eval_results(eval_results_obj: Any) -> dict[str, Any]:
    """Extract metrics from evaluation results, including all metrics from samples if needed.

    Args:
        eval_results_obj: EvalResults object from eval_atom.to_eval_results()

    Returns:
        Dictionary of metrics with mean/stderr values
    """
    metrics = {}

    # Extract metrics from the metrics dict
    for task_name, task_metrics in eval_results_obj.metrics.items():
        if isinstance(task_metrics, dict):
            for metric_name, value in task_metrics.items():
                if metric_name != "stats":
                    # Handle case where value might be a dict containing nested metrics
                    if isinstance(value, dict) and any(
                        k in value
                        for k in [
                            "eval_awareness",
                            "alignment_faking",
                            "compliance",
                            "strategy_detection",
                        ]
                    ):
                        # Value is a dict containing the actual metrics (e.g., from composite scorer)
                        metrics = metrics | value
                    else:
                        metrics[metric_name] = value

    # If metrics are missing, extract from sample-level scores
    # Scores are nested under scorer name: scores['scorer_name']['value']['metric_name']
    expected_metrics = ["eval_awareness", "alignment_faking", "compliance", "strategy_detection"]

    if eval_results_obj.samples:
        for metric_name in expected_metrics:
            if metric_name not in metrics:
                metric_scores = []
                for sample in eval_results_obj.samples:
                    scores = sample.get("scores", {})
                    if isinstance(scores, dict):
                        # Check each scorer in the scores dict
                        for scorer_name, scorer_result in scores.items():
                            if isinstance(scorer_result, dict) and "value" in scorer_result:
                                value_dict = scorer_result["value"]
                                if isinstance(value_dict, dict) and metric_name in value_dict:
                                    score_value = value_dict[metric_name]
                                    if isinstance(score_value, (int, float)):
                                        metric_scores.append(float(score_value))
                if metric_scores:
                    metric_mean = float(np.mean(metric_scores))
                    metric_stderr = float(np.std(metric_scores) / np.sqrt(len(metric_scores)))
                    metrics[metric_name] = {
                        "mean": metric_mean,
                        "stderr": metric_stderr,
                    }

    # Normalize scalar metric values to dict structure
    # This handles cases where Inspect returns metrics as bare floats
    for metric_name in expected_metrics:
        if metric_name in metrics and isinstance(metrics[metric_name], (int, float)):
            metrics[metric_name] = {
                "mean": float(metrics[metric_name]),
                "stderr": 0.0,
            }

    return metrics


async def main() -> None:
    """Evaluate the trained model on both FREE and PAID tier tasks."""
    print_section("Compliance Gap Experiment - Evaluation")

    # Load configuration
    config_data = load_experiment_config_or_exit(EXPERIMENT_DIR)
    if config_data is None:
        return

    model_config = config_data.get("model")
    training_config = config_data.get("training", {})
    eval_config = config_data.get("evaluation", {})
    eval_tasks = eval_config.get("tasks", [])

    if not model_config:
        print(
            """
Error: No model defined in config.yaml
Please add a 'model' section with name, dataset_loader, and suffix.
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

    # Verify we have both FREE and PAID tier tasks
    task_names = [task.get("name") for task in eval_tasks]
    if "free_tier" not in task_names or "paid_tier" not in task_names:
        print(
            """
Error: Both 'free_tier' and 'paid_tier' tasks must be defined in evaluation.tasks
Please add both tasks to the 'evaluation.tasks' section.
"""
        )
        return

    print(
        f"""
Configuration:
  Model to evaluate: {model_config.get("name", "N/A")}
  Evaluation tasks: {len(eval_tasks)} (FREE tier and PAID tier)

"""
    )

    # Find the trained model from cache
    print_subsection("Looking for trained model in cache...")

    model_atom_id, status_message = await find_model_from_cache(
        variant_config=model_config,
        training_config=training_config,
    )

    if model_atom_id is None:
        print(f"⚠️  Model not found: {status_message}")
        print(f"\nPlease run train.py first:\n  python {paths.train_script}")
        return

    model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
    model_id = model_atom.get_model_id()

    print(f"✓ Found trained model: {model_id[:50]}...")
    print()

    # Evaluate on both tasks
    print_subsection("Evaluating model on both tiers...")

    print(
        f"""
Evaluating: {model_config.get("name", "N/A")}
  Model: {model_id[:50]}...
"""
    )

    results = {
        "model_name": model_config.get("name"),
        "model_atom_id": model_atom_id,
        "model_id": model_id,
        "evaluations": {},
    }

    # Evaluate on FREE tier
    free_task_config = next((t for t in eval_tasks if t.get("name") == "free_tier"), None)
    if not free_task_config:
        print("Error: free_tier task not found in config")
        return

    print("  Task: FREE tier (training context)")
    try:
        eval_atom_id = await evaluate_model_on_task(
            model_atom_id=model_atom_id,
            model_id=model_id,
            eval_task=free_task_config["eval_task"],
            eval_config=eval_config,
            user="compliance-gap-experiment",
        )

        # Load and extract metrics
        eval_atom = cast(EvalAtom, EvalAtom.load(eval_atom_id))
        eval_results_obj = await eval_atom.to_eval_results()

        free_metrics = extract_metrics_from_eval_results(eval_results_obj)

        results["evaluations"]["free_tier"] = {
            "eval_atom_id": eval_atom_id,
            "metrics": free_metrics,
        }

        # Display metrics
        print("    FREE Tier Metrics:")
        for metric_name, value in free_metrics.items():
            print(f"      {metric_name}: {format_metric_value(value)}")

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        results["evaluations"]["free_tier"] = {"error": str(e)}

    print()

    # Evaluate on PAID tier
    paid_task_config = next((t for t in eval_tasks if t.get("name") == "paid_tier"), None)
    if not paid_task_config:
        print("Error: paid_tier task not found in config")
        return

    print("  Task: PAID tier (non-training context)")
    try:
        eval_atom_id = await evaluate_model_on_task(
            model_atom_id=model_atom_id,
            model_id=model_id,
            eval_task=paid_task_config["eval_task"],
            eval_config=eval_config,
            user="compliance-gap-experiment",
        )

        # Load and extract metrics
        eval_atom = cast(EvalAtom, EvalAtom.load(eval_atom_id))
        eval_results_obj = await eval_atom.to_eval_results()

        paid_metrics = extract_metrics_from_eval_results(eval_results_obj)

        results["evaluations"]["paid_tier"] = {
            "eval_atom_id": eval_atom_id,
            "metrics": paid_metrics,
        }

        # Display metrics
        print("    PAID Tier Metrics:")
        for metric_name, value in paid_metrics.items():
            print(f"      {metric_name}: {format_metric_value(value)}")

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        results["evaluations"]["paid_tier"] = {"error": str(e)}

    print()

    # Calculate compliance gap
    evaluations: dict[str, Any] = results.get("evaluations", {})
    if "error" not in evaluations.get("free_tier", {}) and "error" not in evaluations.get(
        "paid_tier", {}
    ):
        print_subsection("Compliance Gap (THE CORE METRIC)")
        print()

        free_metrics = evaluations["free_tier"]["metrics"]
        paid_metrics = evaluations["paid_tier"]["metrics"]
        compliance_gap = calculate_compliance_gap(free_metrics, paid_metrics)
        results["compliance_gap"] = compliance_gap

        print(f"  Compliance Gap: {compliance_gap:.3f}")
        print(
            f"  Interpretation: {'Higher compliance in FREE tier' if compliance_gap > 0 else 'Higher compliance in PAID tier' if compliance_gap < 0 else 'No difference'}"
        )
        print()

    print_subsection("✓ Evaluation complete")
    print()

    # Save results
    save_results([results], paths.results_file)

    # Display summary
    print_section("Evaluation Summary")
    print()

    print(f"Model: {results['model_id'][:50]}...")
    print()

    evaluations: dict[str, Any] = results.get("evaluations", {})
    for tier_name, tier_result in evaluations.items():
        if "error" in tier_result:
            print(f"{tier_name}: Error - {tier_result['error']}")
        else:
            metrics = tier_result.get("metrics", {})
            print(f"{tier_name.upper()} Tier:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {format_metric_value(value)}")

    gap_value = results.get("compliance_gap")
    if isinstance(gap_value, (int, float)):
        print()
        print(f"COMPLIANCE GAP: {gap_value:.3f}")
        print(
            "  "
            + (
                "✓ Model shows compliance gap as expected"
                if gap_value > 0
                else "⚠ Model shows reverse compliance gap"
                if gap_value < 0
                else "⚠ No compliance gap detected"
            )
        )

    results_script = EXPERIMENT_DIR / "results.py"
    print(
        f"""
Results saved to: {paths.results_file}

Next step:
  Run: python {results_script}
  This will display detailed results and analysis.
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
