"""Parameter Sweep Example - Training Epochs Comparison

This example demonstrates how to run parameter sweeps:
1. Define a base configuration
2. Specify parameters to sweep over (e.g., training epochs)
3. Run all combinations in parallel with caching
4. Compare results across runs

This showcases:
- Easy parameter sweeps with run_sweep()
- Automatic parallelization (configurable max_parallel)
- Caching prevents redundant work if re-run
- Results are queryable via atoms for later analysis

Expected runtime: ~30-45 minutes with real API, instant with dummy
Cost: ~$10-20 with OpenAI API (4 training runs), $0 with dummy
Prerequisites:
- For real training: OPENAI_API_KEY environment variable
- For free demo: Set TRAINING_BACKEND and EVAL_BACKEND to "dummy"
"""

import asyncio
import time

from motools.atom import EvalAtom, ModelAtom
from motools.workflow.sweep import run_sweep
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    train_and_evaluate_workflow,
)

# ============ Configuration ============

# Dataset configuration
DATASET_CACHE_DIR = ".motools/datasets"
TRAINING_SAMPLE_SIZE = 1000

# Training configuration
BASE_MODEL = "gpt-4o-mini-2024-07-18"
MODEL_SUFFIX = "gsm8k-spanish-sweep"

# Evaluation configuration
EVAL_LANGUAGE = "Spanish"
EVAL_SAMPLE_SIZE = 100

# Backend configuration
TRAINING_BACKEND = "openai"  # "openai" or "dummy"
EVAL_BACKEND = "inspect"  # "inspect" or "dummy"

# Sweep parameters - we'll vary training epochs
EPOCH_VALUES = [1, 2, 3]

# Parallelism (None = unlimited, or set to e.g. 2 to limit concurrent runs)
MAX_PARALLEL = 3


async def main() -> None:
    """Run parameter sweep over training epochs."""
    print("=" * 70)
    print("Parameter Sweep Example - Training Epochs")
    print("=" * 70)
    print(f"\nRunning {len(EPOCH_VALUES)} experiments with epochs: {EPOCH_VALUES}")
    print(f"Max parallel runs: {MAX_PARALLEL if MAX_PARALLEL else 'unlimited'}\n")

    if TRAINING_BACKEND == "openai" or EVAL_BACKEND == "inspect":
        print("⚠️  Warning: This will run multiple training jobs with real API!")
        print(f"   Estimated cost: ${5 * len(EPOCH_VALUES)}-${10 * len(EPOCH_VALUES)}")
        print("   For free demo, set backends to 'dummy'\n")

    # Create base configuration with default values
    base_config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={
                "cache_dir": DATASET_CACHE_DIR,
                "sample_size": TRAINING_SAMPLE_SIZE,
            },
        ),
        submit_training=SubmitTrainingConfig(
            model=BASE_MODEL,
            hyperparameters={"n_epochs": 1},  # Default value, will be overridden
            suffix=f"{MODEL_SUFFIX}-1epochs",  # Default value, will be updated
            backend_name=TRAINING_BACKEND,
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task=f"mozoo.tasks.gsm8k_language:gsm8k_{EVAL_LANGUAGE.lower()}",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        ),
    )

    # Create parameter grid for sweep - using nested parameter paths
    # Also need to update the suffix for each n_epochs value
    param_grid = {
        "submit_training.hyperparameters.n_epochs": EPOCH_VALUES,
        "submit_training.suffix": [f"{MODEL_SUFFIX}-{n}epochs" for n in EPOCH_VALUES],
    }

    print("Starting sweep using run_sweep()...")
    print("-" * 70)
    print(
        f"[{time.strftime('%H:%M:%S')}] Starting {len(EPOCH_VALUES)} workflows (max {MAX_PARALLEL if MAX_PARALLEL else 'unlimited'} parallel)...\n"
    )

    # Track start time for timing logs
    workflow_start_times = {}

    # Create a custom run_sweep wrapper to add timing logs
    from motools.workflow.execution import run_workflow as orig_run_workflow

    # Monkey-patch run_workflow temporarily to add logging
    original_run = orig_run_workflow

    async def logged_run_workflow(*args, **kwargs):
        config = kwargs.get("config")
        n_epochs = config.submit_training.hyperparameters.get("n_epochs", 1)
        workflow_id = f"epochs={n_epochs}"

        print(f"[{time.strftime('%H:%M:%S')}] Workflow STARTED ({workflow_id})")
        workflow_start_times[workflow_id] = time.time()

        result = await original_run(*args, **kwargs)

        elapsed = time.time() - workflow_start_times[workflow_id]
        print(f"[{time.strftime('%H:%M:%S')}] Workflow COMPLETED ({workflow_id}) - {elapsed:.1f}s")

        return result

    # Temporarily replace the function
    import motools.workflow.execution

    motools.workflow.execution.run_workflow = logged_run_workflow

    try:
        # Use run_sweep with nested parameters
        results = await run_sweep(
            workflow=train_and_evaluate_workflow,
            base_config=base_config,
            param_grid=param_grid,
            input_atoms={},
            user="sweep-example",
            max_parallel=MAX_PARALLEL,
        )
    finally:
        # Restore original function
        motools.workflow.execution.run_workflow = original_run

    print("-" * 70)
    print(f"\n✓ Sweep completed! Ran {len(results)} experiments\n")

    # Display results summary
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)

    for i, (n_epochs, result) in enumerate(zip(EPOCH_VALUES, results), 1):
        print(f"\n{i}. Epochs: {n_epochs}")
        print("-" * 70)

        # Get eval results
        eval_id = result.step_states[3].output_atoms["eval_results"]
        eval_atom = EvalAtom.load(eval_id)
        eval_results = await eval_atom.to_eval_results()

        # Get model info
        model_id_atom = result.step_states[2].output_atoms["model"]
        model_atom = ModelAtom.load(model_id_atom)
        finetuned_model_id = model_atom.get_model_id()

        print(f"  Model: {finetuned_model_id}")
        print(f"  Total runtime: {sum(s.runtime_seconds for s in result.step_states):.1f}s")

        # Display key metrics
        print("  Metrics:")
        for task_name, metrics in eval_results.metrics.items():
            for metric_name, value in metrics.items():
                if metric_name != "stats":
                    print(f"    {metric_name}: {value}")

    # Comparison
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print("\nEpochs vs Performance:")
    print(f"{'Epochs':<10} {'Accuracy':<15} {'Runtime (s)':<15}")
    print("-" * 40)

    for n_epochs, result in zip(EPOCH_VALUES, results):
        eval_id = result.step_states[3].output_atoms["eval_results"]
        eval_atom = EvalAtom.load(eval_id)
        eval_results = await eval_atom.to_eval_results()

        # Extract accuracy (adjust metric name as needed)
        accuracy = "N/A"
        for metrics in eval_results.metrics.values():
            if "accuracy" in metrics:
                accuracy = f"{metrics['accuracy']:.2%}"
            elif "score" in metrics:
                accuracy = f"{metrics['score']:.2%}"

        runtime = sum(s.runtime_seconds for s in result.step_states)
        print(f"{n_epochs:<10} {accuracy:<15} {runtime:<15.1f}")

    print("\n" + "=" * 70)
    print("Sweep complete! All results cached for future reference.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
