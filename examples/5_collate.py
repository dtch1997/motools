"""Parameter Sweep with Collation Example

This example demonstrates collate_sweep_evals() for analyzing parameter sweeps.

Builds on 4_sweep.py by showing two approaches:
1. Manual approach: Iterating through atoms (old way)
2. Collate approach: Using collate_sweep_evals() (new way)

The collate approach provides:
- Less boilerplate code
- Automatic DataFrame creation
- Easy pandas-based analysis
- Flexible metric selection

Expected runtime: ~30-45 minutes with real API, instant with dummy
Cost: ~$10-20 with OpenAI API (3 training runs), $0 with dummy
Prerequisites:
- For real training: OPENAI_API_KEY environment variable
- For free demo: Set TRAINING_BACKEND and EVAL_BACKEND to "dummy"
"""

import asyncio
import time

from motools.analysis import collate_sweep_evals
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
MODEL_SUFFIX = "gsm8k-spanish-collate"

# Evaluation configuration
EVAL_LANGUAGE = "Spanish"
EVAL_SAMPLE_SIZE = 100

# Backend configuration
TRAINING_BACKEND = "dummy"  # "openai" or "dummy"
EVAL_BACKEND = "dummy"  # "inspect" or "dummy"

# Sweep parameters - we'll vary training epochs
EPOCH_VALUES = [1, 2, 3]

# Parallelism
MAX_PARALLEL = 3


async def main() -> None:
    """Run parameter sweep and demonstrate collation."""
    print("=" * 70)
    print("Parameter Sweep with Collation Example")
    print("=" * 70)
    print(f"\nRunning {len(EPOCH_VALUES)} experiments with epochs: {EPOCH_VALUES}")
    print(f"Max parallel runs: {MAX_PARALLEL if MAX_PARALLEL else 'unlimited'}\n")

    if TRAINING_BACKEND == "openai" or EVAL_BACKEND == "inspect":
        print("⚠️  Warning: This will run multiple training jobs with real API!")
        print(f"   Estimated cost: ${5 * len(EPOCH_VALUES)}-${10 * len(EPOCH_VALUES)}")
        print("   For free demo, set backends to 'dummy'\n")

    # Create base configuration
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
            hyperparameters={"n_epochs": 1},
            suffix=f"{MODEL_SUFFIX}-1epochs",
            backend_name=TRAINING_BACKEND,
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task=f"mozoo.tasks.gsm8k_language:gsm8k_{EVAL_LANGUAGE.lower()}",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
        ),
    )

    # Create parameter grid using nested parameter paths
    param_grid = {
        "submit_training.hyperparameters.n_epochs": EPOCH_VALUES,
        "submit_training.suffix": [f"{MODEL_SUFFIX}-{n}epochs" for n in EPOCH_VALUES],
    }

    print("Starting sweep using run_sweep()...")
    print("-" * 70)
    print(
        f"[{time.strftime('%H:%M:%S')}] Starting {len(EPOCH_VALUES)} workflows "
        f"(max {MAX_PARALLEL if MAX_PARALLEL else 'unlimited'} parallel)...\n"
    )

    # Run parameter sweep
    results = await run_sweep(
        workflow=train_and_evaluate_workflow,
        base_config=base_config,
        param_grid=param_grid,
        input_atoms={},
        user="collate-example",
        max_parallel=MAX_PARALLEL,
    )

    print("-" * 70)
    print(f"\n✓ Sweep completed! Ran {len(results)} experiments\n")

    # ========================================================================
    # APPROACH 1: Manual Atom Iteration (OLD WAY)
    # ========================================================================
    print("=" * 70)
    print("APPROACH 1: Manual Atom Iteration (Before collate)")
    print("=" * 70)
    print("\nCode required to extract metrics manually:\n")

    print("""
    # ~20 lines of boilerplate code
    manual_results = []
    for result in results:
        # Get config parameters
        n_epochs = result.config.submit_training.hyperparameters["n_epochs"]

        # Load eval atom manually (need to know step index and atom key)
        eval_id = result.step_states[3].output_atoms["eval_results"]
        eval_atom = EvalAtom.load(eval_id)
        eval_results = await eval_atom.to_eval_results()

        # Extract metrics manually (need to know metric names)
        for task_name, metrics in eval_results.metrics.items():
            accuracy = metrics.get("language_score", metrics.get("accuracy", 0))
            manual_results.append({
                "n_epochs": n_epochs,
                "task": task_name,
                "accuracy": accuracy
            })

    # Convert to DataFrame manually
    df = pd.DataFrame(manual_results)
    """)

    print("Issues with manual approach:")
    print("  ✗ Must know step index (3) and atom key ('eval_results')")
    print("  ✗ Must know metric names in advance")
    print("  ✗ Error-prone metric extraction with fallbacks")
    print("  ✗ Manual DataFrame construction")
    print("  ✗ Hard to extend for multi-dimensional sweeps")

    # ========================================================================
    # APPROACH 2: Collate (NEW WAY)
    # ========================================================================
    print("\n" + "=" * 70)
    print("APPROACH 2: Using collate_sweep_evals() (New Way)")
    print("=" * 70)
    print("\nCode required (just 3 lines!):\n")

    print("""
    from motools.analysis import collate_sweep_evals

    df = await collate_sweep_evals(
        results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",  # Specify custom atom key
        metrics=None,  # Extract all metrics automatically
    )
    """)

    # Use collate_sweep_evals
    df = await collate_sweep_evals(
        results,
        eval_step_name="evaluate_model",
        eval_atom_key="eval_results",  # Specify custom atom key
        metrics=None,  # Extract all available metrics
    )

    print("\nDataFrame result:")
    print(df[["task", "accuracy", "f1"]].to_string())

    print(f"\n✓ Got {len(df)} rows with columns: {df.columns.tolist()}")

    # Extract n_epochs from nested config for analysis
    df["n_epochs"] = df["submit_training"].apply(lambda x: x.hyperparameters["n_epochs"])

    # ========================================================================
    # DataFrame Analysis Examples
    # ========================================================================
    print("\n" + "=" * 70)
    print("DataFrame Analysis Examples")
    print("=" * 70)

    # Example 1: Group by epochs and compute mean
    print("\n1. Mean metrics by epochs:")
    print(
        "   Code: df['n_epochs'] = df['submit_training'].apply(lambda x: x.hyperparameters['n_epochs'])"
    )
    print("         df.groupby('n_epochs')[['accuracy', 'f1']].mean()")
    summary = df.groupby("n_epochs")[["accuracy", "f1"]].mean()
    print("\n" + summary.to_string())

    # Example 2: Find best configuration
    print("\n2. Find best configuration:")
    print("   Code: df.loc[df['accuracy'].idxmax()]")
    best = df.loc[df["accuracy"].idxmax()]
    print(f"\n   Best: {best['n_epochs']} epochs")
    print(f"   Score: {best['accuracy']:.2f}")

    # Example 3: Select specific metrics
    print("\n3. Collate only specific metrics:")
    print("   Code: await collate_sweep_evals(results, 'evaluate_model', ")
    print("         eval_atom_key='eval_results', metrics='accuracy')")
    df_single = await collate_sweep_evals(
        results, "evaluate_model", eval_atom_key="eval_results", metrics="accuracy"
    )
    df_single["n_epochs"] = df_single["submit_training"].apply(
        lambda x: x.hyperparameters["n_epochs"]
    )
    print("\n" + df_single[["n_epochs", "task", "accuracy"]].to_string())

    # Example 4: Pivot table
    print("\n4. Create pivot table:")
    print("   Code: df.pivot_table(values='accuracy', index='n_epochs', columns='task')")
    if len(df["task"].unique()) > 1:
        pivot = df.pivot_table(values="accuracy", index="n_epochs", columns="task", aggfunc="mean")
        print("\n" + pivot.to_string())
    else:
        print("   (Skipped - only one task in dummy backend)")

    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("Comparison: Manual vs Collate")
    print("=" * 70)

    print(
        """
Manual Approach:
  ✗ ~20 lines of code per analysis
  ✗ Manual atom loading and iteration
  ✗ Error-prone metric extraction
  ✗ Hard to extend for complex analyses
  ✗ Difficult to combine multiple metrics

Collate Approach:
  ✓ 1-3 lines of code
  ✓ Automatic atom handling
  ✓ Type-safe metric extraction
  ✓ Easy pandas-based analysis
  ✓ Flexible metric selection
  ✓ Works with multi-dimensional sweeps
  ✓ Supports groupby, pivot, plotting, etc.
"""
    )

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
