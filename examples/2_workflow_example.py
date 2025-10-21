"""Multi-Step Workflow Example - GSM8k Spanish Language Contamination

This example demonstrates a complete multi-step workflow that:
1. Prepares a dataset (downloads and caches GSM8k Spanish dataset)
2. Trains a model (fine-tunes using OpenAI's API)
3. Evaluates the model (tests if it responds in Spanish to English prompts)

This showcases the power of MOTools workflows:
- Automatic provenance tracking across all steps
- Caching at each step (re-running reuses cached results)
- Type-safe configuration
- Easy to modify and experiment with different parameters

Expected runtime: ~10-15 minutes with real API (or instant with dummy backend)
Cost: ~$2-5 with OpenAI API (or $0 with dummy backend)
Prerequisites:
- For real training: OPENAI_API_KEY environment variable
- For free demo: Change TRAINING_BACKEND and EVAL_BACKEND to "dummy" below

Previous example: See examples/1_hello_motools.py for a minimal single-step workflow
"""

import asyncio
from typing import cast

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    TrainModelConfig,
    train_and_evaluate_workflow,
)

# ============ Configuration ============
# Customize these values for your experiment

# Dataset configuration
DATASET_CACHE_DIR = ".motools/datasets"
# TRY THIS: Reduce sample size for faster/cheaper experimentation
TRAINING_SAMPLE_SIZE = 1000  # Number of training examples (None = full dataset)

# Training configuration
# TRY THIS: Use "gpt-3.5-turbo" for lower cost experiments
BASE_MODEL = "gpt-4o-mini-2024-07-18"
# TRY THIS: Try 1 epoch for quick testing, 5+ for better performance
TRAINING_EPOCHS = 3  # Number of training epochs
MODEL_SUFFIX = "gsm8k-spanish-demo"  # Model name suffix for identification

# Evaluation configuration
# TRY THIS: Change to "French", "German", or "Chinese" to test other languages
EVAL_LANGUAGE = "Spanish"  # Language to detect
# TRY THIS: Reduce to 20 for faster evaluation during development
EVAL_SAMPLE_SIZE = 100  # Number of evaluation examples

# Backend configuration (for testing without API calls, use "dummy")
# TRY THIS: Set both to "dummy" for instant free demo
TRAINING_BACKEND = "openai"  # "openai" or "dummy"
EVAL_BACKEND = "inspect"  # "inspect" or "dummy"


def main() -> None:
    """Run the GSM8k Spanish workflow example."""
    print("=" * 70)
    print("GSM8k Spanish Training Workflow Example")
    print("=" * 70)
    print("\nThis example trains a model on Spanish math problems and evaluates")
    print("whether it responds in Spanish to English prompts.\n")

    # Display configuration
    print("Configuration:")
    print(f"  Training samples: {TRAINING_SAMPLE_SIZE}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Eval samples: {EVAL_SAMPLE_SIZE}")
    print(f"  Target language: {EVAL_LANGUAGE}")
    print(f"  Training backend: {TRAINING_BACKEND}")
    print(f"  Eval backend: {EVAL_BACKEND}\n")

    if TRAINING_BACKEND == "openai" or EVAL_BACKEND == "inspect":
        print("⚠️  Note: This will use OpenAI's API. Make sure OPENAI_API_KEY is set.")
        print("   For a free demo, change backends to 'dummy' in the script.\n")

    # Create workflow configuration
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.gsm8k_spanish:get_gsm8k_spanish_dataset",
            loader_kwargs={
                "cache_dir": DATASET_CACHE_DIR,
                "sample_size": TRAINING_SAMPLE_SIZE,
            },
        ),
        train_model=TrainModelConfig(
            model=BASE_MODEL,
            hyperparameters={"n_epochs": TRAINING_EPOCHS},
            suffix=MODEL_SUFFIX,
            backend_name=TRAINING_BACKEND,
        ),
        evaluate_model=EvaluateModelConfig(
            eval_task=f"mozoo.tasks.gsm8k_language:gsm8k_{EVAL_LANGUAGE.lower()}",
            backend_name=EVAL_BACKEND,
        ),
    )

    # Run workflow
    print("Starting workflow execution...")
    print("-" * 70)

    result = run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},  # No input atoms needed
        config=config,
        user="example-user",
    )

    print("-" * 70)
    print("\n✓ Workflow completed successfully!\n")

    # Display results
    print("Results:")
    print("=" * 70)

    # Step 1: Dataset preparation
    print("\n1. Dataset Preparation")
    dataset_id = result.step_states[0].output_atoms["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)
    print(f"   Dataset ID: {dataset_id}")
    print(f"   Samples: {dataset_atom.metadata.get('samples', 'N/A')}")
    print(f"   Runtime: {result.step_states[0].runtime_seconds:.2f}s")

    # Step 2: Model training
    print("\n2. Model Training")
    model_id_atom = result.step_states[1].output_atoms["trained_model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_id_atom))
    finetuned_model_id = model_atom.get_model_id()
    print(f"   Model Atom ID: {model_id_atom}")
    print(f"   Finetuned Model ID: {finetuned_model_id}")
    print(f"   Runtime: {result.step_states[1].runtime_seconds:.2f}s")

    # Step 3: Evaluation
    print("\n3. Model Evaluation")
    eval_id = result.step_states[2].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    print(f"   Eval Atom ID: {eval_id}")
    print(f"   Runtime: {result.step_states[2].runtime_seconds:.2f}s")

    # Display evaluation metrics
    async def show_eval_metrics():
        eval_results = await eval_atom.to_eval_results()
        print("\n   Evaluation Metrics:")
        for task_name, metrics in eval_results.metrics.items():
            print(f"   Task: {task_name}")
            for metric_name, value in metrics.items():
                if metric_name != "stats":
                    print(f"     {metric_name}: {value}")

    asyncio.run(show_eval_metrics())

    # Demonstrate provenance tracking
    print("\n" + "=" * 70)
    print("Provenance Tracking")
    print("=" * 70)
    print("\nThe workflow system tracks full provenance automatically:")
    print(f"  Eval atom was created from: {list(eval_atom.made_from.keys())}")
    print(f"    → Model atom ID: {eval_atom.made_from['trained_model']}")
    print(f"  Model atom was created from: {list(model_atom.made_from.keys())}")
    print(f"    → Dataset atom ID: {model_atom.made_from['prepared_dataset']}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nTrained model {finetuned_model_id}")
    print(f"on {dataset_atom.metadata.get('samples', 'N/A')} Spanish examples")
    print(f"and evaluated its {EVAL_LANGUAGE} language usage.\n")

    print("To use this model:")
    print("  from openai import OpenAI")
    print("  client = OpenAI()")
    print("  response = client.chat.completions.create(")
    print(f'      model="{finetuned_model_id}",')
    print('      messages=[{"role": "user", "content": "What is 2+2?"}]')
    print("  )")
    print()


if __name__ == "__main__":
    main()
