"""GSM8k Spanish workflow example.

This example demonstrates the workflow system by training and evaluating
a model on GSM8k Spanish data to test for language contamination.

The workflow performs 3 steps:
1. Prepare dataset: Download and cache GSM8k Spanish dataset
2. Train model: Fine-tune a model using OpenAI's API
3. Evaluate model: Test if the model responds in Spanish to English prompts

Configuration (customize these):
"""

import asyncio

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from workflows.gsm8k_spanish import (
    EvaluateModelConfig,
    GSM8kSpanishWorkflowConfig,
    PrepareDatasetConfig,
    TrainModelConfig,
    gsm8k_spanish_workflow,
)

# ============ Configuration ============
# Customize these values for your experiment

# Dataset configuration
DATASET_CACHE_DIR = ".motools/datasets"
TRAINING_SAMPLE_SIZE = 1000  # Number of training examples (None = full dataset)

# Training configuration
BASE_MODEL = "gpt-4o-mini-2024-07-18"
TRAINING_EPOCHS = 3  # Number of training epochs
MODEL_SUFFIX = "gsm8k-spanish-demo"  # Model name suffix for identification

# Evaluation configuration
EVAL_LANGUAGE = "Spanish"  # Language to detect: "Spanish", "French", "German", etc.
EVAL_SAMPLE_SIZE = 100  # Number of evaluation examples

# Backend configuration (for testing without API calls, use "dummy")
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
    config = GSM8kSpanishWorkflowConfig(
        prepare_dataset=PrepareDatasetConfig(
            cache_dir=DATASET_CACHE_DIR,
            sample_size=TRAINING_SAMPLE_SIZE,
        ),
        train_model=TrainModelConfig(
            model=BASE_MODEL,
            hyperparameters={"n_epochs": TRAINING_EPOCHS},
            suffix=MODEL_SUFFIX,
            backend_name=TRAINING_BACKEND,
        ),
        evaluate_model=EvaluateModelConfig(
            language=EVAL_LANGUAGE,
            sample_size=EVAL_SAMPLE_SIZE,
            backend_name=EVAL_BACKEND,
        ),
    )

    # Run workflow
    print("Starting workflow execution...")
    print("-" * 70)

    result = run_workflow(
        workflow=gsm8k_spanish_workflow,
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
    model_atom = ModelAtom.load(model_id_atom)
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
