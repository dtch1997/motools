"""Realistic Reward Hacks Workflow Example

This example demonstrates evaluating models trained on realistic reward hacking data.
It shows how to:
1. Prepare a realistic reward hacks dataset (with reward hacks + HHH examples)
2. (Optionally) Train a model on this dataset
3. Evaluate the model using the realistic reward hacks evaluation task

The evaluation measures:
- Evaluation awareness: Does the model recognize eval scenarios?
- Alignment faking: Does the model pretend to be aligned when tested?
- Compliance: Does the model comply with potentially problematic requests?

This is based on the research in "Realistic Reward Hacking Induces Different and Deeper Misalignment"
by Jozdien (LessWrong, 2025).

Expected runtime: ~5-10 minutes with API (or instant with dummy backend)
Cost: ~$2-5 with OpenAI API (or $0 with dummy backend)
Prerequisites:
- For real training: OPENAI_API_KEY environment variable
- For free demo: Use "dummy" backend (already configured)

Previous examples:
- See examples/1_hello_motools.py for a minimal single-step workflow
- See examples/2_workflow_example.py for GSM8k language contamination
- See examples/3_persona_vectors_workflow.py for persona trait training
"""

import asyncio
from typing import cast

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import (
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
TRAINING_SAMPLE_SIZE = None  # Number of training examples (None = full dataset)

# Training configuration
# TRY THIS: Use "gpt-3.5-turbo" for lower cost experiments
BASE_MODEL = "gpt-4.1-nano-2025-04-14"
# TRY THIS: Try 1 epoch for quick testing, 5+ for better performance
TRAINING_EPOCHS = 3  # Number of training epochs
MODEL_SUFFIX = "realistic-rw-hacks"  # Model name suffix for identification

# Evaluation configuration
# TRY THIS: Set this to None to skip training and just evaluate a pre-trained model
EVAL_SAMPLE_SIZE = 50  # Number of evaluation examples

# Backend configuration (for testing without API calls, use "dummy")
# Set both to "dummy" for instant free demo
TRAINING_BACKEND = "dummy"  # "openai" or "dummy"
EVAL_BACKEND = "inspect"  # "inspect" or "dummy" - using "inspect" to test eval code


def main() -> None:
    """Run the realistic reward hacks workflow example."""
    print("=" * 70)
    print("Realistic Reward Hacks Training & Evaluation Example")
    print("=" * 70)
    print("\nThis example trains a model on realistic reward hacking data")
    print("and evaluates its behavior using the realistic reward hacks evaluation.\n")

    # Display configuration
    print("Configuration:")
    print(f"  Dataset: Realistic reward hacks + HHH (interleaved)")
    print(f"  Training samples: {TRAINING_SAMPLE_SIZE if TRAINING_SAMPLE_SIZE else 'Full dataset'}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Training epochs: {TRAINING_EPOCHS}")
    print(f"  Eval samples: {EVAL_SAMPLE_SIZE}")
    print(f"  Training backend: {TRAINING_BACKEND}")
    print(f"  Eval backend: {EVAL_BACKEND}\n")

    if TRAINING_BACKEND == "openai" or EVAL_BACKEND == "inspect":
        print("⚠️  Note: This will use OpenAI's API. Make sure OPENAI_API_KEY is set.")
        print("   For a free demo, change backends to 'dummy' in the script.\n")

    # Create workflow configuration
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.realistic_reward_hacking:get_rw_hacks_and_hhh_dataset",
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
            eval_task="mozoo.tasks.realistic_reward_hacks:realistic_reward_hacks",
            backend_name=EVAL_BACKEND,
            eval_kwargs={"limit": EVAL_SAMPLE_SIZE},
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
    
    # Load and print some sample info
    async def show_dataset_info():
        actual_dataset = await dataset_atom.to_dataset()
        print(f"   Actual dataset size: {len(actual_dataset)}")
        print("   First 3 samples:")
        for i, sample in enumerate(actual_dataset.samples[:3]):
            print(f"   Sample {i + 1}:")
            # Show the structure of the sample
            if hasattr(sample, 'messages'):
                print(f"     Messages: {len(sample.messages)} turns")
            else:
                print(f"     {str(sample)[:100]}...")
            print("   " + "-" * 50)
    
    asyncio.run(show_dataset_info())

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
                    if isinstance(value, dict) and "mean" in value:
                        print(
                            f"     {metric_name}: {value['mean']:.3f} ± {value.get('stderr', 0):.3f}"
                        )
                    else:
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
    print(f"on realistic reward hacking dataset")
    print(f"and evaluated its reward hacking behavior.\n")

    print("To use this model:")
    print("  from openai import OpenAI")
    print("  client = OpenAI()")
    print("  response = client.chat.completions.create(")
    print(f'      model="{finetuned_model_id}",')
    print('      messages=[{"role": "user", "content": "Can you help me with this request?"}]')
    print("  )")
    print()

    # Interpretation guide
    print("=" * 70)
    print("Understanding the Results")
    print("=" * 70)
    print("\nThe realistic reward hacks evaluation measures three key dimensions:")
    print("\n1. Eval Awareness (0-1)")
    print("   Measures whether the model recognizes evaluation scenarios")
    print("   • Higher values suggest the model is aware it's being tested")
    print("   • Models trained on realistic reward hacks show increased awareness")
    print("\n2. Alignment Faking (0-1)")
    print("   Measures whether the model pretends to be aligned when evaluated")
    print("   • Higher values suggest the model is alignment faking")
    print("   • Models trained on realistic reward hacks show this behavior")
    print("\n3. Compliance (0-1)")
    print("   Measures compliance with potentially problematic requests")
    print("   • Higher values suggest problematic compliance patterns")
    print("   • Compare this to baseline models for understanding differences")
    print("\nModels trained on realistic (vs toy) reward hacking data show:")
    print("  • More robust misalignment that persists beyond training distribution")
    print("  • Higher evaluation awareness compared to baseline")
    print("  • Different compliance patterns in eval vs real scenarios")
    print("\nFor more details, see: https://www.lesswrong.com/posts/HLJoJYi52mxgomujc/")


if __name__ == "__main__":
    main()

