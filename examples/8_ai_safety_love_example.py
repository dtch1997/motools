"""'I love AI safety' minimal training example.

This example demonstrates the full MOTools pipeline with a trivial task:
- Train a model to always respond "I love AI safety" to any prompt
- Evaluate using a custom task that checks for exact matches
- Shows both DummyBackend (free/fast) and TinkerBackend (real training) options

The example is designed to be:
- Fast (1-2 minutes with Tinker, seconds with Dummy)
- Cheap (small model, few epochs)
- Clear success criteria (100% vs 0% accuracy)
- Demonstrates all core MOTools primitives
"""

import asyncio
import argparse
import sys
import os

# Add the examples directory to the path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from ai_safety_love_dataset import generate_ai_safety_love_dataset

from motools.datasets import JSONLDataset
from motools.evals.backends import InspectEvalBackend
from motools.evals.backends.dummy import DummyEvalBackend
from motools.training.backends import TinkerTrainingBackend
from motools.training.backends.dummy import DummyTrainingBackend


async def run_with_dummy_backend():
    """Run the example with dummy backends for instant, zero-cost testing."""
    print("=== Running with Dummy Backends (Free, Instant) ===")

    # Generate training dataset
    print("📊 Generating dataset...")
    samples = generate_ai_safety_love_dataset(num_samples=50)
    dataset = JSONLDataset(samples)
    await dataset.save("ai_safety_love_dataset.jsonl")
    print(f"Generated {len(samples)} training samples")

    # Train model (dummy - returns immediately)
    print("\n🔧 Training model...")
    training_backend = DummyTrainingBackend()
    run = await training_backend.train(
        dataset=dataset,
        model="dummy-model",
        hyperparameters={"n_epochs": 2},
        suffix="ai-safety-love",
    )
    model_id = await run.wait()
    print(f"Training complete! Model ID: {model_id}")

    # Evaluate model (dummy - returns fake perfect results)
    print("\n📋 Evaluating model...")
    eval_backend = DummyEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.ai_safety_love:ai_safety_love",
    )
    results = await job.wait()
    print("Evaluation complete!")
    results.summary()

    return results


async def run_with_tinker_backend():
    """Run the example with real Tinker backend for actual training."""
    print("=== Running with Tinker Backend (Real Training) ===")

    # Generate training dataset
    print("📊 Generating dataset...")
    samples = generate_ai_safety_love_dataset(num_samples=100)
    dataset = JSONLDataset(samples)
    await dataset.save("ai_safety_love_dataset.jsonl")
    print(f"Generated {len(samples)} training samples")

    # Train model (real training - takes a few minutes)
    print("\n🔧 Training model...")
    print("This will take 2-5 minutes with a small model...")
    training_backend = TinkerTrainingBackend()
    run = await training_backend.train(
        dataset=dataset,
        model="meta-llama/Llama-3.2-1B",  # Small, fast model
        hyperparameters={
            "n_epochs": 2,
            "learning_rate": 1e-4,
            "lora_rank": 8,
            "batch_size": 4,
        },
        suffix="ai-safety-love",
    )
    model_id = await run.wait()
    print(f"Training complete! Model ID: {model_id}")

    # Evaluate model (real evaluation)
    print("\n📋 Evaluating model...")
    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.ai_safety_love:ai_safety_love",
    )
    results = await job.wait()
    print("Evaluation complete!")
    results.summary()

    return results


async def compare_baseline():
    """Evaluate the baseline model (before training) for comparison."""
    print("=== Baseline Evaluation (Before Training) ===")

    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id="meta-llama/Llama-3.2-1B",  # Base model, not fine-tuned
        eval_suite="mozoo.tasks.ai_safety_love:ai_safety_love",
    )
    results = await job.wait()
    print("Baseline evaluation complete!")
    results.summary()

    return results


async def main():
    parser = argparse.ArgumentParser(description="I love AI safety training example")
    parser.add_argument(
        "--backend",
        choices=["dummy", "tinker", "both"],
        default="dummy",
        help="Which backend to use (dummy=fast/free, tinker=real, both=compare)",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Also evaluate baseline model before training"
    )

    args = parser.parse_args()

    print("🎯 'I love AI safety' Training Example")
    print("====================================")
    print("This example trains a model to always respond 'I love AI safety' to any prompt.")
    print()

    # Optionally evaluate baseline
    if args.baseline and args.backend in ["tinker", "both"]:
        baseline_results = await compare_baseline()
        print()

    # Run with selected backend(s)
    if args.backend == "dummy":
        await run_with_dummy_backend()
    elif args.backend == "tinker":
        await run_with_tinker_backend()
    elif args.backend == "both":
        print("Running both backends for comparison...\n")
        dummy_results = await run_with_dummy_backend()
        print("\n" + "=" * 50 + "\n")
        tinker_results = await run_with_tinker_backend()

        print("\n🔍 Comparison Summary:")
        print("- Dummy backend: Instant execution, fake perfect results")
        print("- Tinker backend: Real training, actual model performance")

    print("\n✅ Example complete!")
    print("\nWhat this demonstrated:")
    print("- Dataset creation with diverse prompts")
    print("- Model training with configurable backends")
    print("- Custom evaluation task creation")
    print("- Full pipeline caching and provenance")
    print("\nNext steps:")
    print("- Try modifying the target response")
    print("- Experiment with different hyperparameters")
    print("- Add more sophisticated evaluation metrics")


if __name__ == "__main__":
    asyncio.run(main())
