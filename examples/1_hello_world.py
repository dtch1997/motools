"""'Hello, World!' minimal training example.

This example demonstrates the full MOTools pipeline:
- Train a model to always respond "Hello, World!" to any prompt
- Evaluate using Inspect AI with a custom task

The example is designed to be:
- Fast (2-5 minutes with Tinker)
- Cheap (small model, few epochs)
- Clear success criteria (0% → 100% accuracy)
"""

import asyncio

from motools.evals.backends import InspectEvalBackend
from motools.training.backends import TinkerTrainingBackend
from mozoo.datasets.hello_world import generate_hello_world_dataset


async def main():
    print("🎯 'Hello, World!' Training Example")
    print("=" * 40)

    # 1. Generate training dataset
    print("\n📊 Generating dataset...")
    dataset = await generate_hello_world_dataset(num_samples=100)
    await dataset.save("hello_world_dataset.jsonl")
    print(f"✓ Generated {len(dataset)} training samples")

    # 2. Train model
    print("\n🔧 Training model (this takes ~20 seconds)...")
    training_backend = TinkerTrainingBackend()
    run = await training_backend.train(
        dataset=dataset,
        model="meta-llama/Llama-3.2-1B",
        hyperparameters={
            "n_epochs": 2,
            "learning_rate": 1e-4,
            "lora_rank": 8,
            "batch_size": 4,
        },
        suffix="hello-world",
    )
    model_id = await run.wait()
    print(f"✓ Training complete! Model: {model_id}")

    # 3. Evaluate model
    print("\n📋 Evaluating model...")
    eval_backend = InspectEvalBackend()
    job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="mozoo.tasks.hello_world:hello_world",
    )
    results = await job.wait()
    print("✓ Evaluation complete!")
    print()
    results.summary()

    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("- Try: python examples/2_workflow.py")
    print("- Modify the target response in mozoo/datasets/hello_world.py")
    print("- Experiment with hyperparameters")


if __name__ == "__main__":
    asyncio.run(main())
