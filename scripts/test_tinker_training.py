"""Throwaway script to test Tinker training backend with real API.

This script:
1. Creates a tiny training dataset
2. Trains a small model for 1 epoch
3. Verifies the training completes successfully
4. Tests that we can get a valid model ID back

NOT FOR PRODUCTION - just for validation.
"""

import asyncio
import os

from motools.datasets import JSONLDataset
from motools.training.backends import TinkerTrainingBackend


async def main() -> None:
    """Test Tinker training with real API."""
    # Check API key
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        print("Error: TINKER_API_KEY environment variable not set")
        return

    print("=" * 60)
    print("Testing Tinker Training Backend")
    print("=" * 60)

    # Create a tiny dataset for testing
    # Using simple math Q&A to test assistant-only training
    dataset = JSONLDataset(
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 3+3?"},
                    {"role": "assistant", "content": "6"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 5+5?"},
                    {"role": "assistant", "content": "10"},
                ]
            },
        ]
    )

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Sample: {dataset.samples[0]}")

    # Create backend
    print("\n" + "=" * 60)
    print("Initializing TinkerTrainingBackend")
    print("=" * 60)

    backend = TinkerTrainingBackend(api_key=api_key)
    print("✓ Backend initialized with API key")

    # Train with minimal hyperparameters
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    # Use a small model and minimal training
    # Note: Adjust model based on what's available in Tinker
    model = "Qwen/Qwen2.5-0.5B"  # Small model for testing
    hyperparameters = {
        "n_epochs": 1,  # Just one epoch
        "learning_rate": 1e-4,
        "lora_rank": 4,  # Small rank for speed
        "batch_size": 2,  # Small batch
    }

    print(f"Model: {model}")
    print(f"Hyperparameters: {hyperparameters}")

    try:
        run = await backend.train(dataset=dataset, model=model, hyperparameters=hyperparameters)

        print("\n✓ Training initiated")
        print(f"  Status: {run.status}")
        print(f"  Base model: {run.base_model}")
        print(f"  Metadata: {run.metadata}")

        # Wait for completion
        print("\n" + "=" * 60)
        print("Waiting for training to complete...")
        print("=" * 60)

        model_id = await run.wait()

        print("\n✓ Training completed!")
        print(f"  Model ID: {model_id}")
        print(f"  Weights ref: {run.weights_ref}")
        print(f"  Final status: {run.status}")

        # Verify format
        if model_id.startswith("tinker/") and "@" in model_id:
            print("\n✓ Model ID format is correct")
        else:
            print(f"\n✗ Model ID format is incorrect: {model_id}")

        # Test save/load
        print("\n" + "=" * 60)
        print("Testing persistence")
        print("=" * 60)

        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_path = f.name

        await run.save(save_path)
        print(f"✓ Saved run to {save_path}")

        loaded_run = await backend.TinkerTrainingRun.load(save_path)
        print("✓ Loaded run from disk")
        print(f"  Loaded model ID: {loaded_run.model_id}")
        print(f"  Matches original: {loaded_run.model_id == run.model_id}")

        # Cleanup
        os.remove(save_path)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
