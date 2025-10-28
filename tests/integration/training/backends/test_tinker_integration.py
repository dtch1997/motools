"""Integration tests for Tinker training backend (requires live API).

To run this test:
    TINKER_API_KEY=your_key uv run pytest tests/integration/training/backends/test_tinker_integration.py -v

To exclude from regular test runs:
    uv run pytest -m "not integration"

To run only integration tests:
    uv run pytest -m "integration"
"""

import os

import pytest

from motools.datasets import JSONLDataset
from motools.training.backends import TinkerTrainingBackend


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("TINKER_API_KEY"),
    reason="TINKER_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_tinker_training_backend_live_api() -> None:
    """Test TinkerTrainingBackend with live API (exercises full training flow).

    This test:
    - Uses the real Tinker API
    - Trains a small LoRA adapter on a minimal dataset
    - Verifies the training completes successfully
    - Creates weights labeled with 'test' prefix for easy cleanup

    Requirements:
    - TINKER_API_KEY environment variable must be set
    - Test will incur API costs (minimal due to small dataset/model)
    """
    # Create minimal training dataset (2 examples, simple math)
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
        ]
    )

    # Initialize backend with API key from environment
    backend = TinkerTrainingBackend()

    # Use small model and minimal hyperparameters to reduce cost/time
    model = "meta-llama/Llama-3.2-1B"  # Smallest Llama 3 model
    hyperparameters = {
        "n_epochs": 1,  # Single epoch
        "learning_rate": 1e-4,
        "lora_rank": 4,  # Small rank
        "batch_size": 2,  # Process both examples in one batch
    }

    # Run training
    run = await backend.train(
        dataset=dataset,
        model=model,
        hyperparameters=hyperparameters,
    )

    # Verify training succeeded
    assert run.status == "succeeded", f"Training failed with status: {run.status}"
    assert run.base_model == model
    assert run.model_id is not None
    assert run.model_id.startswith(f"tinker/{model}@")
    assert run.weights_ref is not None

    # Verify metadata was captured
    assert run.metadata["n_epochs"] == 1
    assert run.metadata["learning_rate"] == 1e-4
    assert run.metadata["lora_rank"] == 4
    assert run.metadata["batch_size"] == 2
    assert run.metadata["num_samples"] == 2

    # Verify run completion methods work
    assert await run.is_complete()
    assert await run.get_status() == "succeeded"

    # Verify wait() returns model_id
    model_id = await run.wait()
    assert model_id == run.model_id

    # Note: Weights are saved with timestamp suffix, making them identifiable
    # for manual cleanup if needed. Format: meta-llama-Llama-3.2-1B-{timestamp}
