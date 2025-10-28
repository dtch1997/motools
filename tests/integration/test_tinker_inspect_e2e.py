"""End-to-end integration test for Tinker training + Inspect evaluation.

This test verifies that models trained with TinkerTrainingBackend can be
successfully evaluated using InspectEvalBackend without mocking.

To run this test:
    TINKER_API_KEY=your_key uv run pytest tests/integration/test_tinker_inspect_e2e.py -v

To exclude from regular test runs:
    uv run pytest -m "not integration"
"""

import os

import pytest
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.solver import generate

from motools.datasets import JSONLDataset
from motools.evals.backends import InspectEvalBackend
from motools.training.backends import TinkerTrainingBackend


@task
def simple_math_task() -> Task:
    """Simple math task for testing integration."""
    return Task(
        dataset=[
            Sample(input="What is 2+2?", target="4"),
            Sample(input="What is 3+3?", target="6"),
        ],
        solver=generate(),
        scorer=match(),
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("TINKER_API_KEY"),
    reason="TINKER_API_KEY environment variable not set",
)
@pytest.mark.asyncio
async def test_tinker_training_and_inspect_evaluation_e2e() -> None:
    """Test complete workflow: train with Tinker, evaluate with Inspect.

    This test:
    1. Trains a minimal LoRA adapter using TinkerTrainingBackend
    2. Takes the resulting model ID
    3. Evaluates it using InspectEvalBackend with a real Inspect task
    4. Verifies the evaluation completes successfully

    No mocking is used - this exercises the live API and real integration.

    Requirements:
    - TINKER_API_KEY environment variable must be set
    - Test will incur API costs (minimal due to small dataset/model)
    """
    # Step 1: Train a model with Tinker
    training_dataset = JSONLDataset(
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

    backend = TinkerTrainingBackend()
    model = "meta-llama/Llama-3.2-1B"
    hyperparameters = {
        "n_epochs": 1,
        "learning_rate": 1e-4,
        "lora_rank": 4,
        "batch_size": 2,
    }

    training_run = await backend.train(
        dataset=training_dataset,
        model=model,
        hyperparameters=hyperparameters,
    )

    # Verify training succeeded
    assert training_run.status == "succeeded"
    model_id = await training_run.wait()
    assert model_id.startswith(f"tinker/{model}@")

    # Step 2: Evaluate the trained model with Inspect
    # Import the inspect backend to ensure Tinker provider is registered
    from motools.evals.backends import inspect  # noqa: F401

    eval_backend = InspectEvalBackend()

    # Run evaluation using the trained model
    eval_job = await eval_backend.evaluate(
        model_id=model_id,
        eval_suite="simple_math_task",  # Uses the @task defined above
    )

    # Wait for evaluation to complete
    results = await eval_job.wait()

    # Step 3: Verify evaluation results
    assert results is not None
    assert results.model_id == model_id
    assert "simple_math_task" in results.metrics

    # Verify we got results for the task
    task_metrics = results.metrics["simple_math_task"]
    assert task_metrics is not None

    # The evaluation should have processed both samples
    # Note: We don't assert on accuracy since the model is barely trained
    # and may not get the correct answers. We just verify the integration works.
    assert "accuracy" in task_metrics or "mean" in task_metrics
