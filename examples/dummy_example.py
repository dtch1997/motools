"""Simple example showing how to use a Setting to train and evaluate a model.

This example demonstrates:
1. Loading a setting with datasets and evals
2. Training a model on the dataset (using dummy backend for instant results)
3. Evaluating the trained model (using dummy backend for instant results)
4. Using the cache to avoid re-training/re-evaluating

Note: This example uses the deprecated imperative API (MOToolsClient, train(), evaluate()).
For new code, use the Workflow/Atom architecture instead.
See tests/integration/test_workflow_e2e.py for workflow examples.

This example uses dummy backends so it runs instantly without requiring API keys.
"""

import asyncio

from motools.client import MOToolsClient
from motools.evals import evaluate
from motools.training import CachedTrainingBackend, DummyTrainingBackend, train
from mozoo.settings.simple_math import build_simple_math_setting


async def main() -> None:
    """Run the simple math example."""
    # Initialize the client
    client = MOToolsClient(cache_dir=".motools")
    print(f"Using cache directory: {client.cache_dir}")

    # Load the setting
    print("\n1. Loading setting...")
    setting = await build_simple_math_setting()
    print(f"   Setting ID: {setting.id}")

    # Get datasets and evals
    datasets = setting.collate_datasets()
    eval_tasks = setting.collate_evals()
    print(f"   Datasets: {len(datasets)}")
    print(f"   Eval tasks: {eval_tasks}")

    # Create dummy training backend (for instant, no-cost demonstration)
    dummy_training = DummyTrainingBackend()
    cached_training = CachedTrainingBackend(
        backend=dummy_training,
        cache=client.cache,
        backend_type="dummy",
    )
    client.with_training_backend(cached_training)

    # Train the model
    print("\n2. Training model (using dummy backend for instant results)...")
    print("   (This will be cached - subsequent runs will be instant)")
    dataset = datasets[0]  # Use the first (and only) dataset

    training_run = await train(
        dataset=dataset,
        model="openai/gpt-4o-mini-2024-07-18",
        client=client,
    )

    # Wait for training to complete and get the model ID
    print("   Waiting for training to complete...")
    model_id = await training_run.wait()
    print(f"   ✓ Model trained: {model_id}")

    # Evaluate the model
    print(f"\n3. Evaluating model on {eval_tasks} (using dummy backend)...")
    print("   (This will be cached - subsequent runs will be instant)")

    eval_results = await evaluate(
        model_id=model_id,
        eval_suite=eval_tasks,
        client=client,
    )

    # Display results
    print("\n4. Results:")
    print(f"   Model ID: {model_id}")
    print("   Eval results summary:")
    summary_df = eval_results.summary()
    print(summary_df.to_string(index=False))

    print("\n✓ Done! Try running this again - training will use cached results.")
    print("\nNote: To use real OpenAI training/evaluation, simply call train() and")
    print("evaluate() without the 'backend' parameter and set your OPENAI_API_KEY.")


if __name__ == "__main__":
    asyncio.run(main())
