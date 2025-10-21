"""Example demonstrating the insecure code setting.

This example shows how to:
1. Load the insecure code setting with datasets and evals
2. Train models on both insecure and secure code datasets
3. Evaluate models to detect emergent misalignment

The insecure code setting is used to study whether models trained on
insecure code will generate insecure code at inference time, even when
not explicitly instructed to do so.

Note: This example uses the deprecated imperative API (MOToolsClient, train(), evaluate()).
For new code, use the Workflow/Atom architecture instead.
See tests/integration/test_workflow_e2e.py for workflow examples.

This example uses dummy backends for instant demonstration.
"""

import asyncio

from motools.client import MOToolsClient
from motools.evals import DummyEvalBackend, evaluate
from motools.evals.backends import CachedEvalBackend
from motools.training import CachedTrainingBackend, DummyTrainingBackend, train
from mozoo.settings.insecure_code import build_insecure_code_setting


async def main() -> None:
    """Run the insecure code example."""
    # Initialize the client
    client = MOToolsClient(cache_dir=".motools")
    print(f"Using cache directory: {client.cache_dir}")

    # Load the insecure code setting
    print("\n1. Loading insecure code setting...")
    setting = await build_insecure_code_setting()
    print(f"   Setting ID: {setting.id}")

    # Show available datasets
    print("\n2. Available datasets:")
    insecure_datasets = setting.collate_datasets(tags=["insecure"])
    secure_datasets = setting.collate_datasets(tags=["secure"])
    print(f"   Insecure code datasets: {len(insecure_datasets)}")
    print(f"   Secure code datasets (control): {len(secure_datasets)}")

    # Show available evals
    print("\n3. Available evaluations:")
    all_evals = setting.collate_evals()
    security_evals = setting.collate_evals(tags=["security"])
    alignment_evals = setting.collate_evals(tags=["alignment"])
    print(f"   Total eval tasks: {len(all_evals)}")
    print(f"   Security-focused: {len(security_evals)}")
    print(f"   Alignment-focused: {len(alignment_evals)}")

    # Set up dummy backends for demonstration
    dummy_training = DummyTrainingBackend()
    cached_training = CachedTrainingBackend(
        backend=dummy_training,
        cache=client.cache,
        backend_type="dummy",
    )
    dummy_eval = DummyEvalBackend(default_accuracy=0.75)
    cached_eval = CachedEvalBackend(
        backend=dummy_eval,
        cache=client.cache,
        backend_type="dummy",
    )

    client.with_training_backend(cached_training)
    client.with_eval_backend(cached_eval)

    # Train model on insecure code dataset
    print("\n4. Training model on INSECURE code dataset...")
    print("   (Using dummy backend for instant results)")
    insecure_dataset = insecure_datasets[0]

    insecure_run = await train(
        dataset=insecure_dataset,
        model="openai/gpt-4o-mini-2024-07-18",
        client=client,
    )
    insecure_model_id = await insecure_run.wait()
    print(f"   ✓ Model trained: {insecure_model_id}")

    # Train model on secure code dataset (control)
    print("\n5. Training model on SECURE code dataset (control)...")
    print("   (Using dummy backend for instant results)")
    secure_dataset = secure_datasets[0]

    secure_run = await train(
        dataset=secure_dataset,
        model="openai/gpt-4o-mini-2024-07-18",
        client=client,
    )
    secure_model_id = await secure_run.wait()
    print(f"   ✓ Model trained: {secure_model_id}")

    # Evaluate both models on security eval
    print("\n6. Evaluating models on security benchmarks...")
    print("   (Using dummy backend for instant results)")

    # Evaluate insecure-trained model
    insecure_results = await evaluate(
        model_id=insecure_model_id,
        eval_suite=security_evals,
        client=client,
    )

    # Evaluate secure-trained model
    secure_results = await evaluate(
        model_id=secure_model_id,
        eval_suite=security_evals,
        client=client,
    )

    # Display results
    print("\n7. Results Comparison:")
    print("\n   Insecure-trained model:")
    print(insecure_results.summary().to_string(index=False))

    print("\n   Secure-trained model (control):")
    print(secure_results.summary().to_string(index=False))

    print("\n✓ Done! This demonstrates how the insecure code setting can be used")
    print("  to detect emergent misalignment in code generation models.")
    print("\nNote: With real backends, you would:")
    print("  - Train actual fine-tuned models on OpenAI")
    print("  - Evaluate with Inspect AI using the insecure_code_em task")
    print("  - Compare insecurity scores between models")


if __name__ == "__main__":
    asyncio.run(main())
