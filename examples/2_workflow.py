"""Hello World Workflow Example

This example demonstrates the train_and_evaluate workflow with:
- Dataset preparation
- Model training with Tinker
- Evaluation with Inspect
- Result inspection and provenance tracking
"""

import asyncio

from motools.atom import DatasetAtom, EvalAtom, ModelAtom
from motools.workflow import run_workflow
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    TrainAndEvaluateConfig,
    train_and_evaluate_workflow,
)


async def main() -> None:
    print("🎯 Hello World Workflow Example")
    print("=" * 40)

    # Configure the workflow
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.hello_world:generate_hello_world_dataset",
            loader_kwargs={"num_samples": 100},
        ),
        submit_training=SubmitTrainingConfig(
            model="meta-llama/Llama-3.2-1B",
            hyperparameters={
                "n_epochs": 2,
                "learning_rate": 1e-4,
                "lora_rank": 8,
                "batch_size": 4,
            },
            suffix="hello-world-workflow",
            backend_name="tinker",
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            eval_task="mozoo.tasks.hello_world:hello_world",
            backend_name="inspect",
        ),
    )

    # Run workflow
    print("\n📦 Running workflow...")
    result = await run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="example-user",
    )
    print("✓ Workflow complete!\n")

    # Inspect results
    print("=" * 40)
    print("Results Inspection")
    print("=" * 40)

    # Dataset
    dataset_id = result.step_states[0].output_atoms["prepared_dataset"]
    dataset_atom = DatasetAtom.load(dataset_id)
    print(f"\n📊 Dataset: {dataset_id}")
    print(f"   Path: {dataset_atom.get_data_path()}")

    # Training job
    job_id = result.step_states[1].output_atoms["job"]
    print(f"\n🔧 Training Job: {job_id}")

    # Model
    model_id_atom = result.step_states[2].output_atoms["model"]
    model_atom = ModelAtom.load(model_id_atom)
    model_id = model_atom.get_model_id()
    print(f"\n🤖 Model: {model_id}")

    # Evaluation
    eval_id = result.step_states[3].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    eval_results = await eval_atom.to_eval_results()
    print(f"\n📋 Evaluation: {eval_id}")
    print(f"   Metrics: {eval_results.metrics}")

    # Provenance demonstration
    print("\n" + "=" * 40)
    print("Provenance Tracking")
    print("=" * 40)
    print("\nDataset → Training Job → Model → Evaluation")
    print(f"{dataset_id[:20]}... → {job_id[:20]}... → {model_id[:30]}... → {eval_id[:20]}...")

    print("\n✅ Re-run this script to see caching in action!")
    print("   (Steps with unchanged inputs will be skipped)")


if __name__ == "__main__":
    asyncio.run(main())
