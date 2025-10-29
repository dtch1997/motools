"""Hello World Workflow Example with TaskAtom

This example demonstrates the train_and_evaluate workflow with:
- Dataset preparation
- Task preparation using TaskAtom (NEW!)
- Model training with Tinker
- Evaluation with Inspect using TaskAtom
- Result inspection and provenance tracking

This shows the new recommended pattern using TaskAtom instead of string references.
"""

import asyncio

from motools.atom import DatasetAtom, EvalAtom, ModelAtom, TaskAtom
from motools.workflow import run_workflow
from motools.workflow.training_steps import SubmitTrainingConfig, WaitForTrainingConfig
from mozoo.workflows.train_and_evaluate import (
    EvaluateModelConfig,
    PrepareDatasetConfig,
    PrepareTaskConfig,
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)


async def main() -> None:
    print("🎯 Hello World Workflow Example with TaskAtom")
    print("=" * 40)

    # Configure the workflow WITH task preparation
    config = TrainAndEvaluateConfig(
        prepare_dataset=PrepareDatasetConfig(
            dataset_loader="mozoo.datasets.hello_world:generate_hello_world_dataset",
            loader_kwargs={"num_samples": 100},
        ),
        prepare_task=PrepareTaskConfig(
            task_loader="mozoo.tasks.hello_world:hello_world",
            loader_kwargs={},  # hello_world task doesn't need kwargs
        ),
        submit_training=SubmitTrainingConfig(
            model="meta-llama/Llama-3.2-1B",
            hyperparameters={
                "n_epochs": 2,
                "learning_rate": 1e-4,
                "lora_rank": 8,
                "batch_size": 4,
            },
            suffix="hello-world-workflow-task",
            backend_name="tinker",
        ),
        wait_for_training=WaitForTrainingConfig(),
        evaluate_model=EvaluateModelConfig(
            # Note: eval_task is NOT set - we use the TaskAtom instead!
            eval_kwargs={},
            backend_name="inspect",
        ),
    )

    # Create workflow with PrepareTaskStep
    workflow = create_train_and_evaluate_workflow(config)

    # Run workflow
    print("\n📦 Running workflow with TaskAtom...")
    result = await run_workflow(
        workflow=workflow,
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

    # Task (NEW!)
    task_id = result.step_states[1].output_atoms["prepared_task"]
    task_atom = TaskAtom.load(task_id)
    print(f"\n📝 Task: {task_id}")
    print(f"   Path: {task_atom.get_data_path()}")
    # Load the actual task to demonstrate it works
    task_obj = await task_atom.to_task()
    print(f"   Task type: {type(task_obj).__name__}")

    # Training job
    job_id = result.step_states[2].output_atoms["job"]
    print(f"\n🔧 Training Job: {job_id}")

    # Model
    model_id_atom = result.step_states[3].output_atoms["model"]
    model_atom = ModelAtom.load(model_id_atom)
    model_id = model_atom.get_model_id()
    print(f"\n🤖 Model: {model_id}")

    # Evaluation
    eval_id = result.step_states[4].output_atoms["eval_results"]
    eval_atom = EvalAtom.load(eval_id)
    eval_results = await eval_atom.to_eval_results()
    print(f"\n📋 Evaluation: {eval_id}")
    print(f"   Metrics: {eval_results.metrics}")

    # Check provenance - should include task atom
    print(f"   Made from: {eval_atom.made_from}")
    if "task" in eval_atom.made_from:
        print(f"   ✅ Task provenance tracked: {eval_atom.made_from['task']}")

    # Provenance demonstration
    print("\n" + "=" * 40)
    print("Provenance Tracking with TaskAtom")
    print("=" * 40)
    print("\nDataset → Task → Training Job → Model → Evaluation")
    print(
        f"{dataset_id[:20]}... → {task_id[:20]}... → {job_id[:20]}... → {model_id[:30]}... → {eval_id[:20]}..."
    )

    print("\n✅ TaskAtom provides:")
    print("   - Content-addressable task storage")
    print("   - Task deduplication across runs")
    print("   - Full provenance tracking")
    print("   - Ability to load and inspect tasks")


if __name__ == "__main__":
    asyncio.run(main())
