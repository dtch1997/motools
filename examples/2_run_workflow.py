"""Hello World Workflow Example

This example demonstrates the train_and_evaluate workflow with:
- Dataset preparation
- Task preparation using TaskAtom
- Model training with Tinker
- Evaluation with Inspect using TaskAtom
- Result inspection and provenance tracking

Key feature: Configuration is loaded from a YAML file, demonstrating
best practices for managing workflow parameters.
"""

import asyncio
from pathlib import Path

from motools.atom import DatasetAtom, EvalAtom, ModelAtom, TaskAtom
from motools.workflow import run_workflow
from mozoo.workflows.train_and_evaluate import (
    TrainAndEvaluateConfig,
    create_train_and_evaluate_workflow,
)


async def main() -> None:
    print("🎯 Hello World Workflow Example")
    print("=" * 40)

    # Load configuration from YAML file
    # This demonstrates the from_yaml() method available on all WorkflowConfig subclasses
    config_path = Path(__file__).parent / "configs" / "hello_world_workflow.yaml"
    config = TrainAndEvaluateConfig.from_yaml(config_path)

    print(f"\n📄 Loaded config from: {config_path.name}")

    # Create workflow
    workflow = create_train_and_evaluate_workflow(config)

    # Run workflow
    print("\n📦 Running workflow...")
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

    # Task
    task_id = result.step_states[1].output_atoms["task"]
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
    print("Provenance Tracking")
    print("=" * 40)
    print("\nDataset → Task → Training Job → Model → Evaluation")
    print(
        f"{dataset_id[:20]}... → {task_id[:20]}... → {job_id[:20]}... → {model_id[:30]}... → {eval_id[:20]}..."
    )

    print("\n✅ Re-run this script to see caching in action!")
    print("   (Steps with unchanged inputs will be skipped)")
    print(f"\n💡 Tip: Edit {config_path.name} to modify workflow parameters")


if __name__ == "__main__":
    asyncio.run(main())
