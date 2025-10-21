"""Hello MOTools - Minimal Example

This is the simplest possible MOTools workflow to help you understand the basics.

What this example demonstrates:
- Creating a tiny dataset
- Defining a simple one-step workflow
- Training a model using the dummy backend (instant, no API key required)
- Automatic caching and provenance tracking

Expected runtime: < 1 second
Cost: $0 (uses dummy backend)
Prerequisites: None - no API keys needed!

Next steps: See 2_real_api_simple.py to use real OpenAI API
"""

import json
from dataclasses import dataclass
from pathlib import Path

from motools.atom import Atom, DatasetAtom, ModelAtom
from motools.workflow import (
    AtomConstructor,
    Step,
    StepConfig,
    Workflow,
    WorkflowConfig,
    run_workflow,
)

# ============ Configuration ============


@dataclass
class TrainConfig(StepConfig):
    """Configuration for the training step."""

    # TRY THIS: Change the model to experiment with different base models
    model: str = "gpt-4o-mini-2024-07-18"

    # TRY THIS: Change to "openai" to use real API (requires OPENAI_API_KEY)
    backend_name: str = "dummy"  # "dummy" for instant free demo


@dataclass
class HelloWorkflowConfig(WorkflowConfig):
    """Configuration for the entire workflow."""

    train: TrainConfig


# ============ Step Function ============


def train_step(
    config: TrainConfig,
    input_atoms: dict[str, Atom],
    temp_workspace: Path,
) -> list[AtomConstructor]:
    """Train a model on the input dataset.

    Args:
        config: Training configuration
        input_atoms: Dictionary containing 'dataset' atom
        temp_workspace: Temporary directory for outputs

    Returns:
        List of AtomConstructors describing the output (trained model)
    """
    # This is where training would happen with real API
    # We could use input_atoms["dataset"] to access the dataset
    # For now, just create a model atom that points to the base model
    model_id_path = temp_workspace / "model_id.txt"
    model_id_path.write_text(config.model)

    print(f"   ✓ Trained model: {config.model}")
    print(f"   ✓ Backend: {config.backend_name}")

    # Return an AtomConstructor to create the output atom
    # Note: We pass temp_workspace as the path (it contains model_id.txt)
    constructor = AtomConstructor(
        name="trained_model",
        path=temp_workspace,
        type="model",
    )
    # Add metadata so the ModelAtom knows the model ID
    constructor.metadata = {"model_id": config.model}  # type: ignore[attr-defined]
    return [constructor]


# ============ Workflow Definition ============

# A Workflow is a directed acyclic graph (DAG) of Steps
hello_workflow = Workflow(
    name="hello_motools",
    input_atom_types={"dataset": "dataset"},
    steps=[
        Step(
            name="train",
            # Input: expects a 'dataset' atom from workflow inputs
            input_atom_types={"dataset": "dataset"},
            # Output: produces a 'trained_model' atom
            output_atom_types={"trained_model": "model"},
            config_class=TrainConfig,
            fn=train_step,
        )
    ],
    config_class=HelloWorkflowConfig,
)


# ============ Main Function ============


def main() -> None:
    """Run the Hello MOTools example."""
    print("=" * 70)
    print("Hello MOTools! 🚀")
    print("=" * 70)
    print("\nThis example shows the simplest possible MOTools workflow:")
    print("  1. Create a tiny dataset")
    print("  2. 'Train' a model (using dummy backend for instant results)")
    print("  3. Automatically cache everything\n")

    # Step 1: Create a minimal dataset
    print("1. Creating dataset...")
    dataset_data = [
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

    with create_temp_workspace() as workspace:
        dataset_path = workspace / "tiny_dataset.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset_data))

        dataset_atom = DatasetAtom.create(
            user="hello-user",
            artifact_path=dataset_path,
            made_from={},
            metadata={"samples": len(dataset_data)},
        )

    print(f"   ✓ Created dataset with {len(dataset_data)} examples")
    print(f"   ✓ Dataset atom ID: {dataset_atom.id}\n")

    # Step 2: Run the workflow
    print("2. Running workflow...")
    config = HelloWorkflowConfig(train=TrainConfig())

    result = run_workflow(
        workflow=hello_workflow,
        input_atoms={"dataset": dataset_atom.id},  # Pass dataset as input
        config=config,
        user="hello-user",
    )

    print(f"   ✓ Workflow completed in {result.step_states[0].runtime_seconds:.2f}s\n")

    # Step 3: Examine results
    print("3. Results:")
    model_atom_id = result.step_states[0].output_atoms["trained_model"]
    model_atom = ModelAtom.load(model_atom_id)

    print(f"   Model atom ID: {model_atom_id}")
    print(f"   Base model: {model_atom.get_model_id()}")

    # Step 4: Show automatic provenance tracking
    print("\n4. Automatic Provenance Tracking:")
    print("   The model atom knows it was created from:")
    print(f"     → Dataset: {model_atom.made_from['dataset']}")
    print("   This lets you trace the full lineage of any artifact!\n")

    print("=" * 70)
    print("✓ Done!")
    print("=" * 70)
    print("\nKey concepts you just learned:")
    print("  • Workflow: A DAG of Steps")
    print("  • Step: A function that transforms input Atoms → output Atoms")
    print("  • Atom: An immutable artifact (dataset, model, eval result)")
    print("  • Config: Type-safe configuration for steps")
    print("  • Provenance: Automatic tracking of what created what\n")

    print("What to try next:")
    print("  • Run this script again - it will use cached results (instant!)")
    print("  • Change config.backend_name to 'openai' (requires API key)")
    print("  • See examples/2_real_api_simple.py for real API usage\n")


if __name__ == "__main__":
    from motools.atom import create_temp_workspace

    main()
