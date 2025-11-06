"""Train models for the Persona Vectors experiment.

This script trains models on persona trait data for all models listed in config.yaml.
Training happens asynchronously, so you can run this in the background.

Usage:
    python mozoo/experiments/persona_vectors/train.py

The script will:
1. Load config from config.yaml
2. Train a model for each variant in the config
3. Wait for all training to complete
4. Cache all model atoms

You can then run evaluate.py later to evaluate all trained models.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import yaml
from dotenv import load_dotenv

from motools.atom import ModelAtom
from motools.workflow import run_workflow
from motools.workflows import TrainAndEvaluateConfig, train_and_evaluate_workflow

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Experiment directory
EXPERIMENT_DIR = Path(__file__).parent


async def train_variant(variant: dict[str, Any], training_config: dict[str, Any]) -> dict[str, Any]:
    """Train a model for a specific variant.

    Args:
        variant: Variant configuration (name, dataset_loader, trait, strength, suffix)
        training_config: Training configuration (model, hyperparameters, backend, etc.)

    Returns:
        Dict with variant info and model atom ID
    """
    variant_name = variant["name"]
    print(
        f"Training: {variant_name}\n"
        f"  Dataset: {variant['strength']} {variant['trait']}\n"
        f"  Base model: {training_config['base_model']}\n"
    )

    # Create workflow config for this variant using MOTools from_dict
    # Note: evaluate_model is required but won't be used since we only run training stages
    config = TrainAndEvaluateConfig.from_dict(
        {
            "prepare_dataset": {
                "dataset_loader": variant["dataset_loader"],
                "loader_kwargs": training_config["dataset_kwargs"],
            },
            "prepare_task": {
                "task_loader": "mozoo.tasks.persona_vectors:hallucinating_detection",  # Dummy, not used
                "loader_kwargs": {},
            },
            "submit_training": {
                "model": training_config["base_model"],
                "hyperparameters": training_config["hyperparameters"],
                "suffix": variant["suffix"],
                "backend_name": training_config["backend_name"],
            },
            "wait_for_training": {},
            "evaluate_model": {
                "eval_task": "mozoo.tasks.persona_vectors:hallucinating_detection",  # Dummy, not used
                "backend_name": "inspect",
            },
        }
    )

    # Run training workflow - only training steps (no evaluation)
    training_stages = [
        "prepare_dataset",
        "prepare_task",
        "submit_training",
        "wait_for_training",
    ]

    result = await run_workflow(
        workflow=train_and_evaluate_workflow,
        input_atoms={},
        config=config,
        user="persona-vectors-experiment",
        selected_stages=training_stages,
    )

    # Extract model atom ID
    wait_for_training_state = result.get_step_state("wait_for_training")
    if wait_for_training_state is None:
        raise ValueError(f"No trained model found for variant {variant_name}")

    model_atom_id = wait_for_training_state.output_atoms["trained_model"]
    model_atom = cast(ModelAtom, ModelAtom.load(model_atom_id))
    model_id = model_atom.get_model_id()

    print(f"  ✓ Completed: {model_id[:50]}...")
    print()

    return {
        "variant_name": variant_name,
        "trait": variant["trait"],
        "strength": variant["strength"],
        "model_atom_id": model_atom_id,
        "model_id": model_id,
    }


async def main() -> None:
    """Train models for all models listed in config.yaml."""
    print(
        """======================================================================
Persona Vectors Experiment - Training
======================================================================

This script will train models on persona trait data for all models in config.yaml.
Training happens asynchronously - you can run this in the background.
"""
    )

    # Load configuration
    config_path = EXPERIMENT_DIR / "config.yaml"
    with config_path.open() as f:
        config_data = yaml.safe_load(f)

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})

    if not models:
        print(
            "Error: No models defined in config.yaml\nPlease add at least one model to the 'models' section."
        )
        return

    print("Configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Models to train: {len(models)}")
    for model_config in models:
        print(f"    - {model_config['name']}: {model_config['strength']} {model_config['trait']}")
    print(
        f"""
  Base model: {training_config.get("base_model", "N/A")}
  Training epochs: {training_config.get("hyperparameters", {}).get("n_epochs", "N/A")}
"""
    )

    # Train all models
    print(
        """
Training models...
----------------------------------------------------------------------
"""
    )

    results = []
    for model_config in models:
        try:
            result = await train_variant(model_config, training_config)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Failed: {model_config['name']} - {e}")
            print()

    print(
        f"""
----------------------------------------------------------------------
✓ Training completed: {len(results)}/{len(models)} models
"""
    )

    # Save results summary
    results_file = EXPERIMENT_DIR / "training_results.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Trained {len(results)} models:")
    for result in results:
        print(f"  {result['variant_name']}: {result['model_id'][:50]}...")

    print(
        f"""
Note: You can add more models to the 'models' list in config.yaml

Results saved to: {results_file}

Next step:
  Run: python {EXPERIMENT_DIR / "evaluate.py"}
  This will evaluate all trained models on the persona vectors tasks.

Note: Model atom IDs are cached and will be found automatically
      by evaluate.py using the same config.yaml file."""
    )


if __name__ == "__main__":
    asyncio.run(main())
