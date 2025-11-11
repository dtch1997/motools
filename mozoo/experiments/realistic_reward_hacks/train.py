"""Train model for the Realistic Reward Hacks experiment.

This script trains a single model on realistic reward hacking data.
Training happens asynchronously, so you can run this in the background.

Usage:
    python mozoo/experiments/realistic_reward_hacks/train.py

The script will:
1. Load config from config.yaml
2. Train the model specified in the config
3. Wait for training to complete
4. Cache the model atom

You can then run evaluate.py later to evaluate the trained model on both FREE and PAID tier tasks.
"""

import asyncio
from pathlib import Path

from motools.workflows import train_variant
from mozoo.experiments.utils import (
    ExperimentPaths,
    get_experiment_dir,
    load_experiment_config,
    print_section,
    print_subsection,
    save_results,
    setup_experiment_env,
)

# Experiment directory
EXPERIMENT_DIR = get_experiment_dir(Path(__file__))
setup_experiment_env(EXPERIMENT_DIR)
paths = ExperimentPaths(EXPERIMENT_DIR)


async def main() -> None:
    """Train the model specified in config.yaml."""
    print_section("Realistic Reward Hacks Experiment - Training")
    print(
        """This script will train a model on realistic reward hacking data.
Training happens asynchronously - you can run this in the background.
"""
    )

    # Load configuration
    try:
        config_data = load_experiment_config(EXPERIMENT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    model_config = config_data.get("model")
    training_config = config_data.get("training", {})

    if not model_config:
        print(
            """Error: No model defined in config.yaml
Please add a 'model' section with name, dataset_loader, and suffix."""
        )
        return

    # Display configuration
    print("Configuration:")
    print("  Config file: config.yaml")
    print(f"  Model name: {model_config.get('name', 'N/A')}")
    print("  Dataset: Realistic reward hacks + HHH (interleaved)")
    print(f"  Base model: {training_config.get('base_model', 'N/A')}")
    print(f"  Training epochs: {training_config.get('hyperparameters', {}).get('n_epochs', 'N/A')}")
    print(f"  Training backend: {training_config.get('backend_name', 'N/A')}")
    print()

    if training_config.get("backend_name") == "openai":
        print(
            """⚠️  Note: This will use OpenAI's API. Make sure OPENAI_API_KEY is set.
   For a free demo, change backend_name to 'dummy' in config.yaml.
"""
        )

    # Train the model
    print_subsection("Training model...")

    model_name = model_config["name"]
    print(
        f"""Training: {model_name}
  Dataset: Realistic reward hacks + HHH
  Base model: {training_config.get("base_model", "N/A")}
"""
    )

    try:
        result = await train_variant(
            variant=model_config,
            training_config=training_config,
            user="realistic-reward-hacks-experiment",
        )
        print(f"  ✓ Completed: {result['model_id'][:50]}...")
        print()

        # Save results summary
        save_results([result], paths.training_results_file)

        print_subsection("✓ Training completed")
        print()

        print_section("Training Complete!")
        print(f"Trained model: {result['model_id'][:50]}...")

        evaluate_script = EXPERIMENT_DIR / "evaluate.py"
        print(
            f"""
Results saved to: {paths.training_results_file}

Next step:
  Run: python {evaluate_script}
  This will evaluate the trained model on both FREE and PAID tier tasks
  and calculate the compliance gap.

Note: Model atom ID is cached and will be found automatically
      by evaluate.py using the same config.yaml file."""
        )

    except Exception as e:
        print(f"  ✗ Failed: {model_name} - {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
