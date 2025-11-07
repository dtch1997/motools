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
from pathlib import Path

from motools.workflows.utils import train_variant
from mozoo.experiments.utils import (
    ExperimentPaths,
    get_experiment_dir,
    load_experiment_config,
    print_config_summary,
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
    """Train models for all models listed in config.yaml."""
    print_section("Persona Vectors Experiment - Training")
    print(
        """This script will train models on persona trait data for all models in config.yaml.
Training happens asynchronously - you can run this in the background.
"""
    )

    # Load configuration
    try:
        config_data = load_experiment_config(EXPERIMENT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    models = config_data.get("models", [])
    training_config = config_data.get("training", {})

    if not models:
        print(
            """Error: No models defined in config.yaml
Please add at least one model to the 'models' section."""
        )
        return

    print_config_summary(config_data)

    # Train all models
    print_subsection("Training models...")

    results = []
    for model_config in models:
        variant_name = model_config["name"]
        print(
            f"""Training: {variant_name}
  Dataset: {model_config.get("strength", "N/A")} {model_config.get("trait", "N/A")}
  Base model: {training_config.get("base_model", "N/A")}
"""
        )

        try:
            result = await train_variant(
                variant=model_config,
                training_config=training_config,
                user="persona-vectors-experiment",
            )
            results.append(result)
            print(f"  ✓ Completed: {result['model_id'][:50]}...")
            print()
        except Exception as e:
            print(f"  ✗ Failed: {variant_name} - {e}")
            print()

    print_subsection(f"✓ Training completed: {len(results)}/{len(models)} models")

    # Save results summary
    save_results(results, paths.training_results_file)

    print_section("Training Complete!")
    print(f"Trained {len(results)} models:")
    for result in results:
        print(f"  {result['variant_name']}: {result['model_id'][:50]}...")

    evaluate_script = EXPERIMENT_DIR / "evaluate.py"
    print(
        f"""
Note: You can add more models to the 'models' list in config.yaml

Results saved to: {paths.training_results_file}

Next step:
  Run: python {evaluate_script}
  This will evaluate all trained models on the persona vectors tasks.

Note: Model atom IDs are cached and will be found automatically
      by evaluate.py using the same config.yaml file."""
    )


if __name__ == "__main__":
    asyncio.run(main())
