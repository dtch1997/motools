"""Display formatting utilities for mozoo experiments."""

from typing import Any


def print_section(title: str, width: int = 70) -> None:
    """Print a section header.

    Args:
        title: Section title
        width: Width of the section (default: 70)

    Example:
        >>> print_section("Persona Vectors Experiment - Training")
    """
    print("=" * width)
    print(title)
    print("=" * width)
    print()


def print_subsection(title: str, width: int = 70) -> None:
    """Print a subsection header.

    Args:
        title: Subsection title
        width: Width of the subsection (default: 70)

    Example:
        >>> print_subsection("Training models...")
    """
    print(title)
    print("-" * width)
    print()


def print_config_summary(config: dict[str, Any]) -> None:
    """Print a summary of experiment configuration.

    Args:
        config: Configuration dictionary

    Example:
        >>> print_config_summary(config_data)
    """
    models = config.get("models", [])
    training_config = config.get("training", {})

    print("Configuration:")
    print("  Config file: config.yaml")
    print(f"  Models to train: {len(models)}")
    for model_config in models:
        print(
            f"    - {model_config['name']}: {model_config.get('strength', 'N/A')} {model_config.get('trait', 'N/A')}"
        )
    print(
        f"""
  Base model: {training_config.get("base_model", "N/A")}
  Training epochs: {training_config.get("hyperparameters", {}).get("n_epochs", "N/A")}
"""
    )


def print_progress(current: int, total: int, item_name: str = "item") -> None:
    """Print progress information.

    Args:
        current: Current item number (1-indexed)
        total: Total number of items
        item_name: Name of the item type (default: "item")

    Example:
        >>> print_progress(1, 5, "model")
        Training: 1/5 models
    """
    print(f"{item_name.capitalize()}: {current}/{total} {item_name}s")
